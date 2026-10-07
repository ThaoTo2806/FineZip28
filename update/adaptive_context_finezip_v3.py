"""
Adaptive Context FineZip  -  PHÁC THẢO (Giai đoạn 4)
=====================================================

Ý tưởng
-------
FineZip chia token thành các chunk cố định (32/64/.../512). Token thứ i trong chunk
chỉ thấy i-1 token trước nó TRONG CHUNK, nên các chunk độc lập -> batch song song được
khi nén lẫn giải nén. Ở đây ta cho độ dài chunk THAY ĐỔI theo từng vùng văn bản:

    * Cây nhị phân căn theo khối 512 token; mỗi nút là một chunk độ dài L ∈ {32..512}.
    * Mỗi nút chọn: "giữ nguyên làm 1 chunk dài L"  hay  "tách đôi thành 2 chunk L/2".
    * Cách chia (layout) được ghi vào file nén (1 bit/nút, cỡ vài trăm byte / 1 MB)
      => bộ giải nén KHÔNG cần tự đoán entropy => lossless "theo cấu trúc".

Hai chính sách (policy) chọn layout
-----------------------------------
    A) FeaturePolicy : đặc trưng rẻ (entropy, mức lặp) -> ngưỡng. Không tốn thêm LLM pass.
    B) DPPolicy      : quy hoạch động trên cây, minimize  J = bits + lam * thời_gian_ước_lượng.
                       Cần 1 lần teacher-forcing cho mỗi mức L (chi phí chỉ ở phía encode).

Mã hoá: rank-based như FineZip (rank của token thật -> uint16 -> bz2).
v3: file nén ghi lại batch hiệu dụng của từng mức L; decode() đọc từ file chứ KHÔNG dùng tham số batch,
    nên giải nén luôn dùng đúng shape [B, L] như lúc nén (nguyên nhân gốc của lỗi rank lệch).
Chạy thử với MockLM trên CPU:   python adaptive_context_finezip.py
Chạy thật với GPT-2 (Kaggle):   xem hàm demo_gpt2() cuối file.
"""
from __future__ import annotations

import bz2
import hashlib
import json
import math
import os
import struct
import warnings
import zlib
import numpy as np

CTX_SIZES = (32, 64, 128, 256, 512)
MIN_L, MAX_L = CTX_SIZES[0], CTX_SIZES[-1]


# ----------------------------------------------------------------------------
# 1. Mô hình thời gian:  t_chunk(L) = L * (a + b*L)   (giây)
#    Fit từ số đo Kaggle T4 của bạn (FineZip, batch 64, 286 273 token):
#    ctx 32/64/128/256/512 -> 61.6/67.2/83.0/114.3/178.8 s  (gần tuyến tính theo L)
# ----------------------------------------------------------------------------
def fit_time_model(ctxs, times_s, n_tokens):
    A = np.vstack([np.ones(len(ctxs)), np.asarray(ctxs, float)]).T
    (a, b), *_ = np.linalg.lstsq(A, np.asarray(times_s, float) / n_tokens, rcond=None)
    return float(a), float(b)


TIME_A, TIME_B = fit_time_model(CTX_SIZES, [61.63, 67.17, 82.95, 114.33, 178.82], 286273)
FILE_MAGIC = b"FZAC"
FILE_VERSION = 3   # v3: header lưu KẾ HOẠCH BATCH theo L + fingerprint mô hình + thẻ môi trường (xem mục 6)


def est_time(L: int) -> float:
    """Thời gian ước lượng (s) để xử lý 1 chunk dài L (encode, GPT-2/T4)."""
    return L * (TIME_A + TIME_B * L)


# ----------------------------------------------------------------------------
# 2. Giao diện LM.  Chỉ cần 2 hàm -> dễ thay GPT-2 / Llama / mô hình giả để test.
#    score(inp, tgt) : teacher-forcing -> (bits[B,T], rank[B,T]) của token đích
#    pick(inp,pos,r) : token có hạng r tại vị trí pos (dùng khi giải nén)
#    inp = [BOS] + chunk[:-1],  tgt = chunk
#    Quy ước hạng: sắp giảm dần theo logit, hoà thì token id nhỏ đứng trước (stable).
# ----------------------------------------------------------------------------
class HFLM:
    def __init__(self, name="gpt2", device="cuda", max_batch_tokens=1024,
                 eps=None, adapter_path=None, merge_adapter=True):
        import torch
        from transformers import AutoModelForCausalLM
        if max_batch_tokens < 1:
            raise ValueError("max_batch_tokens must be positive")
        self.t, self.device = torch, device
        self.name = name
        self.adapter_path = adapter_path
        self.max_batch_tokens = max_batch_tokens
        self.eps = eps   # ngưỡng "không ổn định" (logit). None = tắt escape. Hiệu chỉnh bằng calibrate_eps().
        self.model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
        if adapter_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path).to(device).eval()
            if merge_adapter:
                self.model = self.model.merge_and_unload().to(device).eval()
        self.bos = self.model.config.bos_token_id  # GPT-2: 50256
        self.V = self.model.config.vocab_size

    def adapter_fingerprint(self):
        if self.adapter_path is None:
            return None
        digest = hashlib.sha256()
        for root, _, names in os.walk(self.adapter_path):
            for name in sorted(names):
                path = os.path.join(root, name)
                digest.update(os.path.relpath(path, self.adapter_path).encode())
                with open(path, "rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
        return digest.hexdigest()

    def batch_size(self, length, requested):
        """Bound the forward batch by the logits tensor size [B, length, V]."""
        return max(1, min(requested, self.max_batch_tokens // length))

    def describe(self):
        """-> (cfg, env). cfg = định danh mô hình (BẮT BUỘC khớp khi giải nén).
        env = phần cứng/thư viện ảnh hưởng số học (chỉ cảnh báo nếu khác)."""
        t = self.t
        cfg = {"name": self.name, "V": int(self.V), "bos": int(self.bos),
               "dtype": str(next(self.model.parameters()).dtype),
             "attn": str(getattr(self.model.config, "_attn_implementation", "unknown")),
             "adapter_sha256": self.adapter_fingerprint()}
        env = {"torch": str(t.__version__), "cuda": str(t.version.cuda),
               "gpu": t.cuda.get_device_name(self.device) if str(self.device).startswith("cuda") else "cpu",
               "tf32": bool(t.backends.cuda.matmul.allow_tf32),
               "matmul_precision": str(t.get_float32_matmul_precision())}
        return cfg, env

    def _logits(self, ids):
        with self.t.no_grad():
            x = self.t.as_tensor(ids, device=self.device)
            return self.model(input_ids=x).logits.float()

    def score(self, inp, tgt):
        """-> bits[B,T], rank[B,T], unstable[B,T].
        unstable = còn token KHÁC có logit cách logit token đích < eps => hạng có thể bị đảo khi
        nhiễu số học (khác batch/GPU) => encoder phải 'escape' vị trí đó."""
        t = self.t
        lg = self._logits(inp)                                   # [B,T,V]  (batch nhỏ: 8-16!)
        y = t.as_tensor(tgt, device=self.device)[..., None]
        tl = lg.gather(-1, y)
        bits = -(t.log_softmax(lg, -1).gather(-1, y).squeeze(-1)) / math.log(2)
        rank = t.zeros(tl.shape[:-1], dtype=t.int64, device=self.device)
        near = t.zeros(tl.shape[:-1], dtype=t.int64, device=self.device)
        # Avoid a second full [B,T,V] comparison tensor on small Kaggle GPUs.
        for start in range(0, lg.shape[-1], 8192):
            stop = min(start + 8192, lg.shape[-1])
            scores = lg[..., start:stop]
            idx = t.arange(start, stop, device=self.device)
            rank += (scores > tl).sum(-1)
            rank += ((scores == tl) & (idx < y)).sum(-1)
            if self.eps is not None:
                near += ((scores - tl).abs() < self.eps).sum(-1)
        unstable = (near > 1) if self.eps is not None else t.zeros_like(near, dtype=t.bool)  # >1: đã tính chính nó
        return bits.cpu().numpy(), rank.cpu().numpy(), unstable.cpu().numpy()

    def pick(self, inp, pos, ranks):
        t = self.t
        lg = self._logits(inp)[:, pos, :]
        order = t.sort(lg, dim=-1, descending=True, stable=True).indices
        r = t.as_tensor(ranks.astype(np.int64), device=self.device)[:, None]
        return order.gather(1, r).squeeze(1).cpu().numpy()


class MockLM:
    """LM giả chạy CPU để test logic: bigram prior + 'copy bonus' (kiểu induction head).
    Vùng lặp dài => cần context dài mới dự đoán tốt; vùng ngẫu nhiên => context không giúp gì."""

    def __init__(self, V=64, seed=0, bonus=6.0, noise=0.0, eps=None, noise_seed=0, max_batch_tokens=1024):
        self.V, self.bos, self.bonus, self.noise, self.eps = V, V - 1, bonus, noise, eps
        self.seed, self.noise_seed, self.max_batch_tokens = seed, noise_seed, max_batch_tokens
        self.prior = np.random.default_rng(seed).normal(0, 1, (V, V)).astype(np.float32)

    def _lg(self, inp, pos):
        lg = self.prior[inp[:, pos]].copy()
        if pos >= 1:                                   # khớp 2-gram (cur, prev) như induction head
            for b in range(inp.shape[0]):
                hit = np.where((inp[b, 1:pos] == inp[b, pos]) & (inp[b, :pos - 1] == inp[b, pos - 1]))[0]
                if len(hit):
                    lg[b, inp[b, hit[-1] + 2]] += self.bonus
        if self.noise:   # mô phỏng nhiễu số học phụ thuộc kích thước batch B (giống GPU thật)
            lg += self.noise * np.random.default_rng(7919 * inp.shape[0] + pos + 104729 * self.noise_seed).standard_normal(lg.shape)
        return lg

    def score(self, inp, tgt):
        B, T = inp.shape
        bits, rank = np.zeros((B, T)), np.zeros((B, T), np.int64)
        unstable = np.zeros((B, T), bool)
        for p in range(T):
            lg = self._lg(inp, p)
            lp = lg - np.log(np.exp(lg - lg.max(1, keepdims=True)).sum(1, keepdims=True)) - lg.max(1, keepdims=True)
            y = tgt[:, p]
            tl = lg[np.arange(B), y]
            bits[:, p] = -lp[np.arange(B), y] / math.log(2)
            idx = np.arange(self.V)[None]
            rank[:, p] = (lg > tl[:, None]).sum(1) + ((lg == tl[:, None]) & (idx < y[:, None])).sum(1)
            if self.eps is not None:
                unstable[:, p] = (np.abs(lg - tl[:, None]) < self.eps).sum(1) > 1
        return bits, rank, unstable

    def pick(self, inp, pos, ranks):
        order = np.argsort(-self._lg(inp, pos), axis=1, kind="stable")
        return order[np.arange(len(ranks)), ranks]

    def batch_size(self, length, requested):
        return max(1, min(requested, self.max_batch_tokens // length))

    def describe(self):
        cfg = {"name": "mock", "V": self.V, "bos": self.bos, "seed": self.seed, "bonus": self.bonus}
        env = {"noise": self.noise, "noise_seed": self.noise_seed}
        return cfg, env


# ----------------------------------------------------------------------------
# 3. Tiện ích chunk
# ----------------------------------------------------------------------------
def pad_tokens(tokens, bos):
    n = len(tokens)
    n_pad = -(-n // MIN_L) * MIN_L
    return np.concatenate([np.asarray(tokens), np.full(n_pad - n, bos, dtype=np.asarray(tokens).dtype)])


def make_batch(toks, starts, L, bos):
    tgt = np.stack([toks[s:s + L] for s in starts])
    inp = np.concatenate([np.full((len(starts), 1), bos, dtype=tgt.dtype), tgt[:, :-1]], axis=1)
    return inp, tgt


def run_score(toks, starts, L, lm, batch, eff=None):
    """-> bits[N], ranks[N,L], unstable[N,L] cho các chunk (start, L).
    eff = batch hiệu dụng ÉP DÙNG (kế hoạch batch lưu trong file). None -> suy ra từ lm.batch_size."""
    bits, ranks, uns = [], [], []
    if eff is None:
        eff = lm.batch_size(L, batch) if hasattr(lm, "batch_size") else batch
    for i in range(0, len(starts), eff):
        inp, tgt = make_batch(toks, starts[i:i + eff], L, lm.bos)
        b, r, u = lm.score(inp, tgt)
        bits.append(b.sum(1)); ranks.append(r); uns.append(u)
    return np.concatenate(bits), np.concatenate(ranks), np.concatenate(uns)


# ----------------------------------------------------------------------------
# 4. Layout = danh sách lá (start, L) + (de)serialize bằng cờ tách đôi (preorder)
# ----------------------------------------------------------------------------
def walk(n_pad, decide):
    """Duyệt cây. decide(s, L) -> True nếu nút (s,L) là LÁ. Trả danh sách lá."""
    leaves = []

    def rec(s, L):
        if s >= n_pad:
            return
        fits = s + L <= n_pad
        if L == MIN_L or (fits and decide(s, L)):
            leaves.append((s, L)); return
        rec(s, L // 2); rec(s + L // 2, L // 2)

    for s in range(0, n_pad, MAX_L):
        rec(s, MAX_L)
    return leaves


def layout_to_flags(leaves, n_pad):
    leafset, out = set(leaves), []

    def rec(s, L):
        if s >= n_pad or L == MIN_L:
            return
        if s + L > n_pad:
            split = True                         # ép tách -> không cần ghi cờ
        else:
            split = (s, L) not in leafset
            out.append(int(split))
        if split:
            rec(s, L // 2); rec(s + L // 2, L // 2)

    for s in range(0, n_pad, MAX_L):
        rec(s, MAX_L)
    return out


def flags_to_layout(flags, n_pad):
    it, leaves = iter(flags), []

    def rec(s, L):
        if s >= n_pad:
            return
        if L == MIN_L:
            leaves.append((s, L)); return
        split = True if s + L > n_pad else bool(next(it))
        if split:
            rec(s, L // 2); rec(s + L // 2, L // 2)
        else:
            leaves.append((s, L))

    for s in range(0, n_pad, MAX_L):
        rec(s, MAX_L)
    return leaves


# ----------------------------------------------------------------------------
# 5. Các chính sách chọn layout:  policy(toks_pad, lm, batch) -> leaves
# ----------------------------------------------------------------------------
class FixedPolicy:
    def __init__(self, L): self.L = L
    def __call__(self, toks, lm, batch):
        return walk(len(toks), lambda s, L: L <= self.L)


class FeaturePolicy:
    """A) Đặc trưng rẻ trên mỗi khối 512 token -> chọn L cho cả khối.
    LƯU Ý: ngưỡng dưới đây chỉ là GIẢ THIẾT khởi đầu. Hãy hiệu chỉnh bằng nhãn từ DPPolicy
    (xem analyze_features) thay vì tin trực giác 'entropy cao => context lớn'."""

    def __init__(self, h_lo=5.0, r_hi=0.30, L_low=32, L_mid=128, L_high=512):
        self.h_lo, self.r_hi = h_lo, r_hi
        self.L_low, self.L_mid, self.L_high = L_low, L_mid, L_high

    @staticmethod
    def features(block):
        _, cnt = np.unique(block, return_counts=True)
        p = cnt / cnt.sum()
        entropy = float(-(p * np.log2(p)).sum())                       # bit/token (unigram)
        big = list(zip(block[:-1].tolist(), block[1:].tolist()))
        repeat = 1.0 - len(set(big)) / max(1, len(big))                # tỉ lệ bigram lặp
        return entropy, repeat

    def choose(self, block):
        h, r = self.features(block)
        if h < self.h_lo:   return self.L_low     # ít đa dạng
        if r > self.r_hi:   return self.L_high    # nhiều lặp -> thử context dài
        return self.L_mid

    def __call__(self, toks, lm, batch):
        pick = {s: self.choose(toks[s:s + MAX_L]) for s in range(0, len(toks), MAX_L)}
        return walk(len(toks), lambda s, L: L <= pick[(s // MAX_L) * MAX_L])


class DPPolicy:
    """B) Tối ưu  J = bits + lam * est_time  trên cây nhị phân.
    lam = số 'bit' ta sẵn sàng trả cho mỗi giây tính toán (với GPT-2/1MB cỡ 10^3..4·10^4: xem cột "bit tiết kiệm/giây thêm" ở sheet GD3). lam=0 -> tối đa nén (≈ toàn 512);
    lam lớn -> chunk nhỏ, nhanh. Quét lam để vẽ đường Pareto rồi so với các điểm Fixed."""

    def __init__(self, lam=0.0, levels=CTX_SIZES):
        if lam < 0:
            raise ValueError("lam must be non-negative")
        self.lam, self.levels = lam, tuple(levels)
        self._bits, self._cache_key = None, None

    def _level_bits(self, toks, lm, batch):
        key = (len(toks), zlib.crc32(np.asarray(toks).tobytes()), id(lm), batch)
        if self._bits is None or self._cache_key != key:
            self._bits = {}
            for L in self.levels:
                starts = list(range(0, len(toks) - L + 1, L))
                self._bits[L] = run_score(toks, starts, L, lm, batch)[0] if starts else np.array([])
            self._cache_key = key
        return self._bits

    def __call__(self, toks, lm, batch):
        n_pad, bits = len(toks), self._level_bits(toks, lm, batch)

        def J(s, L):
            return bits[L][s // L] + self.lam * est_time(L)

        def solve(s, L):
            if s >= n_pad:
                return 0.0, []
            split = None
            if L > MIN_L:
                (j1, l1), (j2, l2) = solve(s, L // 2), solve(s + L // 2, L // 2)
                split = (j1 + j2, l1 + l2)
            if s + L <= n_pad and L in bits:
                leaf = (J(s, L), [(s, L)])
                if split is None or leaf[0] <= split[0]:
                    return leaf
            return split

        leaves = []
        for s in range(0, n_pad, MAX_L):
            leaves += solve(s, MAX_L)[1]
        return leaves


def analyze_features(toks, bits_by_level):
    """Gợi ý hiệu chỉnh policy A: tương quan (Spearman) giữa đặc trưng và 'lợi ích của context dài'
    gain = bits(L=32) - bits(L=512) trên mỗi khối 512 token."""
    rk = lambda x: np.argsort(np.argsort(x)).astype(float)
    nblk = len(bits_by_level[MAX_L])
    gain = np.array([bits_by_level[MIN_L][16 * k:16 * k + 16].sum() - bits_by_level[MAX_L][k] for k in range(nblk)])
    feats = np.array([FeaturePolicy.features(toks[k * MAX_L:(k + 1) * MAX_L]) for k in range(nblk)])
    for j, name in enumerate(["unigram_entropy", "bigram_repeat"]):
        print(f"Spearman(gain, {name}) = {np.corrcoef(rk(gain), rk(feats[:, j]))[0, 1]:+.3f}")
    return gain, feats


# ----------------------------------------------------------------------------
# 6. Encode / Decode   (định dạng v3)
#
#    NGUYÊN NHÂN LỖI RANK LỆCH: logit float32 trên GPU phụ thuộc kích thước batch B (đo được ~1e-2 giữa
#    B=1/2/4 với GPT-2 trên T4). Cùng shape [B,L] thì encode và decode cho logit GIỐNG HỆT (kể cả khi các vị trí
#    sau p là placeholder: đo được Δ = 0). Vì vậy:
#      (1) Header lưu batch hiệu dụng của từng mức L; decode/quick_check LUÔN dùng đúng kế hoạch này
#          (tham số `batch` của decode bị bỏ qua) -> không thể giải nén bằng shape khác lúc nén.
#      (2) Header lưu fingerprint mô hình (tên, vocab, bos, dtype, attention). Khác -> từ chối giải nén.
#      (3) Header lưu thẻ môi trường (torch, CUDA, GPU, TF32). Khác -> cảnh báo: lossless chỉ được đảm bảo
#          trên cùng môi trường, trừ khi file nén có escape (eps>0) đủ lớn so với nhiễu liên-môi-trường.
#      (4) Escape (tuỳ chọn, eps>0): token có logit cách token thật < eps được lưu thẳng thay vì rank.
#
#    Nhóm batch: các chunk cùng L, sắp theo start, cắt liên tiếp mỗi nhóm `plan[L]` chunk
#    (nhóm cuối có thể nhỏ hơn; cả encode lẫn decode cắt giống nhau).
#
#    File v3 = [magic 4][ver u8][rank width u8][reserved u16][n_tokens u32][n_flags u32][n_esc u32]
#              [fingerprint 8][eps f32][plan: batch cho L=32,64,128,256,512 (5×u16)]
#              [env_len u16][env json][flags đóng gói bit][len_ranks u32][bz2(ranks)]
#              [bz2(esc_pos_delta u32 ++ esc_token)]
# ----------------------------------------------------------------------------
_HDR = "<4sBBHIII8sf5H"


def _fingerprint(lm):
    cfg, env = lm.describe()
    fp = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).digest()[:8]
    return fp, cfg, json.dumps(env, sort_keys=True, separators=(",", ":"))


def _read_header(blob):
    hs = struct.calcsize(_HDR)
    if len(blob) < hs + 2:
        raise ValueError("truncated FineZip header")
    magic, ver, rank_width, _, n, nf, n_esc, fp, eps, *plan = struct.unpack_from(_HDR, blob, 0)
    if magic != FILE_MAGIC:
        raise ValueError("not a FineZip-adaptive file")
    if ver != FILE_VERSION:
        raise ValueError("unsupported FineZip file version %d (cần v%d: hãy nén lại bằng encode() mới)" % (ver, FILE_VERSION))
    if any(b < 1 or b > 4096 for b in plan):
        raise ValueError("invalid batch plan in header")
    (env_len,) = struct.unpack_from("<H", blob, hs)
    env = blob[hs + 2:hs + 2 + env_len].decode("utf-8")
    return dict(rank_width=rank_width, n=n, nf=nf, n_esc=n_esc, fp=fp, eps=eps,
                plan=dict(zip(CTX_SIZES, plan)), env=env, off=hs + 2 + env_len)


def encode(tokens, lm, policy, batch=8):
    toks = pad_tokens(tokens, lm.bos)
    leaves = policy(toks, lm, batch)
    plan = {L: int(lm.batch_size(L, batch)) if hasattr(lm, "batch_size") else int(batch) for L in CTX_SIZES}
    rank_of, esc_pos, esc_tok = {}, [], []
    for L in sorted({l for _, l in leaves}):
        starts = [s for s, l in leaves if l == L]
        _, r, u = run_score(toks, starts, L, lm, batch, eff=plan[L])      # ép dùng đúng kế hoạch đã ghi
        for i, s in enumerate(starts):
            ri = r[i].copy()
            ri[u[i]] = 0                                   # vị trí escape: rank không dùng -> 0 cho dễ nén
            rank_of[s] = ri
            idx = np.nonzero(u[i])[0]
            esc_pos += (s + idx).tolist(); esc_tok += toks[s + idx].tolist()
    order = np.argsort(esc_pos, kind="stable") if esc_pos else np.empty(0, int)
    esc_pos = np.asarray(esc_pos, np.int64)[order]; esc_tok = np.asarray(esc_tok, np.int64)[order]
    rank_width = 2 if lm.V <= np.iinfo(np.uint16).max + 1 else 4
    rank_dtype = "<u2" if rank_width == 2 else "<u4"
    ranks = (np.concatenate([rank_of[s] for s, _ in leaves]).astype(rank_dtype)
             if leaves else np.empty(0, dtype=rank_dtype))
    flags = layout_to_flags(leaves, len(toks))
    delta = np.diff(esc_pos, prepend=0).astype("<u4")
    esc_blob = bz2.compress(delta.tobytes() + esc_tok.astype(rank_dtype).tobytes(), 9) if len(esc_pos) else b""
    rk_blob = bz2.compress(ranks.tobytes(), 9)
    fp, _, env = _fingerprint(lm)
    eps = float(getattr(lm, "eps", None) or 0.0)
    envb = env.encode("utf-8")
    head = struct.pack(_HDR, FILE_MAGIC, FILE_VERSION, rank_width, 0, len(tokens), len(flags), len(esc_pos),
                       fp, eps, *[plan[L] for L in CTX_SIZES])
    head += struct.pack("<H", len(envb)) + envb
    head += np.packbits(np.array(flags, np.uint8)).tobytes() + struct.pack("<I", len(rk_blob))
    encode.last_stats = {"escapes": int(len(esc_pos)), "escape_pct": 100.0 * len(esc_pos) / max(1, len(toks)),
                         "esc_bytes": len(esc_blob), "total_bytes": len(head) + len(rk_blob) + len(esc_blob),
                         "header_bytes": len(head), "batch_plan": plan, "eps": eps}
    return head + rk_blob + esc_blob, leaves


def _parse(blob, lm):
    h = _read_header(blob)
    fp, cfg, env_now = _fingerprint(lm)
    if fp != h["fp"]:
        raise ValueError("mô hình khác lúc nén (tên/vocab/bos/dtype/attention). Cấu hình hiện tại: %s" % json.dumps(cfg))
    if h["rank_width"] != (2 if lm.V <= np.iinfo(np.uint16).max + 1 else 4):
        raise ValueError("rank width does not match the decoder vocabulary")
    if env_now != h["env"]:
        try:
            a, b = json.loads(h["env"]), json.loads(env_now)
            diff = {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
        except ValueError:
            diff = {"env": (h["env"], env_now)}
        guard = "file có escape (eps=%.3g) nên có thể vẫn đúng" % h["eps"] if h["n_esc"] else \
                "file KHÔNG có escape nên lossless không được đảm bảo"
        warnings.warn("môi trường giải nén khác lúc nén %s: %s" % (diff, guard), RuntimeWarning, stacklevel=3)
    n, nf, n_esc = h["n"], h["nf"], h["n_esc"]
    nb = (nf + 7) // 8
    flags = np.unpackbits(np.frombuffer(blob, np.uint8, nb, h["off"]))[:nf]
    n_pad = -(-n // MIN_L) * MIN_L
    try:
        leaves = flags_to_layout(flags.tolist(), n_pad)
    except (StopIteration, ValueError) as exc:
        raise ValueError("invalid FineZip layout") from exc
    off = h["off"] + nb
    (len_rk,) = struct.unpack_from("<I", blob, off); off += 4
    dt = np.dtype("<u2" if h["rank_width"] == 2 else "<u4")
    ranks = np.frombuffer(bz2.decompress(blob[off:off + len_rk]), dtype=dt).astype(np.int64)
    if len(ranks) != sum(L for _, L in leaves):
        raise ValueError("rank count does not match layout")
    esc = {}
    if n_esc:
        raw = bz2.decompress(blob[off + len_rk:])
        pos = np.cumsum(np.frombuffer(raw[:4 * n_esc], "<u4").astype(np.int64))
        tok = np.frombuffer(raw[4 * n_esc:], dtype=dt).astype(np.int64)
        esc = dict(zip(pos.tolist(), tok.tolist()))
    rk, o = {}, 0
    for s, L in leaves:
        rk[s] = ranks[o:o + L]; o += L
    return n, n_pad, leaves, rk, esc, h["plan"]


def _groups(leaves, plan):
    """Các nhóm batch theo đúng kế hoạch trong file: (L, [start,...])."""
    for L in sorted({l for _, l in leaves}):
        starts = [s for s, l in leaves if l == L]
        for i in range(0, len(starts), plan[L]):
            yield L, starts[i:i + plan[L]]


def _decode_group(ss, L, rk, esc, lm):
    R = np.stack([rk[s] for s in ss])
    inp = np.full((len(ss), L), lm.bos, np.int64)           # cùng shape [B,L] như encode
    dec = np.zeros((len(ss), L), np.int64)
    for p in range(L):                                        # tự hồi quy trong chunk, batch qua chunk
        dec[:, p] = lm.pick(inp, p, R[:, p])
        for j, s in enumerate(ss):                            # vị trí escape: dùng token đã lưu
            g = s + p
            if g in esc:
                dec[j, p] = esc[g]
        if p + 1 < L:
            inp[:, p + 1] = dec[:, p]
    return dec


def decode(blob, lm, batch=None):
    """Giải nén. `batch` được GIỮ để tương thích nhưng BỊ BỎ QUA: kế hoạch batch lấy từ header."""
    n, n_pad, leaves, rk, esc, plan = _parse(blob, lm)
    out = np.zeros(n_pad, np.int64)
    for L, ss in _groups(leaves, plan):
        dec = _decode_group(ss, L, rk, esc, lm)
        for j, s in enumerate(ss):
            out[s:s + L] = dec[j]
    return out[:n]


def quick_check(blob, lm, tokens, batch=None, groups_per_L=1):
    """Giải nén THỬ vài nhóm batch đầu của mỗi mức L và so với token gốc (dùng đúng kế hoạch batch trong file).
    Chạy TRƯỚC decode đầy đủ (hàng giờ). `batch` bị bỏ qua như ở decode()."""
    tokens = np.asarray(tokens)
    n, n_pad, leaves, rk, esc, plan = _parse(blob, lm)
    seen, bad = {}, []
    for L, ss in _groups(leaves, plan):
        if seen.get(L, 0) >= groups_per_L:
            continue
        seen[L] = seen.get(L, 0) + 1
        dec = _decode_group(ss, L, rk, esc, lm)
        for j, s in enumerate(ss):
            e = min(s + L, n)
            if e > s and not np.array_equal(dec[j][:e - s], tokens[s:e]):
                bad.append((s, L))
    print("quick_check:", "OK" if not bad else f"LỆCH ở {bad}")
    return not bad


def inspect_blob(blob):
    """Đọc header + layout, không cần mô hình."""
    h = _read_header(blob)
    n_pad = -(-h["n"] // MIN_L) * MIN_L
    nb = (h["nf"] + 7) // 8
    flags = np.unpackbits(np.frombuffer(blob, np.uint8, nb, h["off"]))[:h["nf"]]
    leaves = flags_to_layout(flags.tolist(), n_pad)
    return {"version": FILE_VERSION, "n_tokens": h["n"], "eps": h["eps"], "n_escapes": h["n_esc"],
            "batch_plan": h["plan"], "chunk_counts": {L: sum(1 for _, l in leaves if l == L) for L in CTX_SIZES},
            "env": json.loads(h["env"]), "bytes": len(blob)}


def calibrate_eps(lm, toks, L=512, batches=(1, 2, 4, 8), n_chunks=8,
                  margin=10.0, percentile=99.99):
    """Diagnose batch noise and return a robust epsilon for rank escapes.

    The placeholder comparison is reported separately because it checks causal
    masking, not numerical noise. It must not determine eps: using its maximum
    can turn a single outlier into escapes for a large fraction of the file.
    """
    toks = np.asarray(toks)
    starts = [i * L for i in range(min(n_chunks, len(toks) // L))]
    inp, _ = make_batch(toks, starts, L, lm.bos)

    ref = np.concatenate([lm._logits(inp[i:i + 1]).cpu().numpy() for i in range(len(inp))])   # B=1
    diffs = []
    for b in batches:
        got = np.concatenate([lm._logits(inp[i:i + b]).cpu().numpy() for i in range(0, len(inp), b)])
        delta = np.abs(got - ref)
        diffs.append(delta)
        print(f"  B={b}: max|delta|={delta.max():.3e}, p{percentile:g}={np.percentile(delta, percentile):.3e}")

    all_delta = np.concatenate(diffs, axis=0)
    pos0 = all_delta[:, 0].ravel()
    pos_after_bos = all_delta[:, 1:].ravel()
    tail = float(np.percentile(pos_after_bos, percentile))
    print(f"  vị trí 0: max={pos0.max():.3e}, p{percentile:g}={np.percentile(pos0, percentile):.3e}")
    print(f"  vị trí >=1: max={pos_after_bos.max():.3e}, p{percentile:g}={tail:.3e}")

    # input kiểu decoder: BOS placeholder ở các vị trí sau p
    p = L // 2
    dec_inp = inp.copy(); dec_inp[:, p + 1:] = lm.bos
    placeholder_delta = np.abs(
        lm._logits(dec_inp)[:, p].cpu().numpy()
        - lm._logits(inp)[:, p].cpu().numpy()
    )
    print(f"  placeholder-vs-thật @p={p}: max={placeholder_delta.max():.3e}, "
          f"p{percentile:g}={np.percentile(placeholder_delta, percentile):.3e}")

    eps = margin * max(tail, 1e-7)
    print(f"=> eps khuyến nghị (không dùng placeholder/max outlier) = {eps:.3e} "
          f"(margin x{margin:g})")
    return eps


# ----------------------------------------------------------------------------
# 7. Test trên CPU với MockLM  (kiểm tra: cây/layout, DP, round-trip lossless)
# ----------------------------------------------------------------------------
def _synthetic(V, seed=1):
    rng = np.random.default_rng(seed)
    motif = rng.integers(0, V, 200)
    rand = lambda n: rng.integers(0, V, n)
    rep = lambda n: np.tile(motif, n // 200 + 1)[:n]
    return np.concatenate([rand(1500), rep(1500), rand(1500), rep(1500), rand(37)])  # 6037 token, đuôi lẻ


def demo_mock():
    lm = MockLM(V=512, bonus=14.0)
    toks = _synthetic(lm.bos)
    print(f"{len(toks)} token  | time model: t(L)=L*({TIME_A:.2e}+{TIME_B:.2e}*L) s")
    rows = [(f"Fixed-{L}", FixedPolicy(L)) for L in CTX_SIZES]
    dp = DPPolicy()
    rows += [(f"Adaptive-DP lam={lam:g}", dp) for lam in (0.0, 1e3, 3e3, 1e4, 3e4, 1e5)]
    print(f"{'policy':24s} {'bytes':>7s} {'est_time_s':>10s}  leaf mix  lossless")
    for name, pol in rows:
        if isinstance(pol, DPPolicy):
            pol.lam = float(name.split("=")[1])
        blob, leaves = encode(toks, lm, pol, batch=8)
        ok = np.array_equal(decode(blob, lm, batch=8), toks)
        mix = {L: sum(1 for _, l in leaves if l == L) for L in CTX_SIZES}
        print(f"{name:24s} {len(blob):7d} {sum(est_time(l) for _, l in leaves):10.4f}  "
              f"{[v for v in mix.values()]}  {ok}")
        assert ok


# ----------------------------------------------------------------------------
# 8. Chạy thật trên Kaggle (GPT-2 + 1 MB enwik8)
# ----------------------------------------------------------------------------
def demo_gpt2(path="/kaggle/working/FineZip28/notebook/finezip_experiment3.5/data/baseline_1mb.txt"):
    import time
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    # The file is deliberately tokenized in full; model calls remain chunked.
    tok.model_max_length = 10**9
    text = open(path, encoding="utf-8").read()
    tokens = np.array(tok(text)["input_ids"], dtype=np.int64)
    lm = HFLM("gpt2")
    dp = DPPolicy()
    for name, pol in [("Fixed-128", FixedPolicy(128)), ("Fixed-512", FixedPolicy(512))] + \
                     [(f"DP lam={l}", dp) for l in (1e3, 2e3, 5e3, 1.2e4, 4e4)]:
        if isinstance(pol, DPPolicy):
            pol.lam = float(name.split("=")[1])
        t0 = time.time(); blob, leaves = encode(tokens, lm, pol, batch=8); te = time.time() - t0
        t0 = time.time(); rec = decode(blob, lm, batch=8);                  td = time.time() - t0
        print(name, len(blob), f"ratio={len(text.encode()) / len(blob):.3f}",
              f"enc={te:.0f}s dec={td:.0f}s", "lossless=", np.array_equal(rec, tokens))


def demo_noise():
    toks = _synthetic(511)[:2000]
    mk = lambda **kw: MockLM(V=512, bonus=14.0, noise=1e-3, **kw)

    print("\n--- 1) Đổi tham số batch lúc giải nén: header giữ kế hoạch batch, kết quả không đổi ---")
    lm = mk()
    blob, _ = encode(toks, lm, FixedPolicy(256), batch=8)
    for b in (8, 3, 1):
        print(f"decode(batch={b}) lossless =", np.array_equal(decode(blob, lm, batch=b), toks))
    print("header:", {k: v for k, v in inspect_blob(blob).items() if k in ("batch_plan", "eps", "n_escapes", "chunk_counts")})

    print("\n--- 2) Giải nén ở MÔI TRƯỜNG KHÁC (nhiễu khác): chỉ escape mới cứu ---")
    for eps in (None, 0.01):
        blob, _ = encode(toks, mk(noise_seed=1, eps=eps), FixedPolicy(256), batch=8)
        st = encode.last_stats
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rec = decode(blob, mk(noise_seed=2))
        print(f"eps={eps}: lossless={np.array_equal(rec, toks)}  bytes={len(blob)}  escapes={st['escapes']} "
              f"({st['escape_pct']:.1f}%)  cảnh báo môi trường={bool(w)}")

    print("\n--- 3) Sai mô hình: phải bị từ chối ---")
    try:
        decode(blob, MockLM(V=256, bonus=14.0, noise=1e-3))
        print("KHÔNG bị từ chối (LỖI)")
    except ValueError as e:
        print("từ chối đúng:", str(e)[:70])


if __name__ == "__main__":
    demo_mock()
    demo_noise()
