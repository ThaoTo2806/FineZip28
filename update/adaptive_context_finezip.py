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
Chạy thử với MockLM trên CPU:   python adaptive_context_finezip.py
Chạy thật với GPT-2 (Kaggle):   xem hàm demo_gpt2() cuối file.
"""
from __future__ import annotations

import bz2
import math
import struct
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
FILE_VERSION = 1


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
    def __init__(self, name="gpt2", device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM
        self.t, self.device = torch, device
        self.model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
        self.bos = self.model.config.bos_token_id  # GPT-2: 50256
        self.V = self.model.config.vocab_size

    def _logits(self, ids):
        with self.t.no_grad():
            x = self.t.as_tensor(ids, device=self.device)
            return self.model(input_ids=x).logits.float()

    def score(self, inp, tgt):
        t = self.t
        lg = self._logits(inp)                                   # [B,T,V]  (batch nhỏ: 8-16!)
        y = t.as_tensor(tgt, device=self.device)[..., None]
        tl = lg.gather(-1, y)
        bits = -(t.log_softmax(lg, -1).gather(-1, y).squeeze(-1)) / math.log(2)
        idx = t.arange(lg.shape[-1], device=self.device)
        rank = (lg > tl).sum(-1) + ((lg == tl) & (idx < y)).sum(-1)
        return bits.cpu().numpy(), rank.cpu().numpy()

    def pick(self, inp, pos, ranks):
        t = self.t
        lg = self._logits(inp)[:, pos, :]
        order = t.sort(lg, dim=-1, descending=True, stable=True).indices
        r = t.as_tensor(ranks.astype(np.int64), device=self.device)[:, None]
        return order.gather(1, r).squeeze(1).cpu().numpy()


class MockLM:
    """LM giả chạy CPU để test logic: bigram prior + 'copy bonus' (kiểu induction head).
    Vùng lặp dài => cần context dài mới dự đoán tốt; vùng ngẫu nhiên => context không giúp gì."""

    def __init__(self, V=64, seed=0, bonus=6.0):
        self.V, self.bos, self.bonus = V, V - 1, bonus
        self.prior = np.random.default_rng(seed).normal(0, 1, (V, V)).astype(np.float32)

    def _lg(self, inp, pos):
        lg = self.prior[inp[:, pos]].copy()
        if pos >= 1:                                   # khớp 2-gram (cur, prev) như induction head
            for b in range(inp.shape[0]):
                hit = np.where((inp[b, 1:pos] == inp[b, pos]) & (inp[b, :pos - 1] == inp[b, pos - 1]))[0]
                if len(hit):
                    lg[b, inp[b, hit[-1] + 2]] += self.bonus
        return lg

    def score(self, inp, tgt):
        B, T = inp.shape
        bits, rank = np.zeros((B, T)), np.zeros((B, T), np.int64)
        for p in range(T):
            lg = self._lg(inp, p)
            lp = lg - np.log(np.exp(lg - lg.max(1, keepdims=True)).sum(1, keepdims=True)) - lg.max(1, keepdims=True)
            y = tgt[:, p]
            tl = lg[np.arange(B), y]
            bits[:, p] = -lp[np.arange(B), y] / math.log(2)
            idx = np.arange(self.V)[None]
            rank[:, p] = (lg > tl[:, None]).sum(1) + ((lg == tl[:, None]) & (idx < y[:, None])).sum(1)
        return bits, rank

    def pick(self, inp, pos, ranks):
        order = np.argsort(-self._lg(inp, pos), axis=1, kind="stable")
        return order[np.arange(len(ranks)), ranks]


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


def run_score(toks, starts, L, lm, batch):
    """-> bits[N], ranks[N,L] cho các chunk (start, L)."""
    bits, ranks = [], []
    for i in range(0, len(starts), batch):
        inp, tgt = make_batch(toks, starts[i:i + batch], L, lm.bos)
        b, r = lm.score(inp, tgt)
        bits.append(b.sum(1)); ranks.append(r)
    return np.concatenate(bits), np.concatenate(ranks)


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
# 6. Encode / Decode
#    File = [magic 4][version u8][rank width u8][reserved u16][n_tokens u32]
#           [n_flags u32][flags đóng gói bit][bz2(ranks)]
# ----------------------------------------------------------------------------
def encode(tokens, lm, policy, batch=8):
    toks = pad_tokens(tokens, lm.bos)
    leaves = policy(toks, lm, batch)
    rank_of = {}
    for L in sorted({l for _, l in leaves}):
        starts = [s for s, l in leaves if l == L]
        _, r = run_score(toks, starts, L, lm, batch)
        rank_of.update({s: r[i] for i, s in enumerate(starts)})
    rank_width = 2 if lm.V <= np.iinfo(np.uint16).max + 1 else 4
    rank_dtype = "<u2" if rank_width == 2 else "<u4"
    if leaves:
        ranks = np.concatenate([rank_of[s] for s, _ in leaves]).astype(rank_dtype)
    else:
        ranks = np.empty(0, dtype=rank_dtype)
    flags = layout_to_flags(leaves, len(toks))
    head = struct.pack("<4sBBHII", FILE_MAGIC, FILE_VERSION, rank_width, 0,
                       len(tokens), len(flags))
    head += np.packbits(np.array(flags, np.uint8)).tobytes()
    return head + bz2.compress(ranks.tobytes(), 9), leaves


def decode(blob, lm, batch=8):
    header_size = struct.calcsize("<4sBBHII")
    if len(blob) < header_size:
        raise ValueError("truncated FineZip header")
    magic, version, rank_width, _, n, nf = struct.unpack_from("<4sBBHII", blob, 0)
    if magic != FILE_MAGIC or version != FILE_VERSION:
        raise ValueError("unsupported FineZip file format")
    expected_width = 2 if lm.V <= np.iinfo(np.uint16).max + 1 else 4
    if rank_width != expected_width:
        raise ValueError("rank width does not match the decoder vocabulary")
    nb = (nf + 7) // 8
    if len(blob) < header_size + nb:
        raise ValueError("truncated FineZip layout")
    flags = np.unpackbits(np.frombuffer(blob, np.uint8, nb, header_size))[:nf]
    n_pad = -(-n // MIN_L) * MIN_L
    try:
        leaves = flags_to_layout(flags.tolist(), n_pad)
    except (StopIteration, ValueError) as exc:
        raise ValueError("invalid FineZip layout") from exc
    rank_dtype = "<u2" if rank_width == 2 else "<u4"
    try:
        raw_ranks = bz2.decompress(blob[header_size + nb:])
    except OSError as exc:
        raise ValueError("invalid FineZip rank stream") from exc
    item_size = np.dtype(rank_dtype).itemsize
    if len(raw_ranks) % item_size:
        raise ValueError("corrupt FineZip rank stream")
    ranks = np.frombuffer(raw_ranks, dtype=rank_dtype).astype(np.int64)
    expected_ranks = sum(L for _, L in leaves)
    if len(ranks) != expected_ranks:
        raise ValueError("rank count does not match layout")
    off, rk = 0, {}
    for s, L in leaves:
        rk[s] = ranks[off:off + L]; off += L
    out = np.zeros(n_pad, np.int64)
    for L in sorted({l for _, l in leaves}):
        starts = [s for s, l in leaves if l == L]
        for i in range(0, len(starts), batch):
            ss = starts[i:i + batch]
            R = np.stack([rk[s] for s in ss])
            inp = np.full((len(ss), L), lm.bos, np.int64)       # LUÔN cùng shape [B,L] như encode
            dec = np.zeros((len(ss), L), np.int64)
            for p in range(L):                                   # tự hồi quy trong chunk, batch qua chunk
                dec[:, p] = lm.pick(inp, p, R[:, p])
                if p + 1 < L:
                    inp[:, p + 1] = dec[:, p]
            for j, s in enumerate(ss):
                out[s:s + L] = dec[j]
    return out[:n]


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
def demo_gpt2(path="/kaggle/working/finezip_experiment/data/baseline_1mb.txt"):
    import time
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
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


if __name__ == "__main__":
    demo_mock()
