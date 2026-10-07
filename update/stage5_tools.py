"""
Công cụ Giai đoạn 5 cho Adaptive Context FineZip (codec v3).  Đặt cùng thư mục với adaptive_context_finezip_v3.py

Ý chính: phần ĐẮT (chấm điểm LLM cho 5 mức L, ~135 s trên T4) chỉ làm MỘT lần -> analyze().
Mọi thứ sau đó (DP, đổi mục tiêu, quét λ, khớp L̄, học ngưỡng FeaturePolicy) chạy trên CPU với số liệu đã lưu,
nên thử nhiều cấu hình gần như miễn phí; GPU chỉ cần cho lượt encode cuối (~27 s) + quick_check.

  analyze()              -> A[L] = {"bits": [N], "ranks": [N, L]}
  cost_ac / cost_calibrated / cost_rank_proxy   -> 3 mục tiêu cho DP (H1: bit AC vs rank+bz2)
  TreeDP(cost, lam)      -> policy DP tổng quát (mục tiêu + chi phí giải nén tuỳ chọn)
  lam_for_Lbar()         -> tìm λ để L̄ khớp mục tiêu (so sánh ĐÚNG ở cùng chi phí giải nén)
  fit_cuts / ThresholdPolicy -> FeaturePolicy học ngưỡng từ nhãn của DP (entropy | repeat | llm32)
  report()               -> bảng so với đường Fixed (nội suy bậc 2 + chia sẻ thời gian)
"""
from __future__ import annotations
import numpy as np
import adaptive_context_finezip_v3 as fz

LEVELS = fz.CTX_SIZES
BLOCK = fz.MAX_L


# ----------------------------------------------------------------------------- 1. phân tích (đắt, 1 lần)
def analyze(toks_pad, lm, batch=8, levels=LEVELS):
    A = {}
    for L in levels:
        starts = list(range(0, len(toks_pad) - L + 1, L))
        if not starts:
            A[L] = {"bits": np.zeros(0), "ranks": np.zeros((0, L), np.int64)}; continue
        b, r, _ = fz.run_score(toks_pad, starts, L, lm, batch)
        A[L] = {"bits": b, "ranks": r}
    return A


# ----------------------------------------------------------------------------- 2. ba mục tiêu cho DP
def cost_ac(A):
    return {L: A[L]["bits"] for L in A}


def kappa(A, fixed_bytes, header_bytes):
    """κ_L = (bit thật của Fixed-L: (byte - header)·8) / (tổng bit AC mức L). Sửa lệch 'AC lý tưởng' vs 'rank+bz2'."""
    return {L: (fixed_bytes[L] - header_bytes[L]) * 8.0 / A[L]["bits"].sum() for L in A}


def cost_calibrated(A, kap):
    return {L: kap[L] * A[L]["bits"] for L in A}


def cost_rank_proxy(A, ref=128, alpha=0.5, rmax=4096):
    """Mã tĩnh bậc 0 cho rank, dựng từ histogram rank ở mức `ref` (thô hơn bz2 nhưng cùng 'ngôn ngữ rank')."""
    r0 = np.minimum(A[ref]["ranks"].ravel(), rmax)
    cnt = np.bincount(r0, minlength=rmax + 1).astype(float)
    table = -np.log2((cnt + alpha) / (cnt.sum() + alpha * (rmax + 1)))
    out = {}
    for L in A:
        r = A[L]["ranks"]
        tail = np.where(r >= rmax, np.log2(np.maximum(r, 1)), 0.0)        # rank hiếm: + log2(rank) bit cho chỉ số trong đuôi
        out[L] = table[np.minimum(r, rmax)].sum(1) + tail.sum(1)
    return out


# ----------------------------------------------------------------------------- 3. DP tổng quát trên cây chunk
def t_dec_quadratic(c=1.02e-4):
    return lambda L: c * L * L          # giây / chunk. Thay bằng bảng đo thật tdec[L] khi đã hiệu chuẩn.


class TreeDP:
    """J(chunk) = cost[L][idx] + lam * t_chunk(L). Giao diện policy(toks, lm, batch) như fz.DPPolicy."""

    def __init__(self, cost, lam=0.0, t_chunk=None):
        self.cost, self.lam, self.t_chunk = cost, float(lam), t_chunk or t_dec_quadratic()

    def __call__(self, toks, lm=None, batch=None):
        n_pad, cost, lam, t = len(toks), self.cost, self.lam, self.t_chunk

        def solve(s, L):
            if s >= n_pad:
                return 0.0, []
            split = None
            if L > fz.MIN_L:
                (j1, l1), (j2, l2) = solve(s, L // 2), solve(s + L // 2, L // 2)
                split = (j1 + j2, l1 + l2)
            if s + L <= n_pad and L in cost and s // L < len(cost[L]):
                leaf = (cost[L][s // L] + lam * t(L), [(s, L)])
                if split is None or leaf[0] <= split[0]:
                    return leaf
            return split

        leaves = []
        for s in range(0, n_pad, BLOCK):
            leaves += solve(s, BLOCK)[1]
        return leaves


def lbar(leaves):
    """L̄ = Σ n·L² / Σ n·L : độ dài chunk 'tương đương' về chi phí giải nén (Fixed-L có L̄ = L)."""
    return sum(L * L for _, L in leaves) / sum(L for _, L in leaves)


def lam_for_Lbar(cost, n_pad, target, t_chunk=None, lo=1e-3, hi=1e9, iters=60):
    """Chia đôi theo log λ để L̄(λ) ≈ target (L̄ giảm khi λ tăng). Trả (λ, L̄ đạt được). Chạy trên CPU."""
    dummy = np.zeros(n_pad)
    f = lambda lam: lbar(TreeDP(cost, lam, t_chunk)(dummy))
    for _ in range(iters):
        mid = (lo * hi) ** 0.5
        lo, hi = (mid, hi) if f(mid) > target else (lo, mid)
    best = min((lo, hi), key=lambda l: abs(f(l) - target))      # L̄ nhảy bậc theo λ → lấy phía gần target hơn
    return best, f(best)


# ----------------------------------------------------------------------------- 4. FeaturePolicy học từ nhãn DP
def block_features(toks, name, bits32=None):
    """Đặc trưng mỗi khối 512 token (gồm cả khối đuôi lẻ). name: 'entropy' | 'repeat' | 'llm32'.
    'llm32' = bit/token trung bình ở mức L=32 (MỘT lượt LLM rẻ, thay vì 5 lượt)."""
    nb = -(-len(toks) // BLOCK)
    if name == "llm32":
        idx = np.arange(len(bits32)) // (BLOCK // 32)
        return np.bincount(idx, weights=bits32, minlength=nb)[:nb] / (np.bincount(idx, minlength=nb)[:nb] * 32)
    j = 0 if name == "entropy" else 1
    return np.array([fz.FeaturePolicy.features(toks[b * BLOCK:(b + 1) * BLOCK])[j] for b in range(nb)])


def block_cost_matrix(cost, lam, t_chunk=None, levels=LEVELS):
    """C[b, j] = J nếu cả khối 512 thứ b dùng chunk dài levels[j] (chỉ các khối đủ 512)."""
    t = t_chunk or t_dec_quadratic()
    nb = len(cost[BLOCK])
    C = np.zeros((nb, len(levels)))
    for j, L in enumerate(levels):
        k = BLOCK // L
        C[:, j] = cost[L][:nb * k].reshape(nb, k).sum(1) + lam * t(L) * k
    return C


def fit_cuts(feature, C, levels=LEVELS, max_seg=5, bins=100):
    """Chia trục đặc trưng thành ≤ max_seg đoạn, mỗi đoạn một L cố định, tối thiểu tổng J (DP 1 chiều trên các bin phân vị).
    Trả (cuts, labels): L = labels[searchsorted(cuts, f, 'right')]."""
    order = np.argsort(feature, kind="stable"); f = feature[order]; Cs = C[order]
    nbin = min(bins, len(f)); edges = np.linspace(0, len(f), nbin + 1).astype(int)
    BC = np.add.reduceat(Cs, edges[:-1], axis=0)                            # [nbin, J]
    pre = np.vstack([np.zeros(BC.shape[1]), np.cumsum(BC, 0)])
    seg = lambda i, j: (pre[j] - pre[i])
    INF = 1e30
    dp = np.full((max_seg + 1, nbin + 1), INF); arg = np.zeros((max_seg + 1, nbin + 1), int); dp[0, 0] = 0
    for k in range(1, max_seg + 1):
        for j in range(1, nbin + 1):
            best, bi = INF, 0
            for i in range(k - 1, j):
                v = dp[k - 1, i] + seg(i, j).min()
                if v < best: best, bi = v, i
            dp[k, j], arg[k, j] = best, bi
    k = int(np.argmin(dp[1:, nbin])) + 1; j = nbin; cuts, labels = [], []
    while k > 0:
        i = arg[k, j]; labels.append(levels[int(seg(i, j).argmin())])
        if i > 0: cuts.append(f[edges[i]])
        j, k = i, k - 1
    return np.array(cuts[::-1]), labels[::-1]


def eval_cuts(feature, C, cuts, labels, levels=LEVELS):
    """J của chính sách ngưỡng trên tập đánh giá + so với 'oracle theo khối' và 'Fixed tốt nhất'."""
    pick = np.array([levels.index(labels[i]) for i in np.searchsorted(cuts, feature, side="right")])
    j_pol = C[np.arange(len(C)), pick].sum()
    return {"J_policy": j_pol, "J_oracle_block": C.min(1).sum(), "J_best_fixed": C.sum(0).min(),
            "gain_captured": (C.sum(0).min() - j_pol) / max(1e-9, C.sum(0).min() - C.min(1).sum())}


class ThresholdPolicy:
    """FeaturePolicy đã học ngưỡng. 'llm32' chỉ tốn 1 lượt LLM (mức 32) thay vì 5."""

    def __init__(self, feature, cuts, labels):
        self.feature, self.cuts, self.labels = feature, np.asarray(cuts), list(labels)

    def __call__(self, toks, lm, batch):
        bits32 = None
        if self.feature == "llm32":
            starts = list(range(0, len(toks) - 31, 32))
            bits32 = fz.run_score(toks, starts, 32, lm, batch)[0]
        f = block_features(toks, self.feature, bits32)
        pick = [self.labels[i] for i in np.searchsorted(self.cuts, f, side="right")]
        return fz.walk(len(toks), lambda s, L: L <= pick[s // BLOCK])


# ----------------------------------------------------------------------------- 5. so với đường Fixed
def _fixed_curve(fixed):                       # fixed: [(L̄, bytes)] sắp theo L̄
    fx = sorted(fixed); return np.array([p[0] for p in fx]), np.array([p[1] for p in fx], float)


def fixed_at(Lb, fixed, how="quad"):
    Ls, S = _fixed_curve(fixed); x = np.log2(Ls)
    if how == "share":                         # chia sẻ thời gian: trộn hai Fixed theo tỉ lệ token
        return float(np.interp(Lb, Ls, S))
    k = int(np.clip(np.searchsorted(Ls, Lb, side="right") - 1, 0, len(Ls) - 2))
    idx = [0, 1, 2] if k == 0 else [k - 1, k, k + 1]
    return float(np.polyval(np.polyfit(x[idx], S[idx], 2), np.log2(Lb)))


def report(points, fixed):
    """points: [(tên, L̄, bytes)]. Δ% < 0: nhỏ hơn Fixed ở CÙNG L̄ (cùng chi phí giải nén theo mô hình)."""
    print(f"{'policy':26s} {'L̄':>7s} {'bytes':>9s} {'Δ% bậc2':>9s} {'Δ% share':>9s}")
    for name, Lb, b in points:
        print(f"{name:26s} {Lb:7.1f} {b:9d} {100*(b/fixed_at(Lb, fixed)-1):9.2f} {100*(b/fixed_at(Lb, fixed, 'share')-1):9.2f}")


# ----------------------------------------------------------------------------- 6. test trên CPU với MockLM
def _selftest():
    lm = fz.MockLM(V=512, bonus=14.0)
    toks = fz.pad_tokens(fz._synthetic(511)[:4096 + 64], lm.bos)
    n_pad = len(toks)
    A = analyze(toks, lm, batch=8)
    # (a) TreeDP với bit AC + est_time phải trùng fz.DPPolicy
    for lam in (0.0, 1e3, 1e4, 1e5):
        dp = fz.DPPolicy(lam); dp._bits = {L: A[L]["bits"] for L in A}
        assert TreeDP(cost_ac(A), lam, fz.est_time)(toks) == dp(toks, lm, 8), lam
    print("OK: TreeDP == DPPolicy")
    # (b) κ, rank proxy, khớp L̄
    fixed_b, fixed_h, pts = {}, {}, []
    for L in LEVELS:
        blob, lv = fz.encode(fz._synthetic(511)[:4096 + 64], lm, fz.FixedPolicy(L), batch=8)
        fixed_b[L] = len(blob); fixed_h[L] = fz.encode.last_stats["header_bytes"]; pts.append((lbar(lv), len(blob)))
    kap = kappa(A, fixed_b, fixed_h); print("κ_L =", {L: round(float(v), 3) for L, v in kap.items()})
    costs = {"AC": cost_ac(A), "calibrated": cost_calibrated(A, kap), "rank-proxy": cost_rank_proxy(A)}
    for nm, c in costs.items():
        lam, got = lam_for_Lbar(c, n_pad, 128.0)
        print(f"{nm:12s} λ={lam:10.3g} -> L̄={got:6.1f}")
    # (b2) fit_cuts phải tìm lại ngưỡng đã biết
    rng = np.random.default_rng(0); f = rng.random(400); Ct = np.full((400, 5), 5.0)
    Ct[f < 0.3, 0] = 0; Ct[(f >= 0.3) & (f < 0.7), 2] = 0; Ct[f >= 0.7, 4] = 0
    cuts, labels = fit_cuts(f, Ct, max_seg=5, bins=100)
    assert labels == [32, 128, 512] and np.allclose(cuts, [0.3, 0.7], atol=0.02), (cuts, labels)
    print("OK: fit_cuts tìm lại ngưỡng", np.round(cuts, 3), labels, "| eval:", round(eval_cuts(f, Ct, cuts, labels)["gain_captured"], 2))
    # (c) FeaturePolicy học ngưỡng + round-trip lossless
    lam, _ = lam_for_Lbar(costs["AC"], n_pad, 128.0)
    C = block_cost_matrix(costs["AC"], lam)
    nb = len(C); tr, te = slice(0, nb // 2), slice(nb // 2, nb)
    for name in ("entropy", "repeat", "llm32"):
        bits32 = A[32]["bits"] if name == "llm32" else None
        f = block_features(toks, name, bits32)[:nb]
        cuts, labels = fit_cuts(f[tr], C[tr], max_seg=3, bins=16)
        ev = eval_cuts(f[te], C[te], cuts, labels)
        print(f"feature={name:8s} cuts={np.round(cuts, 2)} labels={labels} gain_captured(test)={ev['gain_captured']:.2f}")
    cuts, labels = fit_cuts(block_features(toks, "llm32", A[32]["bits"])[:nb], C, max_seg=3, bins=16)
    pol = ThresholdPolicy("llm32", cuts, labels)
    tk = fz._synthetic(511)[:4096 + 64]
    blob, lv = fz.encode(tk, lm, pol, batch=8)
    assert np.array_equal(fz.decode(blob, lm), tk); print("OK: ThresholdPolicy round-trip lossless, L̄ =", round(lbar(lv), 1))
    report([("ThresholdPolicy(llm32)", lbar(lv), len(blob))], pts)


if __name__ == "__main__":
    _selftest()
