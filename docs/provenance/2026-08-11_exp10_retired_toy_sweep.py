"""Exp. 10 retired BEFORE any run: the ortho scoring rule is inert by construction.

With per-band z-scored targets the two duo candidates have equal norms, so
argmax Pearson == argmax covariance and the shared component cancels exactly in
cov(shat, s1) - cov(shat, s2) = cov(shat, s1 - s2). Orthogonalising each candidate
against its competitor removes what the comparison already removed. Generative toy
below: plain and ortho accuracies are identical across noise / shared-weight regimes.
Vault: 'Cap. 2 -- Exp. 10 (ESPLORATIVO) ... (11 ago 2026)', retraction section.
"""
import numpy as np

def bp(a, b):
    return np.mean([np.corrcoef(a[i], b[i])[0, 1] for i in range(a.shape[0])])

if __name__ == "__main__":
    L, nb, trials = 400, 3, 400
    print("noise  shared_w  acc_plain  acc_ortho")
    for shared_w in (1.0, 2.0):
        for noise in (4.0, 8.0, 16.0):
            wp = wo = 0
            r = np.random.RandomState(7)
            for _ in range(trials):
                shared, d1, d2 = r.randn(nb, L), r.randn(nb, L), r.randn(nb, L)
                s1, s2 = shared_w * shared + 0.35 * d1, shared_w * shared + 0.35 * d2
                shat = s1 + noise * r.randn(nb, L)
                wp += bp(shat, s1) > bp(shat, s2)
                orth = []
                for c, o in ((s1, s2), (s2, s1)):
                    c = c - c.mean(1, keepdims=True); o = o - o.mean(1, keepdims=True)
                    orth.append(c - (np.sum(c*o, 1, keepdims=True) / (np.sum(o*o, 1, keepdims=True) + 1e-12)) * o)
                wo += bp(shat, orth[0]) > bp(shat, orth[1])
            print(f"{noise:5.1f}  {shared_w:8.1f}  {wp/trials:9.3f}  {wo/trials:9.3f}")
