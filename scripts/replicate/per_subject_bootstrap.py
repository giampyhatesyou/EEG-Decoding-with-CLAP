"""Per-participant accuracies and a hierarchical bootstrap for the MAD-EEG tests of Ch. 9.

ANALYSIS OF 2026-09-27, requested by the supervisor, on data already analysed. It reads the
committed decision records only (docs/provenance, plus the step C records in the run
archive); no EEG is opened and nothing is retrained. The only cuts are by participant and
by mixture.

POSITIVE CONTROL, crossed first (exit 1 otherwise): every total below reproduces the
published count exactly; the Exp. 8 agreement table (159 / 49 / 84 / 84, McNemar 84/133);
the mean r(attended) / r(best unattended) of the mel anchor (0.0202 / 0.0118) and of flux P2
(0.0560 / 0.0513); step C's prior-following 89/124 against a null of 0.4274.
THEN, per test: accuracy per participant with its exact 95% interval; the pooled count with
its exact binomial interval (trials as independent) next to a hierarchical bootstrap
interval (participants resampled with replacement, then, inside each drawn participant,
its mixtures with replacement; 10 000 replicates, seed 42, percentile). For own-vs-other
the second level is the mutual pair, the unit whose two directions share both segments.

Usage: /opt/miniconda3/bin/python scripts/replicate/per_subject_bootstrap.py --madeeg_dir ~/madeeg \
         [--archive DIR] [--fig_dir DIR]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
PROV = os.path.join(ROOT, "docs", "provenance")
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT = "2026-09-27_per_subject_bootstrap.txt"
N_BOOT, SEED = 10_000, 42

DUO = [  # (label, file, published k, n)
    ("anchor, ridge k-fold, log-mel (A1)", "2026-08-27_canary_duo_RESULT_madeeg_records.csv", 86, 154),
    ("anchor, ridge, published protocol (A3)", "2026-08-10_canary_rawsolos_RESULT_madeeg_records.csv", 74, 154),
    ("Exp. 9 flux, published protocol (P1)", "2026-08-11_exp9_P1_rawsolos_flux_RESULT_madeeg_records.csv", 76, 154),
    ("Exp. 9 flux, k-fold (P2)", "2026-08-27_exp9_P2_duo_flux_RESULT_madeeg_records.csv", 77, 154),
    ("Exp. 4 mono replication", "2026-07-30_exp04_mono_confirm_RESULT_madeeg_records.csv", 79, 150),
]
OVO_REF = "2026-08-11_exp07_ovo_sweep_REF_RESULT_madeeg_ownvsother.csv"   # md5 2eaa9262..., 208/376
OVO_FLUX = "2026-08-11_exp08_ovo_flux_F1_RESULT_madeeg_ownvsother.csv"
REG11 = "2026-08-17_exp11_primary_RESULT_pairs.csv"
REG12 = "2026-09-27_exp12_primary_RESULT_pairs.csv"
ALPHA = {"stereo": ("2026-08-17_alpha_real_stereo_RESULT_madeeg_alpha_pairs.csv", 23, 44),
         "mono": ("2026-08-17_alpha_real_mono_RESULT_madeeg_alpha_pairs.csv", 24, 42)}


def mixture(stim):
    """`classique_morceau1_duo_CoFl_theme1_stereo[_lcr]_Co` -> `classique_morceau1_duo_CoFl_theme1`:
    the piece, the pair and the theme, without the attended instrument and the render."""
    return "_".join(stim.split("_")[:5])


def cp(k, n):
    ci = binomtest(k, n).proportion_ci(method="exact")
    return ci.low, ci.high


def hier_boot(df, col, seed=SEED):
    """Percentile interval of sum(col)/n over (participant, then cluster) resamples."""
    rng = np.random.RandomState(seed)
    subs = sorted(df.subject.unique())
    per = {s: df[df.subject == s].groupby("cluster")[col].agg(["sum", "count"]).to_numpy() for s in subs}
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        tot = np.zeros(2)
        for s in rng.randint(0, len(subs), len(subs)):
            g = per[subs[s]]
            tot += g[rng.randint(0, len(g), len(g))].sum(axis=0)
        out[b] = tot[0] / tot[1]
    return np.percentile(out, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--madeeg_dir", required=True)
    ap.add_argument("--archive", default=os.path.join(ROOT, "..", "_baldo_archive_2026-07-18", "results"))
    ap.add_argument("--fig_dir", default="")
    args = ap.parse_args()
    L = ["!! ANALYSIS OF 2026-09-27, requested by the supervisor, on data already analysed.",
         "!! Decision records only; the cuts are by participant and by mixture, nothing else.",
         "!! Every duo, register and alpha decision has the null 0.500 (both instruments of a",
         "!! mixture are targets); own-vs-other 0.500 exact by symmetry; step C's prior-following",
         "!! has the null 0.4274 = P(prior == target), computed from the data.",
         "!! With 8 participants the smallest p attainable at participant level is 1/2^8 = 1/256",
         "!! = 0.0039 (all 8 on the same side): the participant-level view estimates uncertainty,",
         "!! it cannot add power.", ""]

    def say(s=""):
        L.append(s)
        print(s, flush=True)

    def stop(msg):
        say(f"POSITIVE CONTROL FAILED: {msg}. Nothing below this line exists.")
        open(os.path.join(PROV, OUT), "w").write("\n".join(L) + "\n")
        sys.exit(1)

    rd = lambda f, **kw: pd.read_csv(os.path.join(PROV, f), dtype={"subject": str, "subj": str}, **kw)
    tests = []                                       # (label, df[subject, cluster, y], k, n, null)
    for label, f, k, n in DUO:
        d = rd(f)
        tests.append((label, pd.DataFrame(dict(subject=d.subject, cluster=d.stim.map(mixture), y=d.correct)), k, n, 0.5))
    ovo = {}
    for label, f, k in (("own-vs-other, mel reference", OVO_REF, 208), ("own-vs-other, flux (Exp. 8 F1)", OVO_FLUX, 243)):
        d = rd(f)
        ovo[label] = d
        pair = [f"{fo}:{min(a, b)}-{max(a, b)}" for fo, a, b in zip(d.fold, d.seg, d.other)]
        tests.append((label, pd.DataFrame(dict(subject=d.subject, cluster=pair, y=d.correct)), k, 376, 0.5))
    r11, r12 = rd(REG11), rd(REG12)
    r12["cluster"] = r12.stim_hi.map(mixture)
    r11 = r11.merge(r12[r12.spatial == "stereo"][["subj", "dcent", "cluster"]], on=["subj", "dcent"], how="left")
    if r11.cluster.isna().any():
        stop("an Exp. 11 pair has no Exp. 12 twin to name its mixture")
    tests.append(("register stereo (Exp. 11)", pd.DataFrame(dict(subject=r11.subj, cluster=r11.cluster, y=r11.hit.astype(int))), 24, 47, 0.5))
    for lab, sel, k, n in (("register mono (Exp. 12)", r12.spatial == "mono", 22, 42), ("register pooled (Exp. 12)", r12.spatial.notna(), 46, 89)):
        g = r12[sel]
        tests.append((lab, pd.DataFrame(dict(subject=g.subj, cluster=g.cluster, y=g.hit.astype(int))), k, n, 0.5))
    for render, (f, k, n) in ALPHA.items():
        d = rd(f)
        tests.append((f"paired alpha {render}", pd.DataFrame(dict(subject=d.subject, cluster=d.mixture, y=d.correct)), k, n, 0.5))

    # step C: did the model choose its training-fold prior? (madeeg_diagnose's own functions)
    from madeeg_diagnose import load, majority
    rows = load(os.path.join(args.archive, "madeeg_clap_kfold", "madeeg_contrastive_records.csv"),
                os.path.expanduser(args.madeeg_dir))
    dec = [(r["subject"], r["mixture"], int(r["chosen"] == mj), int(mj == r["target"]))
           for r in rows for mj in [majority(rows, r)] if mj is not None]
    sc = pd.DataFrame(dec, columns=["subject", "cluster", "y", "prior_right"])

    say("=== POSITIVE CONTROL (published totals, crossed before anything else) ===")
    for label, d, k, n, _ in tests:
        got = (int(d.y.sum()), len(d))
        say(f"  {label:42s} {got[0]}/{got[1]}  published {k}/{n}  [{'OK' if got == (k, n) else 'FAIL'}]")
        if got != (k, n):
            stop(label)
    a, b = ovo["own-vs-other, mel reference"], ovo["own-vs-other, flux (Exp. 8 F1)"]
    m = a.merge(b, on=["subject", "fold", "seg", "other"], suffixes=("_mel", "_flux"))
    tab = (int(((m.correct_mel == 1) & (m.correct_flux == 1)).sum()), int(((m.correct_mel == 1) & (m.correct_flux == 0)).sum()),
           int(((m.correct_mel == 0) & (m.correct_flux == 1)).sum()), int(((m.correct_mel == 0) & (m.correct_flux == 0)).sum()))
    say(f"  Exp. 8 agreement table both/mel only/flux only/neither {tab}  published (159, 49, 84, 84)  "
        f"[{'OK' if tab == (159, 49, 84, 84) and len(m) == 376 else 'FAIL'}]")
    if tab != (159, 49, 84, 84) or len(m) != 376:
        stop("Exp. 8 agreement table")
    diffs = {}
    for label, f, pub in (("mel anchor (A1)", DUO[0][1], (0.0202, 0.0118)), ("flux k-fold (P2)", DUO[3][1], (0.0560, 0.0513))):
        d = rd(f)
        got = (round(float(d.r_attended.mean()), 4), round(float(d.r_best_unattended.mean()), 4))
        say(f"  mean r(attended) / r(best unattended), {label:17s} {got}  published {pub}  [{'OK' if got == pub else 'FAIL'}]")
        if got != pub:
            stop(label)
        diffs[label] = pd.DataFrame(dict(subject=d.subject, cluster=d.stim.map(mixture), y=d.r_attended - d.r_best_unattended))
    got = (int(sc.y.sum()), len(sc), round(sc.prior_right.mean(), 4))
    say(f"  step C prior-following {got[0]}/{got[1]}, null {got[2]}  published 89/124, 0.4274  "
        f"[{'OK' if got == (89, 124, 0.4274) else 'FAIL'}]")
    if got != (89, 124, 0.4274):
        stop("step C prior-following")

    say()
    say("=== PER PARTICIPANT AND POOLED (exact 95% intervals; hierarchical bootstrap percentile) ===")
    summary, g2 = [], []
    for label, d, k, n, null in tests:
        say(f"\n  {label}: {k}/{n} = {k / n:.4f}, null {null:.3f}, from {d.subject.nunique()} participants "
            f"and {d.groupby('subject').cluster.nunique().sum()} participant-clusters")
        per = []
        for s, g in d.groupby("subject"):
            kk, nn = int(g.y.sum()), len(g)
            lo, hi = cp(kk, nn)
            per.append((s, kk, nn, lo, hi))
            say(f"    {s}  {kk:>3d}/{nn:<3d} = {kk / nn:.3f}   [{lo:.3f}, {hi:.3f}]")
        above = sum(kk / nn > null for _, kk, nn, _, _ in per)
        below = sum(kk / nn < null for _, kk, nn, _, _ in per)
        ps = binomtest(above, above + below, 0.5, alternative="greater").pvalue if above + below else float("nan")
        lo, hi = cp(k, n)
        hb = hier_boot(d, "y")
        say(f"    participants above the null {above}, below {below}, at it {len(per) - above - below}   "
            f"sign test over participants (null 50/50) one-sided p = {ps:.4f}")
        say(f"    pooled {k / n:.4f}   exact binomial 95% [{lo:.3f}, {hi:.3f}] (trials as independent)   "
            f"hierarchical bootstrap 95% [{hb[0]:.3f}, {hb[1]:.3f}]   null {null:.3f} "
            f"{'INSIDE' if hb[0] <= null <= hb[1] else 'outside'} the hierarchical interval")
        summary.append((label, per, k / n, hb, null))

    say("\n=== G1: the two paired contrasts and the prior ===")
    d = m.assign(y=m.correct_flux - m.correct_mel,
                 cluster=[f"{fo}:{min(x, z)}-{max(x, z)}" for fo, x, z in zip(m.fold, m.seg, m.other)])
    say("  own-vs-other, flux minus mel (Exp. 8 F1 - reference, 376 paired decisions, 188 mutual pairs, 8 participants):")
    for s, g in d.groupby("subject"):
        say(f"    {s}  mel {int(g.correct_mel.sum()):>2d}/{len(g)}  flux {int(g.correct_flux.sum()):>2d}/{len(g)}  "
            f"difference {g.y.mean():+.3f}  (flux-only {int((g.y > 0).sum())}, mel-only {int((g.y < 0).sum())})")
    hb = hier_boot(d, "y")
    say(f"    pooled difference {d.y.mean():+.4f} (243 - 208 = 35/376)   null 0   hierarchical bootstrap 95% "
        f"[{hb[0]:+.4f}, {hb[1]:+.4f}]   null {'INSIDE' if hb[0] <= 0 <= hb[1] else 'outside'}")
    pos = sum(g.y.mean() > 0 for _, g in d.groupby("subject"))
    neg = sum(g.y.mean() < 0 for _, g in d.groupby("subject"))
    say(f"    participants with flux above mel {pos}, below {neg}   sign test (null 50/50) one-sided p = "
        f"{binomtest(pos, pos + neg, 0.5, alternative='greater').pvalue:.4f}")
    g2.append(("own-vs-other flux - mel (McNemar 84/133)", d.y.mean(), hb, 0.0))
    g2 += [(lab, acc, hb_, nl) for lab, _, acc, hb_, nl in summary if lab.startswith("own-vs-other, flux")]

    say("\n  r(attended) - r(best unattended), mean per participant (null 0):")
    say(f"    {'':6s} {'mel anchor (A1)':>16s} {'flux k-fold (P2)':>17s}")
    subs = sorted(diffs["mel anchor (A1)"].subject.unique())
    for s in subs:
        say(f"    {s}  {diffs['mel anchor (A1)'].query('subject == @s').y.mean():+16.4f} "
            f"{diffs['flux k-fold (P2)'].query('subject == @s').y.mean():+17.4f}")
    for lab, dd in diffs.items():
        hb = hier_boot(dd, "y")
        say(f"    {lab}: pooled {dd.y.mean():+.4f}   hierarchical bootstrap 95% [{hb[0]:+.4f}, {hb[1]:+.4f}]   "
            f"null 0 {'INSIDE' if hb[0] <= 0 <= hb[1] else 'outside'}")

    say("\n  step C, the model chose its training-fold prior (89/124 decidable trials):")
    for s, g in sc.groupby("subject"):
        say(f"    {s}  follows {int(g.y.sum()):>2d}/{len(g):<2d} = {g.y.mean():.3f}   null for this participant "
            f"{g.prior_right.mean():.3f}")
    hb = hier_boot(sc, "y")
    sc2 = sc.assign(y=sc.y - sc.prior_right)
    hd = hier_boot(sc2, "y")
    say(f"    pooled 0.7177   hierarchical bootstrap 95% [{hb[0]:.3f}, {hb[1]:.3f}]   null 0.4274 "
        f"{'INSIDE' if hb[0] <= 0.4274 <= hb[1] else 'outside'}")
    say(f"    following minus its null, resampled together: {sc2.y.mean():+.4f}   95% [{hd[0]:+.4f}, {hd[1]:+.4f}]   "
        f"0 {'INSIDE' if hd[0] <= 0 <= hd[1] else 'outside'}")
    g2.append(("step C prior-following 89/124 (minus its null)", sc2.y.mean(), hd, 0.0))

    say("\n=== G2 CHECK: results reported as positive, is the null inside the hierarchical interval? ===")
    for lab, est, hb, nl in g2:
        say(f"  {lab:48s} {est:+.4f}  [{hb[0]:+.4f}, {hb[1]:+.4f}]  null {nl:.3f} "
            f"{'INSIDE -> G2' if hb[0] <= nl <= hb[1] else 'outside'}")
    say("  (15/18 of the stem similarity has no participant level: its interval over mixtures is in")
    say("   2026-09-27_separability_18_mixtures.txt)")
    open(os.path.join(PROV, OUT), "w").write("\n".join(L) + "\n")
    print(f"\n-> {OUT}")

    if args.fig_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 4, figsize=(9.5, 7.2), sharey=True)
        for ax, (label, per, acc, hb, null) in zip(axes.flat, summary):
            for i, (s, kk, nn, lo, hi) in enumerate(per):
                ax.errorbar(i, kk / nn, yerr=[[kk / nn - lo], [hi - kk / nn]], fmt="o", ms=3, color="0.3", lw=0.8)
            ax.errorbar(len(per) + 0.8, acc, yerr=[[acc - hb[0]], [hb[1] - acc]], fmt="D", ms=4, color="k", lw=1.4)
            ax.axhline(null, color="tab:red", ls="--", lw=0.8)
            ax.set_xticks(list(range(len(per))) + [len(per) + 0.8], [s[-1] for s, *_ in per] + ["all"], fontsize=6)
            ax.set_title(label, fontsize=7)
            ax.set_ylim(0, 1)
        for ax in axes.flat[len(summary):]:
            ax.axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel("accuracy", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.fig_dir, "per_subject_accuracy.pdf"))
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        for j, (lab, dd) in enumerate(diffs.items()):
            v = [dd[dd.subject == s].y.mean() for s in subs]
            ax.plot(np.arange(len(subs)) + (j - 0.5) * 0.25, v, "o" if j else "s", ms=4,
                    mfc="none" if j else "0.3", mec="0.3", ls="none", label=lab)
        ax.axhline(0, color="tab:red", ls="--", lw=0.8)
        ax.set_xticks(range(len(subs)), [s[-1] for s in subs], fontsize=7)
        ax.set_xlabel("participant", fontsize=8)
        ax.set_ylabel("r(attended) - r(unattended)", fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(args.fig_dir, "per_subject_differential.pdf"))
        print(f"-> {args.fig_dir}/per_subject_{{accuracy,differential}}.pdf")


if __name__ == "__main__":
    main()
