# 6 · `docs/provenance/` — the output of the runs, as the runs printed it

*Reading path: [1 README](../../README.md) → [2 REPO_MAP](../02_REPO_MAP.md) → [3 OVERVIEW](../03_OVERVIEW.md) → [4 CODE_TOUR](../04_CODE_TOUR.md) → [5 replicate](../../scripts/replicate/README.md) → **6 you are here** → [7 METHOD_RULES](../07_METHOD_RULES.md).*

Every figure this project reports has a file here, and this note says which. The
files are **verbatim run output**: nothing in this directory is edited after the
fact, including when it is wrong (see the two traps at the bottom). Corrections
live in the notes that cite them, never in the evidence.

- `*_RESULT_*` — the output of the run that produced a reported number. It was
  copied here from a run directory because `runs/` is gitignored, so without the
  copy the number would exist in a clone only as a value written in a script.
- everything else — output written straight to this directory: paired tests,
  audits, and the preparation gates of an experiment (its nulls, its controls,
  its thresholds) crossed *before* the run they belong to.

Anything not here and cited anyway is a defect. Report it.

## Which file carries which number

**Chapter 1 (Akama dataset)** is not in this directory: its 103 folds are pinned
in `../../results_manifest.tsv` and rendered into `../../RESULTS.md`, which is a
stronger form of provenance — each pin is verified against its run's own
`hparams.yaml` and `report.py` refuses to write the table if one disagrees.

| number | file |
|---|---|
| the axis study of 2026-07-29, read outside any pre-registration — the eleven `ax_*` runs that survive, maximum **86/150 = 0.5733** (`ax_ica_perinstr`: per-instrument ridge, notch+ICA, 8 mel at 64 Hz); the list of all sixteen configurations was never written to a file, but no run dated 2026-07-29 prints a higher accuracy | `2026-07-29_ax_*_RESULT_*` |
| the published-protocol anchor without `--test_eeg raw`: **75/154 = 0.4870** (with the flag: 74/154 = 0.4805, pinned in `results_manifest.tsv`) | `2026-07-29_madeeg_gate_solos_RESULT_*` |
| **Exp. 4** — the mono half, the only confirmatory look: **79/150 = 0.5267**, F1 0.5267/0.5225/0.5160 (p 0.2839 and 95% CI [0.444, 0.609] are exact binomial on 79/150, not printed by the run); the spent genre split **classical 47/90, pop 32/60** is a count over the `stim` column of the same records | `2026-07-30_exp04_mono_confirm_RESULT_*` |
| **Exp. 6** — own-vs-other gate: ridge **202/376**, CCA **207/376**, ridge+ICA **208/376**, CCA+ICA **209/376**; McNemar CCA vs ridge **66/131** (with ICA) and **72/139** (without), exact paired p: ridge 0.0851, CCA 0.0265, ridge+ICA 0.0235, CCA+ICA 0.0230 | `2026-08-10_exp06_ovo_{ridge,cca,ridge_ica,cca_ica}_RESULT_*`; `2026-08-20_exp06_ovo_gate_RESULT_{mcnemar,paired_stats}.txt` (run on the 2026-08-20 replicas, whose CSVs are byte-identical to the originals) |
| **Exp. 7** — own-vs-other sweep: reference **208/376**, best **213/376**, no candidate clears Bonferroni | `2026-08-11_ovo_sweep_mcnemar.txt` (output) · `2026-08-11_ovo_sweep_paired_stats.py` (the exact paired test, code) |
| **Exp. 7** — the nine configurations of the sweep, per decision (the inputs of the McNemar file above) | `2026-08-11_exp07_ovo_sweep_{REF,C1,...,C8}_RESULT_*` |
| **Exp. 8** — flux **243/376**, McNemar 84/133 p 0.0015; variant 238/376 | `2026-08-11_exp8_flux_mcnemar.txt` |
| **Exp. 8** — the runs behind it: flux F1 **243/376**, exact paired p **1.78e-08** against the null; variant F2 238/376; the regression check of the reference. Mean own-band correlation **0.0360** (regcheck) → **0.0742** (flux F1) | `2026-08-11_exp08_ovo_flux_{F1,F2,regcheck}_RESULT_*` |
| **Exp. 9** — flux on attention **76/154**, **77/154** (mel 74 and 86); stem correlation flux **0.2874** vs mel **0.1772**, 30/36 | `2026-08-11_exp9_flux_attention_mcnemar.txt` |
| **Exp. 9**, protocol P1 — the two runs the McNemar above compares: mel anchor 74/154 (`canary_rawsolos`) and flux 76/154 | `2026-08-10_canary_rawsolos_RESULT_*`, `2026-08-11_exp9_P1_rawsolos_flux_RESULT_*` |
| **Exp. 9** — the attended-minus-unattended differential, mel **0.0202/0.0118** vs flux **0.0560/0.0513** | `2026-08-27_canary_duo_RESULT_*`, `2026-08-27_exp9_P2_duo_flux_RESULT_*` (records added on 2026-09-25, from the same run directories; the summaries are unchanged) |
| prior-following, steps C and D — **0.7177** and **0.8145** against a null of **0.4274** | `2026-08-11_diagnose_step{C,D}.txt` |
| **Exp. 13** — MFCC-13 within **0.0533** / floor **0.0256**, 32/36, verdict GO | `2026-08-12_exp13_stem_separability.txt` |
| **Exp. 14** — MFCC **214/376** p 0.004224, mel-64 **212/376** p 0.007623 | `2026-08-12_exp14_mfcc_tracking.txt` |
| **Exp. 15** — D(mel) **+0.1001** vs D(MFCC) **−0.1119** p 0.9833, MDD **0.2467**, 65/154 | `2026-08-12_exp15_mfcc_differential.txt` |
| **Exp. 16A** — the negative control: noise-512 passes both pre-registered criteria (36/36, gap −0.0002, w/floor **0.922**) | `2026-08-12_exp16A_negative_control.txt` |
| **Exp. 16A** — the CLAP arm itself, **prepared and never executed**: the file says so in its first line and carries no CLAP number. CLAP was measured later, by Exp. 17 | `2026-08-12_exp16A_clap_separability.txt` |
| **Exp. 16B** — CCA with band power **202/376** p 0.08185, below the 208 bar | `2026-08-12_exp16B_ccaviews_ownvsother.txt` |
| **Exp. 17** — CLAP w/floor **3.837** against a ceiling of 1.80 → NO-GO | `2026-08-12_exp17_clap_separability.txt` (+ extraction log and manifest) |
| **Exp. 18** — preparation gates, then S1 **278/536 = 0.5187**, null 0.5000 measured | `2026-08-12_exp18_matchmismatch.txt`, `2026-08-13_exp18_S1_RESULT_*` |
| **Exp. 19** — preparation, then S1 **244/536**, S2 **113/188 = 0.6011** p 0.003404, both with the full training curve | `2026-08-13_exp19_S{1,2}.txt`, `2026-08-13_exp19_S{1,2}_RESULT_*` |
| **Exp. 11** — register, stereo **24/47**; sweep 38/47 at +18.9 %, 26/47 at +9.2 %; controls 47/47 and 0.4968 | `2026-08-17_exp11_*_RESULT_*` |
| **Exp. 11, detection floor** — band-power change of each injected amplitude (×1.395, ×1.189, ×1.092, ×1.036; no label read) | `2026-09-25_floor_power_ratio.txt` (`scripts/replicate/floor_power_ratio.py`) |
| **Exp. 12** — register, mono **22/42**, pooled **46/89**, stereo-from-raw 24/47 | `2026-08-17_exp12_*_RESULT_*` |
| paired alpha — **23/44** stereo, **24/42** mono control, injection **44/44** | `2026-08-17_alpha_*_RESULT_*` |
| paired alpha, the injection at other amplitudes — **43/44** at half amplitude, **44/44** at 1.0 and at double | `2026-08-10_alpha_ctrl2_a{0.5,1.0,2.0}_inject_RESULT_*` |
| arm D leakage audit — pseudo-labels **0.6209** window-CV vs **0.4847** trial-CV | `2026-08-17_armD_leakage_audit_RESULT_*` |
| arm A protocol parity — F1 **0.5267** / **0.4800** / **0.5400**; same-melody 46/89 vs diff 33/61 | `2026-08-17_armA_paper_*_RESULT_*`, `2026-08-11_armA_same_diff_melody.txt` — the split has had code again since 2026-09-01: `scripts/replicate/armA_same_diff_melody.sh` reproduces both bands exactly |
| step C **58/154 = 0.3766** | pinned in `results_manifest.tsv` as `madeeg_clap_kfold` |
| step D **70/154 = 0.4545**; ridge on the envelope target **0.5065** | `2026-08-27_madeeg_clap_kfold_within_RESULT_*`, `2026-08-27_madeeg_ridge_duo_env_RESULT_*` |
| synthetic positive controls, steps C and D — **148/154** and **142/154** | `2026-08-27_madeeg_clap_selftest{,_within}_RESULT_*` |
| **Exp. 10** — retired before running: the toy that shows the rule is algebraically inert | `2026-08-11_exp10_retired_toy_sweep.py` |

## Two traps in these files, both left in place on purpose

**1. `2026-08-27_madeeg_clap_selftest_RESULT_summary.txt` reads `ABOVE CHANCE`
and does not say that the EEG is synthetic.** It is the artefact of the incident
that produced method rule 9: a control whose caveat lived only on the terminal.
Its 148/154 is a wiring check on a fixed linear operator applied to the attended
source, never a statement about real EEG. Its step-D twin, written later, carries
the banner the older file lacks. Both are kept, unedited, as the record of the
scar. See `../07_METHOD_RULES.md` §9.

**2. The two self-test summaries print `pre-registered threshold 88/154`.** That
is the *reference* threshold of the real experiment, not the 0.90 the code
declares for the control itself. The control's own bar is in
`src/madeeg_contrastive.py`, not in its summary.

## Comparing a rerun against a file here

Compare **numbers**, not bytes. Prose lines in the summaries have changed since
these files were written (emoji removed, Italian translated), so a byte diff will
show differences that are not regressions. Bytes matter only where an md5 is
declared: `scripts/replicate/canaries.sh` pins three, and they are checked
against the stored files above, not against a rerun.
