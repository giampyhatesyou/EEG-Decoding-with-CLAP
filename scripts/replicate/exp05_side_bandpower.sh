#!/usr/bin/env bash
# =============================================================================
# Exp. 5 - the attended SIDE from band power. NEVER RUN AS WRITTEN.
#
# This script executes nothing and produces no number. It exists because the
# Exp. 5 pre-registration is on record, three of its thresholds are still in use,
# and a replication directory that simply DOES NOT CONTAIN Exp. 5 would look like
# an oversight rather than a decision.
#
# WHAT IT MEASURED: whether band power (alpha in particular) allows decoding which
#   SIDE the attended instrument came from - and whether this happens in the
#   STEREO rendering (+/-45 degrees) and NOT in the MONO one.
# DATE       : criterion written 2026-08-09, before any line of code and any
#              number. ARCHIVED 2026-08-11.
# ORIGIN     : EXTERNAL, in writing, dated - a written suggestion received around
#              2026-08-08 ("split up the EEG signal in different frequency bands,
#              or better even, use the frequency decomposition as the features for
#              the decoder"), pointing to de Vries, Marinato & Baldauf (2021).
#              The hypothesis does not come from our data. That does not make it
#              confirmatory (the duos were already spent), but it removes any
#              suspicion that it was picked after seeing what was convenient.
# QUESTION   : "can the side of the attended instrument be read from the frequency
#              decomposition?"
# NULL       : 0.5 in theory, but the OPERATIONAL null was 75/149 = 0.5034 - the
#              constant guesser that always answers "left". That is the floor, and
#              it had to be printed next to the statistic.
# PRE-REGISTERED THRESHOLD: 86/149 = 0.5772 on the primary (p = 0.0356; 85 would
#              have given 0.0505, so there was no freedom of choice).
# VERDICT    : ARCHIVED WITHOUT A NUMBER. The SIDE decision was dropped on
#              2026-08-10 and the 86/149 threshold with it. No run corresponds to
#              this pre-registration.
# BUDGET     : zero. Never having been executed, it spent nothing.
# GPU        : n/a.
#
# WHAT SURVIVES, AND IT IS THE PART THAT MATTERS - three thresholds written on
#    2026-08-09, i.e. TWO DAYS BEFORE any number existed, and later reused by other
#    experiments. It is the material proof that those thresholds were not chosen
#    after seeing the data:
#
#      30/47  ->  reused by Exp. 11 (spectral register, stereo duos)
#      27/42  ->  reused by Exp. 12 (mono duos) and by Exp. 6 [H-prime]
#      53/89  ->  reused by Exp. 12 (pooled)
#      28/44  ->  the Exp. 6 [H] variant after excluding the 3 centred targets
#                 (44 opposite-side pairs instead of 47)
#
# AND THE METADATA DISCOVERY that made everything else possible, counted on
#    2026-08-09, which spends no held-out looks (it counts metadata, not EEG):
#    `wav_info.panning` in `madeeg_preprocessed.yaml` gives, for every trial and
#    every instrument, 0.2 = left / 0.8 = right / 0.5 = centre. Hence:
#      149 stereo duos with a defined target side  (75 L / 74 R)
#        5 stereo duos with the target at CENTRE -> excluded A PRIORI by the criterion
#          (all `pop_mixtape_duo_GtVx_theme1_stereo_Vx`: the voice is centred)
#       47 stereo twin pairs, of which 44 on opposite sides
#       42 mono twin pairs
#       22 mixtures with a twin in BOTH renderings
#
# THE CONFOUND DECLARED IN ADVANCE, which no successor has removed: in the pop
#    tracks (`BsDr`, `GtVx`) the panning is fixed, so side is identical to
#    instrument and a side decoder may be an instrument decoder. In the classical
#    tracks it is not: 10 (track, instrument) combinations out of 20 appear with the
#    attended instrument both left and right, over 79 trials - but the flip is
#    BETWEEN subjects, not within subject (0 subjects heard the same instrument from
#    both sides), so that contrast is BETWEEN-SUBJECT.
#
# NO FILE IN THE REPO IMPLEMENTS EXP. 5 AS WRITTEN. Verified 2026-08-15: there is
#    no band-power SIDE decoder. The things that resemble it, and are NOT the same:
#      - `--alpha_li` (Exp. 6 [H]) decides the side, but with the alpha LATERALITY
#        INDEX on twin pairs, not with a decoder trained on band power;
#      - `src/madeeg_spectral_attention.py` (Exp. 11/12) is band power + LDA, but
#        decides the REGISTER (high/low), not the SIDE;
#      - `--cca_views lateralization` (model 9) is a FEATURE, not a test, and inside
#        1-8 Hz it is identically zero (alpha is out of band).
#    Writing "Exp. 5 was done by X" would be false for all three.
#
# PRE-REGISTRATION: "Chapter 2 - Exp. 5: attended side from band power,
#              pre-registered criterion (2026-08-09)" (archived 2026-08-11, NOT rewritten)
# PROVENANCE : none - no result file exists, and none should. The successors are
#              exp06_alpha_paired.sh, exp11_spectral_register_stereo.sh and
#              exp12_spectral_register_mono.sh
# =============================================================================
set -euo pipefail

cat <<'EOF'
Exp. 5 - ARCHIVED, NEVER RUN AS WRITTEN. This script executes nothing.

There is no command to relaunch because no code implements the pre-registered test:
the SIDE decision was dropped on 2026-08-10 and the 86/149 threshold with it. The
pre-registration stays on record, unrewritten.

What to run INSTEAD, if what you are after is one of its successors:

  the SIDE, on twin pairs, with the alpha laterality index      (Exp. 6 [H])
    bash scripts/replicate/exp06_alpha_paired.sh
    -> primary 23/44 = 0.5227 (threshold 28/44), mono control 24/42 (threshold 27/42)

  the REGISTER (high/low), Morlet band power + LDA, STEREO duos  (Exp. 11)
    bash scripts/replicate/exp11_spectral_register_stereo.sh
    -> 24/47 = 0.5106 (threshold 30/47, written 2026-08-09)

  the same on the MONO duos and on the pooled set                (Exp. 12)
    bash scripts/replicate/exp12_spectral_register_mono.sh
    -> mono 22/42 (threshold 27/42), pooled 46/89 (threshold 53/89), both from 2026-08-09

  band_power as a CCA VIEW on own-vs-other                       (Exp. 16 arm B)
    bash scripts/replicate/exp16b_ccaviews.sh
    -> 202/376 = 0.5372, the 208/376 bar not crossed
    NOTE: there `band_power` inside 1-8 Hz is ONLY delta+theta: alpha and beta are
       zero by construction and the module discards them. That negative holds for
       delta+theta, NOT for the wider-band proposal, which would require a NEW
       PRE-REGISTRATION, not one more flag on an old run.
EOF
exit 0
