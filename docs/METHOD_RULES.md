# Method rules

Twelve rules the code enforces. They are not generic good practice: each one is
here because it went wrong at least once in this project, and the incident is
written next to the rule on purpose. A rule without its scar is a rule that gets
skipped.

Source files refer to these by number, as `method rule N`.

Where a rule is enforced mechanically rather than by convention, the enforcing
file is named.

---

## 1. Verify state at the source, never from a note

A claim about what has been run, pushed or measured is a claim, not a fact.
Check it against the thing itself — `git log`, the run directory, the summary
file — before acting on it.

> On 2026-07-28 a session started from the premise that three commits were still
> unpushed. They had been on `origin` for a day. Acting on the note instead of on
> `git` would have produced a report that was false about a fact verifiable in
> two seconds. The same failure recurred seven times over the project, most
> expensively on 2026-08-16, when an experiment recorded as "prepared, not run"
> had in fact been executed three days earlier and its result falsified the
> published conclusion of the experiment before it.

## 2. Every experiment leaves a record that explains the CAUSE, not just the number

A record saying *"the run gave 0.3766"* is useless two weeks later. A useful one
says **why** it gave that number, **what was measured** and **what was only
inferred**, with an explicit line between the two: the number, what causes it
with the command that reproduces it, the alternative hypotheses tested and ruled
out with the number that rules them out, where measurement ends and
interpretation begins, and what is still open.

> On 2026-07-27 step C gave 0.3766. The number alone said "failed". Looking for
> its cause found that the model follows the prior of the training folds (89/124
> against a null of 0.4274) — and that mechanism became the bridge between
> Chapter 1 and Chapter 2, i.e. the chapter's contribution. The number was a
> failure; the cause was the result.

## 3. Every number has a positive control, crossed BEFORE the real number

Before believing a number — especially a bad one — you have to know the
instrument works. The positive control feeds the pipeline data where the answer
is there by construction and checks that it is recovered. The threshold is
declared **in the code, before the run**, and justified.

Cross it before looking at the real number. Look at the real number first and you
can no longer distinguish *"there is no signal"* from *"I wired it wrong"* — and
if the real number were good you would have every incentive to skip the control.

A control that passes says **only** that the wiring holds. It says nothing about
the real data.

> The step C positive control (148/154) did not only validate the wiring: running
> the diagnostic on top of it revealed that the null for "follows the prior" was
> not 0.5. Two already-written sub-analyses were retracted. The control caught the
> error, not a re-reading.

Enforced in: `madeeg_reconstruction.py --self_test`,
`madeeg_contrastive.py --self_test`, and the hard-exit gates inside
`madeeg_stem_separability.py`, `madeeg_exp14_tracking.py`,
`madeeg_exp15_differential.py`, `madeeg_exp16b_ccaviews.py`,
`madeeg_spectral_attention.py`.

## 4. Always print the null next to the statistic

Chance is never 0.5 by default: it is a quantity computed from the data alone. A
percentage without its null is not a result, it is a number. And the null pairs
with the **unit** of the accuracy standing next to it — a per-window accuracy and
a per-trial accuracy do not share one.

> Testing "follows the prior" against 0.5 instead of against
> P(prior = target) = 0.4274 produced the wrong conclusion that the model
> identifies the precise excerpt. Retracted. Separately, arm D's window-CV
> accuracy has a null of 0.5136 by construction while its trial-CV accuracy has a
> null of 0.500 exact; quoting one for both was caught twice.

## 5. The criterion is written BEFORE the number that has to cross it

Threshold, unit of analysis, statistical test and hyperparameters are fixed in
writing before the code that produces the figure exists. Afterwards they are not
touched.

And a second look at the same data is **declared**: it is exploratory by
construction, however good the number. The only route to a confirmatory claim is
a holdout that was never touched — and counting something on a holdout is already
looking at it.

> Step C gives 0.3766 against a threshold of 0.5714 written in advance. Flipping
> the sign of the rule would give 0.62, above threshold. That is exactly the move
> pre-registration exists to prevent.

## 6. Never flip the sign

If a below-chance result becomes good by inverting the decision rule, that is not
a result: it is exploitation of the k-fold structure. Explain why it is below
chance; do not turn it over.

## 7. Measure over the set. Do not generalise from one case

One trial, one file, one fold is not the dataset. If a property matters, count it
over all of them and print the count.

> A number about the dataset was once published by generalising from a single
> trial.

## 8. The default does not move. A change lives behind a flag

No number already reported may change because of a modification that does not
concern it. The old path stays literally the same code; the new one is an explicit
branch. And the flag is **asserted**: a mistyped value must BREAK, not silently
fall back to the old behaviour while passing for the new one.

> `clip_loss.py` produces every Chapter 1 number. Its self-check prints
> `0.628491` and `0.951610`: those are canaries. If they change, the refactor
> changed the science. The file is frozen at `bb016fd` for that reason.

Enforced in: the `--loss`, `--estimator`, `--target`, `--spatial`, `--eeg_clean`
and `--filters` asserts across the MAD-EEG drivers, and in
`scripts/replicate/canaries.sh`.

## 9. A control never writes over a result, and the caveat lives INSIDE the file

Separate directories for controls, variants and real runs. And the warning goes in
the summary file, above the numbers, not only on the terminal: the file outlives
the terminal.

> The step C control summary said `step C ... ABOVE CHANCE` without saying the
> EEG was synthetic. It was a number ready to be quoted by mistake. Fixed in
> `7021cec`.

## 10. Do not invent. If it was not verified, write "to be confirmed"

Numbers, paths, flags, citations, dates. A plausible invented path costs more than
a question. Several headers under `scripts/replicate/` carry an explicit
`TO BE CONFIRMED` for exactly this reason.

## 11. GPU steps are printed, not launched

A script that needs a GPU prints the exact command and stops. Nothing in
`scripts/replicate/` starts a training run implicitly; `exp18_matchmismatch.sh`
requires an explicit `--train-s1` / `--train-s2` flag to do so locally.

## 12. If you were wrong, retract in writing

A retracted claim stays on record, struck through, with the reason. It is not
deleted and history is not rewritten: it prevents the same error twice, and it
tells the reader how much to trust the other claims in the same document. The
project accumulated 28 such retractions; they are material for the methods
chapter, not embarrassment to be hidden.

---

## The reflex, in one line

> Before writing a number: what is its null, what is its positive control, and was
> the threshold written in advance?

If one of the three answers is missing, the number is not ready.
