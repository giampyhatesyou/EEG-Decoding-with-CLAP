#!/usr/bin/env python3
# CHANGED(baseline): new file — not in upstream. Interactive launcher that replaces the
#                    per-combination shell scripts in scripts/. It does not touch any
#                    training/evaluation code; it only assembles the same command line the
#                    scripts do and runs it as a subprocess.
"""Interactive launcher for the EEG<->audio attention experiments.

One entry point instead of the per-combination scripts in scripts/. It asks only
for the axes that actually vary between experiments and bakes the fixed Akama
protocol -- the ~23 flags every script repeats verbatim -- into a single
``PROTOCOL`` constant (the single source of truth). Every run prints the
equivalent ``python main.py ...`` / ``checkpoint_test.py ...`` command before
executing, so an interactive run stays reproducible as a one-liner and matches
scripts/train.sh / test.sh flag-for-flag.

This is a thin subprocess wrapper: it changes nothing in main.py /
checkpoint_test.py / the loss / the dataset. The science is identical because the
emitted command line is identical to the canonical scripts.

Usage:
    cd src && python run.py            # interactive
    python run.py --dry-run            # print the commands it would build, no prompts
    python run.py --selftest           # assert the assembled command matches train.sh/test.sh
"""
import argparse
import glob
import os
import shlex
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# --- The fixed Akama protocol --------------------------------------------------
# The flags every canonical script repeats verbatim. This dict is the single
# source of truth: change a hyperparameter here and every command the CLI builds
# changes with it. Each value matches the paper's experimental setup.
PROTOCOL = {
    "dataset": "preprocessing_eegmusic",
    "test_dataset": "preprocessing_eegmusic_test",
    "max_epochs": 1000,             # paper: max 1000 epochs (EarlyStopping patience 10, min 50)
    "batch_size": 8,                # paper: batch 8 (x accumulate_grad_batches 6)
    "eeg_length": 768,              # 3 s EEG input
    "loss_function": "clip_loss",
    "eeg_normalization": "MetaAI",  # RobustScaler + clamp (Defossez et al.)
    "clamp_value": 20,              # paper: clamp +/-20 after RobustScaler
    "learning_rate": 0.003,
    "supervised": 1,
    "dim_reduction": 1,
    "split_seed": 42,               # data-split seed (fixed across runs)
    "detach_z_c": 0,
    "window_size": 1280,            # 5 s sliding window (train/valid)
    "stride": 256,                  # 1 s stride
    "test_window_size": 768,        # 3 s test window
    "test_stride": 256,
    "start_position": 0,
    "key": "all",                   # all four target tasks
}
ATTENTION_VALUES = [4, 5]           # high-attention = self-reported score 4-5
DEFAULT_SEED = 42                   # optimisation seed (vary for error bars)

# --- The axes that actually vary ----------------------------------------------
PHASES = {
    "train": "Train a model (main.py)",
    "test": "Evaluate a checkpoint (checkpoint_test.py)",
    "train+test": "Train then evaluate on the same config",
    "experiments": "Supervisor diagnostic controls (EEG-only / audio-only classifiers)",
    "sanity": "Negative-control sweep (none / labels / audio_pair)",
}
# Model axis = the architectural switch (--audio_repr). The keys are the flag
# values passed to main.py verbatim; later models join here as they are wired in.
AUDIO_REPRS = {
    "raw": "akama baseline - 4 independent SampleCNN2DEEG audio encoders",
    "clap": "CLAP - frozen LAION-CLAP backbone + shared projection head",
}
# Planned models shown (greyed-out) so the roadmap is visible, but not selectable
# until main.py's audio_repr switch supports them — otherwise the value would
# silently fall through to the raw path and run the wrong model.
AUDIO_REPRS_DISABLED = {
    "clap+spectraclip": "planned extension - not implemented in main.py yet",
}
# Supervisor-proposed diagnostic controls (experiments #1/#2), kept separate from
# the contrastive baseline and reached through the dedicated "experiments" phase
# rather than the generic train/test flows. Each pins a single training objective
# (a SEPARATE cross-entropy LightningModule; the contrastive loss/metric/split are
# untouched). classify_eeg uses no audio encoder, so its audio_repr is irrelevant
# (pinned raw, CLAP never loaded); classify_audio's audio_repr selects how the song
# stems are encoded. The user-facing names mirror what they isolate.
EXPERIMENTS = {
    "control_no_audio": {
        "objective": "classify_eeg",
        "needs_audio_repr": False,
        "desc": "Control without audio - supervised EEG-only classifier (#2)",
    },
    "control_no_eeg": {
        "objective": "classify_audio",
        "needs_audio_repr": True,
        "desc": "Control without EEG - audio-only classifier, negative control (#1)",
    },
}
EXP_ACTIONS = {
    "train": "Train the control (main.py)",
    "test": "Evaluate a control checkpoint (checkpoint_test.py)",
    "train+test": "Train then evaluate the control",
}
CV_MODES = {
    "within": "within-subject, 15s-window split (songs shared train/test; paper Table 1)",
    "leave_subject_out": "cross-subject LOSO (paper Table 2; paper subjects 3,7,2)",
    "leave_song_out": "cross-song (audit extension, not in the paper)",
}
# The EEG-audio delay ablation (all-0 / all-200) is no longer asked: the launcher
# always uses the headline all-0 (no-delay) model. The 200 ms variant is still
# reachable via scripts/train.sh --shifting_time 200 if ever needed.


# --- Command assembly (pure; no third-party imports) ---------------------------
def build_command(entrypoint, *, audio_repr, cv_mode, cv_held_out_id,
                  training_date, shifting_time=0, seed=DEFAULT_SEED, devices=1,
                  workers=None, checkpoint_path=None, shuffle_test_mode=None,
                  test_breakdown=None, objective="contrastive"):
    """Assemble the exact ``python <entrypoint> ...`` argv from PROTOCOL + axes.

    ``entrypoint`` is "main.py" (train) or "checkpoint_test.py" (test). The
    emitted flags match scripts/train.sh / test.sh, with audio_repr / cv_* passed
    explicitly instead of relying on the YAML defaults (same values). ``--objective``
    is emitted only for the non-default (classify_*) objectives, so the contrastive
    command stays byte-identical to scripts/train.sh and the selftest keeps passing.
    """
    cmd = [sys.executable, "-u", entrypoint]
    for key, value in PROTOCOL.items():
        cmd += [f"--{key}", str(value)]
    cmd += ["--attention_values", *[str(v) for v in ATTENTION_VALUES]]
    cmd += ["--devices", str(devices)]
    cmd += ["--shifting_time", str(shifting_time)]
    cmd += ["--seed", str(seed)]
    cmd += ["--audio_repr", audio_repr]
    if objective and objective != "contrastive":
        cmd += ["--objective", objective]
    cmd += ["--cv_mode", cv_mode, "--cv_held_out_id", str(cv_held_out_id)]
    cmd += ["--training_date", training_date]
    if workers is not None:
        cmd += ["--workers", str(workers)]
    if checkpoint_path:
        cmd += ["--checkpoint_path", str(checkpoint_path)]
    if shuffle_test_mode:
        cmd += ["--shuffle_test_mode", shuffle_test_mode]
    if test_breakdown is not None:
        cmd += ["--test_breakdown", str(test_breakdown)]
    return cmd


def format_cmdline(cmd):
    """One-line, copy-pasteable rendering of an argv list."""
    return " ".join(shlex.quote(c) for c in cmd)


def run_subprocess(cmd, *, env=None):
    """Echo the equivalent command line, then run it from src/."""
    print("\n$ (cd src && " + format_cmdline(cmd) + ")\n", flush=True)
    return subprocess.run(cmd, cwd=str(SRC_DIR), env=env).returncode


# --- Result lookup / parsing ---------------------------------------------------
def _log_dir():
    """Resolve the results dir the way main.py does, with a plain fallback."""
    try:
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))
        from utils import paths
        resolved = Path(paths.resolve_log_dir("../results"))
        return resolved if resolved.is_absolute() else (SRC_DIR / resolved).resolve()
    except Exception:
        return PROJECT_ROOT / "results"


# --- Interactive helpers (questionary imported lazily) -------------------------
def _questionary():
    try:
        import questionary
        return questionary
    except ImportError:
        sys.exit("This CLI needs 'questionary':\n"
                 "    python -m pip install questionary")


# Navigation sentinels for the back-stack wizard (questionary has no native
# ESC-to-previous; we offer a "↩ back" menu entry and treat Ctrl-C as back).
BACK = object()
SKIP = object()


def pick(message, options, *, disabled=None, back=False):
    """Single-choice select over an ordered {value: description} dict.

    With back=True a '↩ back' entry is offered and Ctrl-C returns BACK; with
    back=False (the default for non-wizard callers) Ctrl-C exits.
    """
    questionary = _questionary()
    choices = []
    if back:
        choices.append(questionary.Choice(title="↩ back", value=BACK))
    choices += [questionary.Choice(title=f"{value}  -  {desc}", value=value)
                for value, desc in options.items()]
    for value, reason in (disabled or {}).items():
        choices.append(questionary.Choice(title=f"{value}  -  {reason}",
                                          value=value, disabled=reason))
    answer = questionary.select(message, choices=choices).ask()
    if answer is None:
        return BACK if back else sys.exit("aborted")
    return answer


def _ask(question, *, back):
    """Run a questionary prompt, mapping Ctrl-C to BACK (with a 'back' message
    instead of the default 'Cancelled by user') when the caller allows it."""
    answer = question.ask(kbi_msg="↩ back") if back else question.ask()
    if answer is None:
        return BACK if back else sys.exit("aborted")
    return answer


def ask_int(message, default=None, *, back=False):
    questionary = _questionary()
    hint = "  [Ctrl-C: back]" if back else ""
    while True:
        raw = _ask(questionary.text(
            message + hint, default=("" if default is None else str(default))), back=back)
        if raw is BACK:
            return BACK
        try:
            return int(raw)
        except ValueError:
            print("  please enter an integer")


def ask_text(message, default="", *, back=False):
    questionary = _questionary()
    answer = _ask(questionary.text(message + ("  [Ctrl-C: back]" if back else ""),
                                   default=default), back=back)
    return answer if answer is BACK else answer.strip()


def ask_path(message, *, back=False):
    questionary = _questionary()
    answer = _ask(questionary.path(message + ("  [Ctrl-C: back]" if back else "")), back=back)
    return answer if answer is BACK else answer.strip()


def ask_confirm(message, default=True):
    questionary = _questionary()
    answer = questionary.confirm(message, default=default).ask()
    if answer is None:
        sys.exit("aborted")
    return answer


def _wizard(steps, state):
    """Run a list of step(state) functions with back navigation. Each step
    returns BACK (return to the previous prompted step), SKIP (auto-advance,
    not recorded so back skips it), or None (advance, recorded). Returns state,
    or None if the user backed out past the first step.
    """
    history = []
    i = 0
    while i < len(steps):
        res = steps[i](state)
        if res is BACK:
            if not history:
                return None
            i = history.pop()
            continue
        if res is not SKIP:
            history.append(i)
        i += 1
    return state


# --- Axis prompts --------------------------------------------------------------
def ask_eval_setting(*, back=False):
    """Return (cv_mode, cv_held_out_id, label), or BACK."""
    while True:
        cv_mode = pick("Evaluation setting:", CV_MODES, back=back)
        if cv_mode is BACK:
            return BACK
        if cv_mode == "within":
            return cv_mode, -1, "within"
        prompt, dflt, lbl = (("Held-out subject id (paper used 3, 7, 2):", 3, "loso_sub")
                             if cv_mode == "leave_subject_out"
                             else ("Held-out song id:", 36, "lso_song"))
        sid = ask_int(prompt, default=dflt, back=back)
        if sid is BACK:
            continue  # re-pick cv_mode
        return cv_mode, sid, f"{lbl}{sid}"


def suggest_tag(audio_repr, eval_label, objective="contrastive"):
    if objective == "classify_eeg":
        return f"clf_eeg_{eval_label}"         # no audio encoder -> audio_repr irrelevant
    if objective == "classify_audio":
        return f"clf_audio_{audio_repr}_{eval_label}"
    return f"{audio_repr}_{eval_label}"


def ask_train_axes(*, objective="contrastive", audio_repr=None, back=False):
    """Collect the training axes via a back-navigable wizard. Returns an axes dict
    or BACK (backed out past the first prompted step).

    The generic train flow is contrastive-only, so ``objective`` defaults to
    "contrastive" and is never prompted here; the two supervised controls reach
    this with their objective (and, for classify_audio, audio_repr) pinned by the
    experiments phase, which skips the corresponding prompts.
    """
    st = {"objective": objective, "audio_repr": audio_repr}

    def s_audio(s):
        if s["objective"] == "classify_eeg":
            s["audio_repr"] = "raw"   # no audio encoder; pin raw (CLAP never loaded)
            return SKIP
        if s["audio_repr"] is not None:
            return SKIP               # pinned by the caller (classify_audio)
        v = pick("Model:", AUDIO_REPRS, disabled=AUDIO_REPRS_DISABLED, back=back)
        if v is BACK:
            return BACK
        s["audio_repr"] = v

    def s_eval(s):
        r = ask_eval_setting(back=back)
        if r is BACK:
            return BACK
        s["cv_mode"], s["cv_held_out_id"], s["eval_label"] = r

    def s_seed(s):
        v = ask_int("Optimisation seed:", default=DEFAULT_SEED, back=back)
        if v is BACK:
            return BACK
        s["seed"] = v

    def s_label(s):
        v = ask_text("Run label (--training_date):",
                     default=suggest_tag(s["audio_repr"], s["eval_label"], s["objective"]),
                     back=back)
        if v is BACK:
            return BACK
        s["training_date"] = v

    if _wizard([s_audio, s_eval, s_seed, s_label], st) is None:
        return BACK
    return dict(objective=st["objective"], audio_repr=st["audio_repr"],
                cv_mode=st["cv_mode"], cv_held_out_id=st["cv_held_out_id"],
                seed=st["seed"], training_date=st["training_date"])


# Argparse defaults, for runs whose hparams.yaml predates these flags (e.g. an
# early within-subject run that never logged cv_mode is, by definition, within).
_HP_DEFAULTS = {"audio_repr": None, "cv_mode": "within",
                "cv_held_out_id": "-1", "shifting_time": "0",
                "objective": "contrastive"}  # runs predating the flag are contrastive


def _run_config(ckpt_path):
    """Read (audio_repr, cv_mode, cv_held_out_id, shifting_time) from the version
    dir's hparams.yaml. Missing keys fall back to the argparse defaults so older
    within-subject runs still match a within query; audio_repr stays None when
    unreadable, so such runs drop out of any representation-filtered list.
    """
    cfg = dict(_HP_DEFAULTS)
    hp = Path(ckpt_path).parents[1] / "hparams.yaml"
    try:
        for line in hp.read_text().splitlines():
            s = line.strip()
            for k in cfg:
                if s.startswith(k + ":"):
                    cfg[k] = s.split(":", 1)[1].strip().strip("'\"")
    except Exception:
        pass
    return cfg


def runs_with_checkpoint(audio_repr=None, cv_mode=None, cv_held_out_id=None,
                         shifting_time=None, objective=None):
    """training_dates under results/ with a best-checkpoint.ckpt whose training
    config matches the given axes (read from each run's hparams.yaml).

    Matching on (objective, audio_repr, cv_mode, cv_held_out_id, shifting_time) is a
    reproducibility guard: a checkpoint is only offered for a test whose split it
    was actually trained for. A leave_subject_out=3 model is therefore not listed
    when testing within-subject or a different held-out subject — which would
    otherwise evaluate that model on data it saw during training. Likewise a
    classify_* checkpoint is not offered for a contrastive evaluation (its
    topology and metric differ), and vice versa.
    """
    want = {"objective": objective, "audio_repr": audio_repr, "cv_mode": cv_mode,
            "cv_held_out_id": None if cv_held_out_id is None else str(cv_held_out_id),
            "shifting_time": None if shifting_time is None else str(shifting_time)}
    bases = {PROJECT_ROOT / "results"}
    try:
        bases.add(_log_dir())
    except Exception:
        pass
    tds = set()
    for base in bases:
        for h in glob.glob(str(base / "*" / "nmed-CL-*" / "version_*"
                                / "checkpoints" / "best-checkpoint.ckpt")):
            cfg = _run_config(h)
            if all(v is None or cfg.get(k) == v for k, v in want.items()):
                tds.add(Path(h).parts[-5])
    return sorted(tds)


def resolve_checkpoint(training_date):
    """Newest best-checkpoint.ckpt under results/<training_date>/, or None.
    Mirrors checkpoint_test.py's auto-discovery so the CLI can show the exact
    path it will load before running (transparency, not a black box).
    """
    hits = []
    bases = {PROJECT_ROOT / "results"}
    try:
        bases.add(_log_dir())
    except Exception:
        pass
    for base in bases:
        hits += glob.glob(str(base / training_date / "nmed-CL-*" / "version_*"
                              / "checkpoints" / "best-checkpoint.ckpt"))
    return max(hits, key=os.path.getmtime) if hits else None


def authors_checkpoints_for(audio_repr, cv_mode, cv_held_out_id):
    """Authors' released checkpoints in checkpoints/ valid for this test split.
    They published only raw models: model-all0 (within), model-sub{N} (LOSO N).
    Returns a list of (label, repo-relative path).
    """
    if audio_repr not in (None, "raw"):
        return []
    want = None
    if cv_mode in (None, "within"):
        want = "model-all0.ckpt"
    elif cv_mode == "leave_subject_out":
        want = f"model-sub{cv_held_out_id}.ckpt"
    if want and (PROJECT_ROOT / "checkpoints" / want).exists():
        return [(f"{want}  (authors' baseline)", f"../checkpoints/{want}")]
    return []


def ask_checkpoint(default_label, *, audio_repr=None, cv_mode=None,
                   cv_held_out_id=None, shifting_time=None, objective=None, back=False):
    """Return (checkpoint_path_or_None, training_date_override_or_None), or BACK.

    The "existing" list merges results/ runs matching the test split (the
    reproducibility guard) with the authors' released checkpoints valid for that
    split, so the baseline is selectable from the launcher. Picking a run lets
    checkpoint_test.py auto-discover its checkpoint under results/<run>/.

    Back navigation is two-level: backing out of the path/list sub-prompt returns
    to the source choice ("existing" vs "path"); only backing out of that source
    choice (offered when ``back=True``) returns BACK to the caller.
    """
    setting = "within" if cv_mode in (None, "within") else f"{cv_mode}={cv_held_out_id}"
    obj_tag = None if objective in (None, "contrastive") else objective
    desc = " ".join(x for x in (obj_tag, audio_repr, setting) if x)
    questionary = _questionary()
    while True:
        how = pick("Checkpoint to evaluate:", {
            "existing": f"pick a matching run / authors' checkpoint [{desc}]",
            "path": "type an explicit .ckpt path",
        }, back=back)
        if how is BACK:
            return BACK
        if how == "path":
            # back here returns to the source choice above, not out to the caller.
            p = ask_path("Path to .ckpt:", back=True)
            if p is BACK:
                continue
            return (p, None)
        runs = runs_with_checkpoint(audio_repr, cv_mode, cv_held_out_id, shifting_time, objective)
        # The authors released only raw contrastive checkpoints; never offer them
        # for a classify_* evaluation.
        authors = (authors_checkpoints_for(audio_repr, cv_mode, cv_held_out_id)
                   if objective in (None, "contrastive") else [])
        if not runs and not authors:
            print(f"  no checkpoint matches [{desc}] — give an explicit path")
            p = ask_path("Path to .ckpt:", back=True)
            if p is BACK:
                continue
            return (p, None)
        choices = [questionary.Choice(title="↩ back", value=BACK)]
        for r in runs:
            choices.append(questionary.Choice(title=f"run: {r}", value=("run", r)))
        for label, p in authors:
            choices.append(questionary.Choice(title=label, value=("path", p)))
        sel = questionary.select(f"Which checkpoint?  [{desc}]", choices=choices).ask()
        if sel is None or sel is BACK:
            continue  # back -> re-show the source choice
        kind, val = sel
        return (None, val) if kind == "run" else (val, None)


# --- Phase flows ---------------------------------------------------------------
def flow_train(axes=None):
    if axes is None:
        axes = ask_train_axes(objective="contrastive", back=True)
        if axes is BACK:
            return BACK  # backed out of the train wizard -> phase menu
    cmd = build_command("main.py", **axes)
    if ask_confirm("Run training now?"):
        return run_subprocess(cmd), axes
    print("\nNot run. Equivalent command:\n  " + format_cmdline(cmd))
    return None, axes


def _run_test(axes, *, ckpt, shuffle):
    """Print a transparency banner (model / split / the checkpoint that will be
    loaded), then build and optionally run the evaluation."""
    setting = ("within" if axes["cv_mode"] in (None, "within")
               else f'{axes["cv_mode"]}={axes["cv_held_out_id"]}')
    _obj = axes.get("objective", "contrastive")
    model_desc = axes['audio_repr'] if _obj == "contrastive" else _obj
    print(f"\n→ Testing: {model_desc} model  |  {setting}  |  run '{axes['training_date']}'")
    if ckpt:
        print(f"  checkpoint: {ckpt}")
    else:
        rp = resolve_checkpoint(axes["training_date"])
        if rp:
            try:
                rp = str(Path(rp).relative_to(PROJECT_ROOT))
            except Exception:
                pass
        print(f"  checkpoint (auto-discovered): {rp or '(none found — checkpoint_test will error)'}")
    cmd = build_command("checkpoint_test.py", checkpoint_path=ckpt,
                        shuffle_test_mode=(None if shuffle == "none" else shuffle),
                        test_breakdown=1, **axes)
    if ask_confirm("Run evaluation now?"):
        return run_subprocess(cmd), axes
    print("\nNot run. Equivalent command:\n  " + format_cmdline(cmd))
    return None, axes


def flow_test(axes=None, *, ask_shuffle=True, objective="contrastive", audio_repr=None):
    # train+test path: the model was just trained under axes[training_date];
    # auto-discover its checkpoint, no prompts.
    if axes is not None:
        return _run_test(axes, ckpt=None, shuffle="none")

    # objective/audio_repr default to the contrastive baseline for the generic
    # test phase; the experiments phase pins them to a supervised control.
    st = {"objective": objective, "audio_repr": audio_repr}

    def s_audio(s):
        if s["objective"] == "classify_eeg":
            s["audio_repr"] = "raw"  # no audio encoder used; pin raw (CLAP never loaded)
            return SKIP
        if s["audio_repr"] is not None:
            return SKIP              # pinned by the caller (classify_audio)
        v = pick("Model:", AUDIO_REPRS, disabled=AUDIO_REPRS_DISABLED, back=True)
        if v is BACK:
            return BACK
        s["audio_repr"] = v

    def s_eval(s):
        r = ask_eval_setting(back=True)
        if r is BACK:
            return BACK
        s["cv_mode"], s["cv_held_out_id"], s["eval_label"] = r

    def s_ckpt(s):
        # shifting_time is pinned to 0 (the headline all-0 model), so the
        # checkpoint-matching guard only offers all-0 runs.
        res = ask_checkpoint(
            suggest_tag(s["audio_repr"], s["eval_label"], s["objective"]),
            audio_repr=s["audio_repr"], cv_mode=s["cv_mode"],
            cv_held_out_id=s["cv_held_out_id"], shifting_time=0,
            objective=s["objective"], back=True)
        if res is BACK:
            return BACK
        s["ckpt"], s["run_override"] = res

    def s_label(s):
        if s.get("run_override") is not None:
            s["training_date"] = s["run_override"]  # output into the picked run's dir
            return SKIP
        v = ask_text("Run label (--training_date, output dir):",
                     default=suggest_tag(s["audio_repr"], s["eval_label"], s["objective"]),
                     back=True)
        if v is BACK:
            return BACK
        s["training_date"] = v

    def s_shuffle(s):
        # The negative-control shuffles are implemented only in the contrastive
        # test_step; the supervised classifier ignores shuffle_test_mode, so never
        # offer it for a classify_* checkpoint (it would be a misleading no-op).
        if not ask_shuffle or s["objective"] != "contrastive":
            s["shuffle"] = "none"
            return SKIP
        v = pick("Evaluation mode:", {
            "none": "normal evaluation",
            "labels": "negative control: shuffle task labels (accuracy must -> chance)",
            "audio_pair": "negative control: shuffle EEG<->audio pairing (-> chance)",
        }, back=True)
        if v is BACK:
            return BACK
        s["shuffle"] = v

    if _wizard([s_audio, s_eval, s_ckpt, s_label, s_shuffle], st) is None:
        return BACK  # backed out past the first step -> phase menu

    # No seed prompt on the test path: with a loaded checkpoint and shuffle=False a
    # normal evaluation is deterministic regardless of seed, so build_command's
    # default (the protocol's 42, also the seed of the negative-control shuffle)
    # is used. Training keeps the seed axis (error bars); testing does not need it.
    axes = dict(objective=st["objective"], audio_repr=st["audio_repr"], cv_mode=st["cv_mode"],
                cv_held_out_id=st["cv_held_out_id"], training_date=st["training_date"])
    return _run_test(axes, ckpt=st["ckpt"], shuffle=st["shuffle"])


def flow_train_test():
    axes = ask_train_axes(objective="contrastive", back=True)
    if axes is BACK:
        return BACK  # backed out of the train wizard -> phase menu
    rc, _ = flow_train(axes)
    if rc not in (0, None):
        print("training failed; skipping test")
        return
    flow_test(axes, ask_shuffle=False)


def flow_experiments():
    """Supervisor diagnostic controls — the two unimodal supervised classifiers.

    A thin wrapper that pins ``--objective`` (and ``audio_repr`` where it is
    meaningful) and then reuses the very same train/test machinery as the
    contrastive flows: the emitted command line, the run tags and the
    checkpoint-matching guard are identical to driving these objectives by hand.
    This entry only makes the two controls discoverable as named experiments
    instead of a buried objective sub-prompt. Returns BACK (-> phase menu) or None.
    """
    while True:
        st = {}

        def s_which(s):
            v = pick("Diagnostic control:",
                     {k: e["desc"] for k, e in EXPERIMENTS.items()}, back=True)
            if v is BACK:
                return BACK
            s["objective"] = EXPERIMENTS[v]["objective"]
            s["needs_repr"] = EXPERIMENTS[v]["needs_audio_repr"]

        def s_repr(s):
            # classify_eeg uses no audio encoder; pin raw so CLAP is never loaded.
            if not s["needs_repr"]:
                s["audio_repr"] = "raw"
                return SKIP
            v = pick("Stem encoder:", AUDIO_REPRS,
                     disabled=AUDIO_REPRS_DISABLED, back=True)
            if v is BACK:
                return BACK
            s["audio_repr"] = v

        def s_action(s):
            v = pick("Action:", EXP_ACTIONS, back=True)
            if v is BACK:
                return BACK
            s["action"] = v

        if _wizard([s_which, s_repr, s_action], st) is None:
            return BACK  # backed out past the first step -> phase menu

        obj, repr_, action = st["objective"], st["audio_repr"], st["action"]
        if action == "test":
            if flow_test(objective=obj, audio_repr=repr_) is BACK:
                continue  # backed out of the test wizard -> re-pick the control
            return
        axes = ask_train_axes(objective=obj, audio_repr=repr_, back=True)
        if axes is BACK:
            continue  # backed out of the train config -> re-pick the control
        rc, _ = flow_train(axes)
        if action == "train+test":
            if rc not in (0, None):
                print("training failed; skipping test")
            else:
                flow_test(axes, ask_shuffle=False)
        return


def flow_sanity():
    audio_repr = pick("Model of the checkpoint:", AUDIO_REPRS,
                      disabled=AUDIO_REPRS_DISABLED)
    ckpt = ask_text("Checkpoint (repo-relative or absolute):",
                    default="checkpoints/model-all0.ckpt")
    tag = ask_text("Tag (training_date prefix):", default="sanity")
    script = PROJECT_ROOT / "scripts" / "test_sanity.sh"
    env = dict(os.environ, CKPT=ckpt, TAG=tag, AUDIO_REPR=audio_repr)
    print(f"\n$ CKPT={shlex.quote(ckpt)} TAG={shlex.quote(tag)} "
          f"AUDIO_REPR={audio_repr} bash scripts/test_sanity.sh\n")
    subprocess.run(["bash", str(script)], cwd=str(PROJECT_ROOT), env=env)


# --- Non-interactive helpers (for verification) --------------------------------
def cmd_dry_run():
    """Print the commands the CLI would build for a representative matrix."""
    samples = [
        ("train  raw  within", build_command(
            "main.py", audio_repr="raw", cv_mode="within", cv_held_out_id=-1,
            training_date="raw_within")),
        ("train  clap within", build_command(
            "main.py", audio_repr="clap", cv_mode="within", cv_held_out_id=-1,
            training_date="clap_within")),
        ("test   raw  within", build_command(
            "checkpoint_test.py", audio_repr="raw", cv_mode="within", cv_held_out_id=-1,
            training_date="test", checkpoint_path="../checkpoints/model-all0.ckpt",
            test_breakdown=1)),
        ("test   raw  loso3 ", build_command(
            "checkpoint_test.py", audio_repr="raw", cv_mode="leave_subject_out",
            cv_held_out_id=3, training_date="raw_loso_sub3", test_breakdown=1)),
        ("train  clf_eeg within", build_command(
            "main.py", audio_repr="raw", cv_mode="within", cv_held_out_id=-1,
            objective="classify_eeg", training_date="clf_eeg_within")),
        ("train  clf_audio raw within", build_command(
            "main.py", audio_repr="raw", cv_mode="within", cv_held_out_id=-1,
            objective="classify_audio", training_date="clf_audio_raw_within")),
    ]
    for label, cmd in samples:
        print(f"\n# {label}\n{format_cmdline(cmd)}")


def cmd_selftest():
    """Assert build_command reproduces scripts/train.sh and test.sh flag values."""
    # Flag:value pairs that scripts/train.sh passes (audio_repr / cv_* are left to
    # the YAML defaults in the script: raw / within / -1).
    expected_train = {
        "dataset": "preprocessing_eegmusic", "test_dataset": "preprocessing_eegmusic_test",
        "devices": "1", "max_epochs": "1000", "batch_size": "8", "eeg_length": "768",
        "loss_function": "clip_loss", "eeg_normalization": "MetaAI", "clamp_value": "20",
        "learning_rate": "0.003", "supervised": "1", "dim_reduction": "1",
        "shifting_time": "0", "split_seed": "42", "detach_z_c": "0", "window_size": "1280",
        "stride": "256", "test_window_size": "768", "test_stride": "256", "seed": "42",
        "start_position": "0", "key": "all", "training_date": "test",
    }
    cmd = build_command("main.py", audio_repr="raw", cv_mode="within",
                        cv_held_out_id=-1, training_date="test")
    got = _flags_to_dict(cmd)
    ok = True
    for flag, value in expected_train.items():
        if got.get(flag) != value:
            ok = False
            print(f"  MISMATCH --{flag}: train.sh={value!r} build_command={got.get(flag)!r}")
    if got.get("attention_values") != "4 5":
        ok = False
        print(f"  MISMATCH --attention_values: train.sh='4 5' build_command={got.get('attention_values')!r}")
    # The CLI additionally passes the script's YAML defaults explicitly:
    for flag, value in (("audio_repr", "raw"), ("cv_mode", "within"), ("cv_held_out_id", "-1")):
        if got.get(flag) != value:
            ok = False
            print(f"  MISMATCH --{flag}: expected {value!r} got {got.get(flag)!r}")
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _flags_to_dict(cmd):
    """Collapse an argv into {flag: 'space joined values'} for assertions."""
    out, current = {}, None
    for token in cmd[3:]:  # skip python -u entrypoint
        if token.startswith("--"):
            current = token[2:]
            out[current] = ""
        elif current is not None:
            out[current] = (out[current] + " " + token).strip()
    return out


# --- Entry point ---------------------------------------------------------------
def interactive():
    print("EEG attention experiments - interactive launcher")
    print("(fixed Akama protocol is baked in; you pick only what varies)")
    print("(step back with the '↩ back' menu entry, or Ctrl-C in a text/path prompt)\n")
    dispatch = {
        "train": flow_train,
        "test": flow_test,
        "train+test": flow_train_test,
        "experiments": flow_experiments,
        "sanity": flow_sanity,
    }
    while True:
        phase = pick("Phase:", PHASES, back=True)
        if phase is BACK:
            return
        if dispatch[phase]() is BACK:
            continue  # the flow backed out -> re-show the phase menu
        return


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands the CLI would build, then exit")
    parser.add_argument("--selftest", action="store_true",
                        help="assert the assembled command matches train.sh/test.sh")
    args = parser.parse_args()
    if args.dry_run:
        cmd_dry_run()
        return 0
    if args.selftest:
        return cmd_selftest()
    interactive()
    return 0


if __name__ == "__main__":
    sys.exit(main())
