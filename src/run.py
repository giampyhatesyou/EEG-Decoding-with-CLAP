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
import csv as _csv
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
    "compare": "Direct baseline<->CLAP comparison (side-by-side table)",
    "reproduce-paper": "Replicate Akama Table 1 + Table 2 from the authors' checkpoints",
    "sanity": "Negative-control sweep (none / labels / audio_pair)",
}
AUDIO_REPRS = {
    "raw": "Akama baseline - 4 independent SampleCNN2DEEG audio encoders",
    "clap": "Extension - frozen LAION-CLAP backbone + shared projection head",
}
CV_MODES = {
    "within": "within-subject, new songs (paper Table 1)",
    "leave_subject_out": "cross-subject LOSO (paper Table 2; paper subjects 3,7,2)",
    "leave_song_out": "cross-song (audit extension, not in the paper)",
}
# EEG-audio delay variants (paper ablation). attn-0 (train on high-attention
# trials only) is NOT wired in main.py, so it is offered but disabled.
DELAY_VARIANTS = {
    "all-0": (0, "all trials, no EEG-audio delay (headline / best model)"),
    "all-200": (200, "all trials, 200 ms EEG-audio delay (paper ablation)"),
}
DELAY_DISABLED = {
    "attn-0": "train on high-attention trials only - not implemented in main.py",
}

# Reference numbers for the within-subject comparison table.
REFERENCE = {
    ("raw", "within"): {"all": 0.8641, "attn": 0.8454, "src": "Akama Table 1"},
    ("clap", "within"): {"all": 0.9237, "attn": None, "src": "thesis CLAP"},
}


# --- Command assembly (pure; no third-party imports) ---------------------------
def build_command(entrypoint, *, audio_repr, cv_mode, cv_held_out_id,
                  training_date, shifting_time=0, seed=DEFAULT_SEED, devices=1,
                  workers=None, checkpoint_path=None, shuffle_test_mode=None,
                  test_breakdown=None):
    """Assemble the exact ``python <entrypoint> ...`` argv from PROTOCOL + axes.

    ``entrypoint`` is "main.py" (train) or "checkpoint_test.py" (test). The
    emitted flags match scripts/train.sh / test.sh, with audio_repr / cv_* passed
    explicitly instead of relying on the YAML defaults (same values).
    """
    cmd = [sys.executable, "-u", entrypoint]
    for key, value in PROTOCOL.items():
        cmd += [f"--{key}", str(value)]
    cmd += ["--attention_values", *[str(v) for v in ATTENTION_VALUES]]
    cmd += ["--devices", str(devices)]
    cmd += ["--shifting_time", str(shifting_time)]
    cmd += ["--seed", str(seed)]
    cmd += ["--audio_repr", audio_repr]
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


def find_summary(training_date):
    """Newest test_breakdown_summary.txt written under a training_date, or None."""
    hits = []
    for root in {PROJECT_ROOT / "results", _log_dir()}:
        hits += glob.glob(str(root / training_date / "nmed-CL-*" / "version_*"
                               / "test_breakdown_summary.txt"))
    return max(hits, key=os.path.getmtime) if hits else None


def parse_summary(path):
    """Pull {'all': float|None, 'attn': float|None} from a summary file."""
    out = {"all": None, "attn": None}
    for line in Path(path).read_text().splitlines():
        if "global accuracy (records, all)" in line:
            out["all"] = _safe_float(line.rsplit(":", 1)[-1])
        elif "global accuracy (records, attn)" in line:
            out["attn"] = _safe_float(line.rsplit(":", 1)[-1])
    return out


def macro_accuracy(summary_path):
    """Macro accuracy = mean of the 4 per-task accuracies, or None."""
    csv_path = Path(summary_path).with_name("test_per_task_summary.csv")
    if not csv_path.exists():
        return None
    accs = []
    with open(csv_path) as fh:
        for row in _csv.DictReader(fh):
            value = _safe_float(row.get("accuracy"))
            if value is not None:
                accs.append(value)
    return sum(accs) / len(accs) if accs else None


def _safe_float(text):
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def _fmt(value):
    return f"{value:.4f}" if isinstance(value, float) else "--"


# --- Interactive helpers (questionary imported lazily) -------------------------
def _questionary():
    try:
        import questionary
        return questionary
    except ImportError:
        sys.exit("This CLI needs 'questionary' and 'rich':\n"
                 "    python -m pip install questionary rich")


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


def ask_int(message, default=None, *, back=False):
    questionary = _questionary()
    hint = "  [Ctrl-C: back]" if back else ""
    while True:
        raw = questionary.text(message + hint,
                               default=("" if default is None else str(default))).ask()
        if raw is None:
            return BACK if back else sys.exit("aborted")
        try:
            return int(raw)
        except ValueError:
            print("  please enter an integer")


def ask_text(message, default="", *, back=False):
    questionary = _questionary()
    answer = questionary.text(message + ("  [Ctrl-C: back]" if back else ""),
                              default=default).ask()
    if answer is None:
        return BACK if back else sys.exit("aborted")
    return answer.strip()


def ask_path(message, *, back=False):
    questionary = _questionary()
    answer = questionary.path(message + ("  [Ctrl-C: back]" if back else "")).ask()
    if answer is None:
        return BACK if back else sys.exit("aborted")
    return answer.strip()


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


def ask_delay(*, back=False):
    """Return shifting_time (0 or 200), or BACK."""
    variant = pick("Model variant (EEG-audio delay):",
                   {k: v[1] for k, v in DELAY_VARIANTS.items()},
                   disabled=DELAY_DISABLED, back=back)
    if variant is BACK:
        return BACK
    return DELAY_VARIANTS[variant][0]


def suggest_tag(audio_repr, eval_label, shifting_time):
    tag = f"{audio_repr}_{eval_label}"
    if shifting_time:
        tag += f"_d{shifting_time}"
    return tag


def ask_train_axes():
    audio_repr = pick("Audio representation / model:", AUDIO_REPRS)
    cv_mode, held, eval_label = ask_eval_setting()
    # The EEG-audio delay ablation (all-0 / all-200) is a paper baseline knob;
    # CLAP always uses the all-0 (no-delay) Akama best-model setting, so we don't
    # ask for it there.
    shifting_time = 0 if audio_repr == "clap" else ask_delay()
    seed = ask_int("Optimisation seed:", default=DEFAULT_SEED)
    tag = ask_text("Run label (--training_date):",
                   default=suggest_tag(audio_repr, eval_label, shifting_time))
    return dict(audio_repr=audio_repr, cv_mode=cv_mode, cv_held_out_id=held,
                shifting_time=shifting_time, seed=seed, training_date=tag)


# Argparse defaults, for runs whose hparams.yaml predates these flags (e.g. an
# early within-subject run that never logged cv_mode is, by definition, within).
_HP_DEFAULTS = {"audio_repr": None, "cv_mode": "within",
                "cv_held_out_id": "-1", "shifting_time": "0"}


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
                         shifting_time=None):
    """training_dates under results/ with a best-checkpoint.ckpt whose training
    config matches the given axes (read from each run's hparams.yaml).

    Matching on (audio_repr, cv_mode, cv_held_out_id, shifting_time) is a
    reproducibility guard: a checkpoint is only offered for a test whose split it
    was actually trained for. A leave_subject_out=3 model is therefore not listed
    when testing within-subject or a different held-out subject — which would
    otherwise evaluate that model on data it saw during training.
    """
    want = {"audio_repr": audio_repr, "cv_mode": cv_mode,
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
                   cv_held_out_id=None, shifting_time=None, back=False):
    """Return (checkpoint_path_or_None, training_date_override_or_None), or BACK.

    The "existing" list merges results/ runs matching the test split (the
    reproducibility guard) with the authors' released checkpoints valid for that
    split, so the baseline is selectable from the launcher. Picking a run lets
    checkpoint_test.py auto-discover its checkpoint under results/<run>/.
    """
    setting = "within" if cv_mode in (None, "within") else f"{cv_mode}={cv_held_out_id}"
    desc = " ".join(x for x in (audio_repr, setting) if x)
    how = pick("Checkpoint to evaluate:", {
        "existing": f"pick a matching run / authors' checkpoint [{desc}]",
        "path": "type an explicit .ckpt path",
    }, back=back)
    if how is BACK:
        return BACK
    if how == "path":
        p = ask_path("Path to .ckpt:", back=back)
        return BACK if p is BACK else (p, None)
    runs = runs_with_checkpoint(audio_repr, cv_mode, cv_held_out_id, shifting_time)
    authors = authors_checkpoints_for(audio_repr, cv_mode, cv_held_out_id)
    if not runs and not authors:
        print(f"  no checkpoint matches [{desc}] — give an explicit path")
        p = ask_path("Path to .ckpt:", back=back)
        return BACK if p is BACK else (p, None)
    questionary = _questionary()
    choices = []
    if back:
        choices.append(questionary.Choice(title="↩ back", value=BACK))
    for r in runs:
        choices.append(questionary.Choice(title=f"run: {r}", value=("run", r)))
    for label, p in authors:
        choices.append(questionary.Choice(title=label, value=("path", p)))
    sel = questionary.select(f"Which checkpoint?  [{desc}]", choices=choices).ask()
    if sel is None:
        return BACK if back else sys.exit("aborted")
    if sel is BACK:
        return BACK
    kind, val = sel
    return (None, val) if kind == "run" else (val, None)


# --- Phase flows ---------------------------------------------------------------
def flow_train(axes=None):
    axes = axes or ask_train_axes()
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
    print(f"\n→ Testing: {axes['audio_repr']} model  |  {setting}  |  run '{axes['training_date']}'")
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


def flow_test(axes=None, *, ask_shuffle=True):
    # train+test path: the model was just trained under axes[training_date];
    # auto-discover its checkpoint, no prompts.
    if axes is not None:
        return _run_test(axes, ckpt=None, shuffle="none")

    st = {}

    def s_audio(s):
        v = pick("Audio representation / model:", AUDIO_REPRS, back=True)
        if v is BACK:
            return BACK
        s["audio_repr"] = v

    def s_eval(s):
        r = ask_eval_setting(back=True)
        if r is BACK:
            return BACK
        s["cv_mode"], s["cv_held_out_id"], s["eval_label"] = r

    def s_delay(s):
        if s["audio_repr"] == "clap":
            s["shifting_time"] = 0  # CLAP is fixed to all-0; no prompt
            return SKIP
        v = ask_delay(back=True)
        if v is BACK:
            return BACK
        s["shifting_time"] = v

    def s_seed(s):
        v = ask_int("Optimisation seed:", default=DEFAULT_SEED, back=True)
        if v is BACK:
            return BACK
        s["seed"] = v

    def s_ckpt(s):
        res = ask_checkpoint(
            suggest_tag(s["audio_repr"], s["eval_label"], s["shifting_time"]),
            audio_repr=s["audio_repr"], cv_mode=s["cv_mode"],
            cv_held_out_id=s["cv_held_out_id"], shifting_time=s["shifting_time"],
            back=True)
        if res is BACK:
            return BACK
        s["ckpt"], s["run_override"] = res

    def s_label(s):
        if s.get("run_override") is not None:
            s["training_date"] = s["run_override"]  # output into the picked run's dir
            return SKIP
        v = ask_text("Run label (--training_date, output dir):",
                     default=suggest_tag(s["audio_repr"], s["eval_label"], s["shifting_time"]),
                     back=True)
        if v is BACK:
            return BACK
        s["training_date"] = v

    def s_shuffle(s):
        if not ask_shuffle:
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

    if _wizard([s_audio, s_eval, s_delay, s_seed, s_ckpt, s_label, s_shuffle], st) is None:
        return BACK  # backed out past the first step -> phase menu

    axes = dict(audio_repr=st["audio_repr"], cv_mode=st["cv_mode"],
                cv_held_out_id=st["cv_held_out_id"], shifting_time=st["shifting_time"],
                seed=st["seed"], training_date=st["training_date"])
    return _run_test(axes, ckpt=st["ckpt"], shuffle=st["shuffle"])


def flow_train_test():
    axes = ask_train_axes()
    rc, _ = flow_train(axes)
    if rc not in (0, None):
        print("training failed; skipping test")
        return
    flow_test(axes, ask_shuffle=False)


def flow_compare():
    source = pick("Comparison source:", {
        "checkpoints": "evaluate two existing checkpoints (fast, minutes)",
        "scratch": "train raw and clap from scratch, then compare (hours, GPU)",
    })
    cv_mode, held, eval_label = ask_eval_setting()
    seed = ask_int("Optimisation seed:", default=DEFAULT_SEED)
    shifting_time = 0  # the comparison is on the headline all-0 model
    rows = []
    for repr_name in ("raw", "clap"):
        tag = f"compare_{repr_name}_{eval_label}"
        axes = dict(audio_repr=repr_name, cv_mode=cv_mode, cv_held_out_id=held,
                    shifting_time=shifting_time, seed=seed, training_date=tag)
        if source == "scratch":
            print(f"\n=== {repr_name}: train from scratch ===")
            if run_subprocess(build_command("main.py", **axes)) != 0:
                print(f"{repr_name} training failed; skipping")
                continue
            ckpt = None
        else:
            print(f"\n=== {repr_name}: pick checkpoint ===")
            ckpt, run = ask_checkpoint(tag, audio_repr=repr_name, cv_mode=cv_mode,
                                       cv_held_out_id=held, shifting_time=shifting_time)
            if run is not None:
                axes["training_date"] = run
        print(f"\n=== {repr_name}: evaluate ===")
        run_subprocess(build_command("checkpoint_test.py", checkpoint_path=ckpt,
                                     test_breakdown=1, **axes))
        rows.append((repr_name, axes["training_date"]))
    render_compare(rows, eval_label)


def flow_reproduce():
    phase = pick("Which paper tables?", {
        "within loso": "Table 1 (within) + Table 2 (LOSO), both from checkpoints",
        "within": "Table 1 only (within-subject, model-all0.ckpt)",
        "loso": "Table 2 only (LOSO, model-sub{3,7,2}.ckpt)",
    })
    script = PROJECT_ROOT / "scripts" / "reproduce_akama.sh"
    env = dict(os.environ, REPRO_PHASES=phase)
    print("\n$ REPRO_PHASES=" + shlex.quote(phase) + " bash scripts/reproduce_akama.sh\n")
    subprocess.run(["bash", str(script)], cwd=str(PROJECT_ROOT), env=env)


def flow_sanity():
    audio_repr = pick("Audio representation of the checkpoint:", AUDIO_REPRS)
    ckpt = ask_text("Checkpoint (repo-relative or absolute):",
                    default="checkpoints/model-all0.ckpt")
    tag = ask_text("Tag (training_date prefix):", default="sanity")
    script = PROJECT_ROOT / "scripts" / "test_sanity.sh"
    env = dict(os.environ, CKPT=ckpt, TAG=tag, AUDIO_REPR=audio_repr)
    print(f"\n$ CKPT={shlex.quote(ckpt)} TAG={shlex.quote(tag)} "
          f"AUDIO_REPR={audio_repr} bash scripts/test_sanity.sh\n")
    subprocess.run(["bash", str(script)], cwd=str(PROJECT_ROOT), env=env)


# --- Comparison table ----------------------------------------------------------
def render_compare(rows, eval_label):
    from rich.console import Console
    from rich.table import Table

    table = Table(title=f"Baseline (raw) <-> CLAP   [{eval_label}]")
    table.add_column("variant", style="bold")
    table.add_column("global acc (all)", justify="right")
    table.add_column("global acc (attn)", justify="right")
    table.add_column("macro (all)", justify="right")
    table.add_column("reference", justify="right")
    table.add_column("source")

    for repr_name, tag in rows:
        summary = find_summary(tag)
        if summary:
            acc = parse_summary(summary)
            macro = macro_accuracy(summary)
        else:
            acc, macro = {"all": None, "attn": None}, None
        ref = REFERENCE.get((repr_name, eval_label), {})
        table.add_row(repr_name, _fmt(acc["all"]), _fmt(acc["attn"]), _fmt(macro),
                      _fmt(ref.get("all")), ref.get("src", "--"))
    Console().print(table)
    print("(global acc all/attn from test_breakdown_summary.txt; macro = mean of "
          "the 4 per-task accuracies)")


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
    print("(use the '↩ back' menu entry to step back)\n")
    dispatch = {
        "train": flow_train,
        "test": flow_test,
        "train+test": flow_train_test,
        "compare": flow_compare,
        "reproduce-paper": flow_reproduce,
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
