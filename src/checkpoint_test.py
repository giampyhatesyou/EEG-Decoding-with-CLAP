import sys
import os

# --- Safe GPU/CPU auto-detection ---
# If torch.cuda.is_available() crashes (e.g. broken NVIDIA fabric manager on cluster),
# hide CUDA so PyTorch Lightning falls back cleanly to CPU.
try:
    import torch as _torch
    _ok = _torch.cuda.is_available()
except Exception:
    _ok = False
if not _ok:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

import resource
try:
    # Set memlock limit to hard limit (unlimited) programmatically
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard))
except Exception:
    pass

# Ensure the local 'attention' directory is at the top of sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from audiomentations import AddGaussianNoise, Gain
from datasets import get_dataset
from models import SampleCNN2DEEG, CLAPEncoder
from modules import EEGContrastiveLearning
from utils import yaml_config_hook, get_logger, file_writer, paths
from preprocessing import eeg_data_processing, experiment_data_processing
import pandas as pd
import datetime
import random
from pathlib import Path
import torch


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PredANN")

    config = yaml_config_hook("../configs/baseline.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    parser.add_argument('--mode', type=str)
    parser.add_argument('--start_position', type=int)
    parser.add_argument('--evaluation_length', type=int)
    parser.add_argument('--attention_values', type=int, nargs='+')
    parser.add_argument('--subject_id', type=int, default=None)
    parser.add_argument('--song_id', type=int, default=None)
    parser.add_argument('--key', type=str)
    parser.add_argument('--test_window_size', type=int)
    parser.add_argument('--test_stride', type=int)
    parser.add_argument('--log_dir', type=str, default="../results", help="Base directory for logs and checkpoints (relative to src/)")
    args = parser.parse_args()

    # --- Dynamic, host-aware path / worker resolution. ---
    # Same contract as main.py: YAML defaults define the vanilla repo layout;
    # known hosts redirect to shared storage; explicit CLI overrides win;
    # EEG_* env vars take absolute precedence.
    _yaml_dataset_dir_default = config['dataset_dir']
    _log_dir_default = "../results"
    if args.dataset_dir == _yaml_dataset_dir_default:
        args.dataset_dir = paths.resolve_dataset_dir(_yaml_dataset_dir_default)
    if args.log_dir == _log_dir_default:
        args.log_dir = paths.resolve_log_dir(_log_dir_default)
    args.workers = paths.resolve_workers(args.workers)

    _env_info = paths.describe()
    print(f"[paths] host={_env_info['hostname']} "
          f"host_profile_keys={_env_info['host_profile_keys']} "
          f"cpu={_env_info['cpu_count']} "
          f"cpu_effective={_env_info['cpu_effective']}")
    print(f"[paths] dataset_dir={args.dataset_dir}")
    print(f"[paths] log_dir={args.log_dir}")
    print(f"[paths] workers={args.workers}")

    pl.seed_everything(args.seed, workers=True)

    train_transform = {}
    if args.openmiir_augmentation == "gaussiannoise":
        train_transform = [
            AddGaussianNoise(min_amplitude=args.min_amplitude,
                             max_amplitude=args.max_amplitude, p=0.5),
        ]
        print("augematation is gaussiannoise")

    elif args.openmiir_augmentation == "gain":
        train_transform = [
            Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.5)
        ]
        print("augematation is gain")

    elif args.openmiir_augmentation == "gaussiannoise+gain":
        train_transform = [
            AddGaussianNoise(min_amplitude=args.min_amplitude,
                             max_amplitude=args.max_amplitude, p=0.5),
            Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.5)
        ]
        print("augematation is gaussiannoise+gain")

    else:
        print("no augmentation")

    train_log = pd.DataFrame(
        columns=["Loss/train", "Accuracy/train_eeg", "Accuracy/train_audio"])
    valid_log = pd.DataFrame(
        columns=["Loss/valid", "Accuracy/valid_eeg", "Accuracy/valid_audio"])

    print(f"[cv] mode={args.cv_mode} held_out_id={args.cv_held_out_id}")
    train_dataset = get_dataset(
        args.dataset, args.dataset_dir, subset="train", download=False,
        cv_mode=args.cv_mode, cv_held_out_id=args.cv_held_out_id)
    train_dataset.set_sliding_window_parameters(args.window_size, args.stride)
    train_dataset.set_eeg_normalization(
        args.eeg_normalization, args.clamp_value)
    train_dataset.set_other_parameters( args.eeg_length, args.audio_clip_length, args.shifting_time, args.start_position)
    random.seed(42)
    train_random_numbers = [random.randint(
        0, args.eeg_sample_rate * 30 - args.eeg_length - 1) for _ in range(1200)]
    train_dataset.set_random_numbers(train_random_numbers)

    if args.openmiir_augmentation != "no_augmentation":
        train_dataset.set_transform(train_transform)

    valid_dataset = get_dataset(
        args.dataset, args.dataset_dir, subset="valid", download=False,
        cv_mode=args.cv_mode, cv_held_out_id=args.cv_held_out_id)
    valid_dataset.set_sliding_window_parameters(args.window_size, args.stride)
    valid_dataset.set_eeg_normalization(
        args.eeg_normalization, args.clamp_value)
    valid_dataset.set_other_parameters( args.eeg_length, args.audio_clip_length, args.shifting_time, args.start_position)
    random.seed(42)
    valid_random_numbers = [random.randint(
        0, args.window_size - args.eeg_length - 1) for _ in range(1200)]
    valid_dataset.set_random_numbers(valid_random_numbers)

    test_dataset = get_dataset(
        args.test_dataset, args.dataset_dir, subset="test", download=False,
        cv_mode=args.cv_mode, cv_held_out_id=args.cv_held_out_id)
    test_dataset.set_test_data_length(args.test_data_length)
    test_dataset.set_sliding_window_parameters(args.test_window_size, args.test_stride)
    test_dataset.set_eeg_normalization(
        args.eeg_normalization, args.clamp_value)
    test_dataset.set_other_parameters(
        args.eeg_length, args.audio_clip_length, args.shifting_time, args.start_position)
    random.seed(42)
    test_random_numbers = [random.randint(
        0, args.window_size - args.eeg_length - 1) for _ in range(1200)]
    test_dataset.set_random_numbers(test_random_numbers)

    # --- test loader shuffling policy ---
    # Normal evaluation (shuffle_test_mode="none") keeps shuffle=False so the
    # printed numbers are bit-for-bit reproducible against the legacy test.sh
    # output. When running a negative-control sweep (labels/audio_pair), we
    # MUST shuffle: the dataset emits the 13 sliding windows of a trial
    # consecutively, so with batch_size=8 and shuffle=False most batches
    # would carry a single task and the intra-batch permutation in
    # test_step would degenerate into a no-op. Forcing shuffle=True here
    # makes the negative control actually meaningful.
    _shuffle_mode = getattr(args, "shuffle_test_mode", "none")
    _test_shuffle = _shuffle_mode != "none"
    if _test_shuffle:
        print(f"[test_loader] shuffle=True (shuffle_test_mode={_shuffle_mode!r})")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=_test_shuffle,
    )

    print(f"Size of train dataset: {len(train_dataset)}")
    print(f"Size of valid dataset: {len(valid_dataset)}")
    print(f"Size of test dataset: {len(test_dataset)}")

    if args.dataset == "preprocessing_eegmusic":
        encoder_eeg = SampleCNN2DEEG(
            out_dim=train_dataset.labels(),
            kernal_size=3,
        )

        if args.audio_repr == "clap":
            # Must match the audio_repr used at training time, otherwise the
            # checkpoint's audio-side weights won't load (different topology).
            _shared_audio = CLAPEncoder(
                out_dim=100,
                hidden_dim=args.clap_proj_hidden_dim,
                pretrained=args.clap_pretrained,
            )
            encoder_vocal = _shared_audio
            encoder_drum = _shared_audio
            encoder_bass = _shared_audio
            encoder_others = _shared_audio
            print(
                f"[audio] CLAPEncoder (shared) hidden={args.clap_proj_hidden_dim} "
                f"pretrained={args.clap_pretrained!r}"
            )
        else:
            encoder_vocal = SampleCNN2DEEG(
                out_dim=train_dataset.labels(),
                kernal_size=3,
            )
            encoder_drum = SampleCNN2DEEG(
                out_dim=train_dataset.labels(),
                kernal_size=3,
            )
            encoder_bass = SampleCNN2DEEG(
                out_dim=train_dataset.labels(),
                kernal_size=3,
            )
            encoder_others = SampleCNN2DEEG(
                out_dim=train_dataset.labels(),
                kernal_size=3,
            )
            print("[audio] SampleCNN2DEEG (4 independent encoders, Akama baseline)")

    print('EEG Contrastive learning')
    module = EEGContrastiveLearning(
        valid_dataset, args, encoder_eeg, encoder_vocal, encoder_drum, encoder_bass, encoder_others,key=args.key)

    logger = TensorBoardLogger(
        "{}/{}".format(args.log_dir, args.training_date), name="nmed-CL-{}".format(args.dataset))

    early_stop_callback = EarlyStopping(
        monitor="Valid/loss", patience=10
    )

    # --- Quiet progress when stdout is not a TTY (see main.py for rationale). ---
    _force = os.environ.get("EEG_FORCE_PROGRESS")
    if _force == "1":
        _is_tty = True
    elif _force == "0":
        _is_tty = False
    else:
        _is_tty = sys.stdout.isatty()
    _progress_bar_cb = TQDMProgressBar(refresh_rate=1 if _is_tty else 0)
    print(f"[paths] tty={_is_tty} (progress bar {'on' if _is_tty else 'off'})")

    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        sync_batchnorm=True,
        max_epochs=args.max_epochs,
        deterministic=True,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        accelerator=args.accelerator,
        devices=args.devices,
        resume_from_checkpoint=args.checkpoint_path if args.checkpoint_path else None,
        accumulate_grad_batches=6,
        callbacks=[_progress_bar_cb],
    )
    # Auto-discover latest checkpoint if not explicitly provided via --checkpoint_path.
    # Scans {log_dir}/{training_date}/nmed-CL-{dataset}/version_*/checkpoints/ and picks the most recent version.
    # To use a specific checkpoint, pass --checkpoint_path <path> from the command line.
    if not args.checkpoint_path:
        import glob
        pattern = f"{args.log_dir}/{args.training_date}/nmed-CL-{args.dataset}/version_*/checkpoints/best-checkpoint.ckpt"
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No checkpoint found matching: {pattern}")
        args.checkpoint_path = matches[-1]
        print(f"Auto-discovered checkpoint: {args.checkpoint_path}")

    print('[[[ START ]]]', datetime.datetime.now())
    checkpoint_path = args.checkpoint_path
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    # strict=False is required: the authors' checkpoints carry an extra
    # `projector1` head (legacy multi-task setup) that the current model omits.
    # Printing the key diff means a genuine mismatch -- e.g. loading a raw-encoder
    # checkpoint into the CLAP topology -- surfaces here instead of silently
    # loading partial weights and reporting a meaningless accuracy.
    load_result = module.load_state_dict(checkpoint['state_dict'], strict=False)
    if load_result.missing_keys:
        print(f"[load] {len(load_result.missing_keys)} missing key(s) not in checkpoint; first few: {load_result.missing_keys[:8]}")
    if load_result.unexpected_keys:
        print(f"[load] {len(load_result.unexpected_keys)} unexpected key(s) in checkpoint (ignored); first few: {load_result.unexpected_keys[:8]}")
    if not load_result.missing_keys and not load_result.unexpected_keys:
        print("[load] checkpoint state_dict matched the model exactly")
    trainer.test(module,dataloaders=test_loader)
    print('[[[ FINISH ]]]', datetime.datetime.now())


