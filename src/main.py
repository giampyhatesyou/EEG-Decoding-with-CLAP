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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from audiomentations import AddGaussianNoise, Gain
from datasets import get_dataset
from models import SampleCNN2DEEG, CLAPEncoder
from modules import EEGContrastiveLearning
from utils import yaml_config_hook, get_logger, file_writer, paths
from preprocessing import eeg_data_processing
import pandas as pd
import datetime
import random
from pathlib import Path
import torch

class RandomWindowUpdateCallback(pl.Callback):
    def __init__(self, dataset):
        self.dataset = dataset

    def on_train_epoch_start(self, trainer, pl_module):
        self.dataset.update_random_window()


class _EpochSummaryCallback(pl.Callback):
    """One concise line per epoch, designed for non-TTY logs (cluster jobs,
    `tee`'d files, etc.) where the per-batch tqdm progress bar would otherwise
    spam thousands of lines. Pure observability: no effect on training,
    gradients, or metrics — it only reads ``trainer.callback_metrics`` that
    Lightning already populates.
    """

    @staticmethod
    def _fmt(v):
        try:
            return f"{float(v):.4f}"
        except Exception:
            return "n/a"

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        train_loss = m.get('Loss/train_step',
                           m.get('Loss/train',
                                 m.get('train_loss')))
        val_loss = m.get('Valid/loss', m.get('val_loss'))
        print(
            f"[epoch {trainer.current_epoch:>4d}] "
            f"train_loss={self._fmt(train_loss)} "
            f"val_loss={self._fmt(val_loss)}",
            flush=True,
        )

def preprocess():
    input = config['dataset_dir']
    output_base = f"{config['dataset_dir']}/eeg/"
    preprocess_logger = get_logger('preprocess')

    input_path = Path(f"{input}/eeg/")
    eeg_input_list = list(input_path.glob("*.csv"))
    if not eeg_input_list:
        preprocess_logger.error("Input eeg file does not exist.")
        return

    subject_num = 0
    subject_list = []
    for eeg_input in eeg_input_list:
        subject_list.append(
            {"subject_id": subject_num, "subject_name": eeg_input.stem})
        experiment_result = Path(f"{input}/experiment/{eeg_input.stem}.json")
        if not experiment_result.exists():
            preprocess_logger.error("Experiment result file does not exist.")
            return

        eeg_data_processing(eeg_input, experiment_result,
                            subject_num, output_base)
        subject_num += 1

    return subject_list

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
    parser.add_argument('--log_dir', type=str, default="../results", help="Base directory for saving logs and checkpoints (relative to src/)")
    args = parser.parse_args()

    # --- Dynamic, host-aware path / worker resolution. ---
    # YAML defaults define the "vanilla" repo layout (so other machines clone &
    # run as-is). On known hosts (e.g. meg-server-3) we silently redirect heavy
    # I/O paths to shared storage. An explicit CLI override always wins, and
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

    if args.mode == "preprocess":
        result = preprocess()
        file_writer(f"{config['dataset_dir']}/preprocess_result.txt", result)
        exit()
        
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=True,
    )

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

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=False,
    )
    
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=False,
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
            # Variant A: one frozen LAION-CLAP backbone + a single trainable
            # projection head, shared across all 4 stems. See clap_encoder.py
            # for the design rationale. PyTorch deduplicates parameters when
            # the same nn.Module is referenced from multiple attributes, so
            # the optimizer sees the projection head's weights exactly once.
            _shared_audio = CLAPEncoder(
                out_dim=100,  # matches SampleCNN2DEEG.projector2 output width
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

    log_dir_base = getattr(args, 'log_dir', '../results')
    logger = TensorBoardLogger(
        f"{log_dir_base}/{args.training_date}", name="nmed-CL-{}".format(args.dataset))

    early_stop_callback = EarlyStopping(
        monitor="Valid/loss",    
        mode="min",           
        patience=10,          
        verbose=True,
        strict=False
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="Valid/loss",
        mode="min",
        save_top_k=1,         
        filename="best-checkpoint"
    )

    callback = RandomWindowUpdateCallback(train_dataset)

    # --- Quiet progress when stdout is not a TTY. ---
    # On interactive shells we keep the normal tqdm progress bar (refresh every
    # batch). When stdout is redirected (cluster jobs, `tee`'d logs, `nohup`),
    # tqdm degrades into one line *per* batch and floods the log; we disable
    # the per-batch bar (refresh_rate=0) and emit a single epoch summary line
    # instead. Override via EEG_FORCE_PROGRESS=1 (always on) or
    # EEG_FORCE_PROGRESS=0 (always off).
    _force = os.environ.get("EEG_FORCE_PROGRESS")
    if _force == "1":
        _is_tty = True
    elif _force == "0":
        _is_tty = False
    else:
        _is_tty = sys.stdout.isatty()
    _progress_bar_cb = TQDMProgressBar(refresh_rate=1 if _is_tty else 0)
    _callbacks = [early_stop_callback, checkpoint_callback, _progress_bar_cb]
    if not _is_tty:
        _callbacks.append(_EpochSummaryCallback())
    print(f"[paths] tty={_is_tty} (progress bar {'on' if _is_tty else 'off, epoch summaries instead'})")

    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        sync_batchnorm=True,
        max_epochs=args.max_epochs,
        min_epochs=50,
        deterministic=True,
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        accelerator=args.accelerator,
        resume_from_checkpoint=args.resume_checkpoint_path if args.resume_checkpoint_path else None,
        accumulate_grad_batches=6,
        callbacks=_callbacks,
    )
    print('[[[ START ]]]', datetime.datetime.now())
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    print('[[[ FINISH ]]]', datetime.datetime.now())