import os
import pickle
import numpy as np
import torch
import torchaudio
import pandas as pd
import random
import re
from torch import Tensor
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler
from audiomentations import Compose
from torch.utils.data import Dataset
from typing import Tuple, Optional
from utils import get_logger
from pathlib import Path


def get_window(df,eeg_length,window_size,stride,start_position):
    df.insert(9, "window", np.zeros(len(df.index)), True)
    newdf = pd.DataFrame(np.repeat(df.values,int((eeg_length -start_position - window_size)/stride + 1), axis=0))
    newdf.columns = df.columns

    for i in range(len(newdf.index)):
        newdf.at[i, 'window']=i%int((eeg_length - window_size)/stride + 1)

    return newdf

def get_test_window(df,eeg_length,window_size,stride):
    df.insert(9, "window", np.zeros(len(df.index)), True)
    newdf = pd.DataFrame(np.repeat(df.values,int((eeg_length - window_size)/stride + 1), axis=0))
    newdf.columns = df.columns

    for i in range(len(newdf.index)):
        newdf.at[i, 'window']=i%int((eeg_length - window_size)/stride + 1)

    return newdf

class Preprocessing_EEGMusic_dataset(Dataset):
    logger = get_logger("dataloader_debug")

    def __init__(
        self,
        root: str,
        base_dir: str = None,
        download: bool = False,
        subset: Optional[str] = None,
        cv_mode: str = "within",
        cv_held_out_id: int = -1,
    ):

        self.root = root
        self.base_dir = base_dir if base_dir is not None else root
        self.subset = subset
        # --- Cross-validation routing ------------------------------------------
        # cv_mode controls how _get_file_list assembles each split:
        #   "within"             -> reproduce the original within-subject split
        #                            written into the folder tree (default).
        #   "leave_song_out"     -> train/valid: original split BUT drop the
        #                            held-out song; test: every (subject) row
        #                            for that song, regardless of original split.
        #   "leave_subject_out"  -> train/valid: original split BUT drop the
        #                            held-out subject; test: every (song) row
        #                            for that subject, regardless of original
        #                            split.
        # cv_held_out_id is the song or subject integer id being held out for
        # the current fold. Ignored when cv_mode == "within".
        self.cv_mode = cv_mode
        self.cv_held_out_id = int(cv_held_out_id)
        self.eeg_normalization = None
        self.transform = None
        self.eeg_length = 256*15
        self.eeg_sample_rate = 256
        self.eeg_clip_length = 768
        self.test_data_length = 256
        self.whole_audio_length = 15
        self.audio_length = 0
        self.audio_sample_rate = 44100
        self.train_test_splitting = 'random_split_15s'
        self.random_numbers = []
        self.start_position = 0
        self.evaluation_length = 768
        self.audio_clip = 3

        assert subset is None or subset in ["train", "valid", "test", "CV", "SW_valid", "SW_train"], (
            "When `subset` not None, it must take a value from "
            + "{'train', 'valid', 'test', 'CV'}."
        )
        self.window_size = None
        self.stride = None
        self.fold = None
        self.mode = None
        self.start = []
        self.start_value = 0

        self.fs = 125.0
        self.lowcut = 1.0
        self.highcut = 50.0
        self.df_chunk_column = 0
        self.df_window_column = 0

        self.df = self._get_file_list(
            self.base_dir, self.subset).reset_index(drop=True)

    def file_path(self, n: int) -> str:
        pass

    def set_transform(self, transform):
        self.transform = Compose(transform)

    def set_other_parameters(self, eeg_clip_length, audio_clip_length, shifting_time, start_position):
        self.eeg_clip_length = eeg_clip_length
        self.audio_clip = audio_clip_length
        self.audio_length = audio_clip_length * self.audio_sample_rate
        self.shifting_time = shifting_time
        self.start_position = start_position
        self.df = self._get_file_list(
            self.base_dir, self.subset).reset_index(drop=True)


        start = 0
        new_valid_df=get_window(self.df,self.eeg_length,self.window_size,self.stride,start)
        new_test_df=get_test_window(self.df,self.eeg_length,self.window_size,self.stride)
        self.df_subset = new_valid_df
        self.df_subset_test=new_test_df

    def update_random_window(self):
        start = random.randint(self.start_position, self.stride - 1)
        self.df_subset = get_window(
            self.df, self.eeg_length, self.window_size, self.stride, start
        )
       
    def set_random_numbers(self, random_numbers):
        self.random_numbers = random_numbers

    def set_sliding_window_parameters(self, window_size, stride):
        self.window_size = window_size
        self.stride = stride

    def labels(self):
        num_label = len(self.df_subset.task.unique())
        return num_label

    def set_eeg_normalization(self, eeg_normalization, clamp_value=None):
        self.eeg_normalization = eeg_normalization
        self.clamp_value = clamp_value

    def getitem(self, n, isClip=True):
        
        eeg_path = self.df_subset.iloc[n, 4]
        window = self.df_subset.iloc[n, 9]
        with open(eeg_path, 'rb') as f:
            eeg = pickle.load(f)

        
        audio_path_list = [
            {"key": x-5, "value": self.df_subset.iloc[n, x]} for x in range(5, 9)]

        audio_list = []
        for audio_path in audio_path_list:
            with open(audio_path["value"], 'rb') as f:
                audio, audio_sample_rate = torchaudio.load(f)
                audio = audio[:, int(window*self.stride*44100/256)-int((self.audio_clip-3)/2*44100):int((window*self.stride+self.window_size)*44100/256)+int((self.audio_clip+3)/2*44100)] 
                audio_list.append({"key": audio_path["key"], "value": audio})

        task = int(self.df_subset.iloc[n, 2])
        attention_score = int(self.df_subset.iloc[n, 3])
        subject = (int(self.df_subset.iloc[n, 0]))
        song = int(self.df_subset.iloc[n, 1])
        
        eeg = eeg[:, int(int((window)*self.stride+(0.256*self.shifting_time))):int(int((window)*self.stride+self.window_size)+(0.256*self.shifting_time))]
       
        if isClip:
            all_eeg_length = eeg.size(1)
     
            end = int(all_eeg_length-self.eeg_clip_length-(self.audio_clip-3)/2*self.eeg_sample_rate-1)
            
            eeg_start = random.randint(int((self.audio_clip-3)/2*self.eeg_sample_rate), end)
            eeg = eeg[:, eeg_start: eeg_start + self.eeg_clip_length]
            

            for k, audio in enumerate(audio_list):
                audio_start = int(eeg_start/self.eeg_sample_rate * self.audio_sample_rate)+int(
                    (self.audio_clip-3)/2*self.audio_sample_rate)                
                audio_list[k]["value"] = audio["value"][:, int(audio_start): int(
                    audio_start + self.audio_clip * self.audio_sample_rate)]
                self.logger.debug(f"audio, {audio['value'].shape}")

        if self.eeg_normalization == "channel_mean":
            eeg = self.normalize_EEG(eeg)
        elif self.eeg_normalization == "all_mean":
            eeg = self.normalize_EEG_2(eeg)
        elif self.eeg_normalization == "constant_multiple":
            eeg = self.normalize_EEG_3(eeg)
        elif self.eeg_normalization == "MetaAI":
            eeg = self.normalize_EEG_4(eeg, self.clamp_value)

        if self.transform != None:
            eeg = eeg.to('cpu').detach().numpy().copy()
            eeg = self.transform(eeg, sample_rate=self.eeg_sample_rate)
            eeg = torch.from_numpy(eeg.astype(np.float32)).clone()

            for k, audio in enumerate(audio_list):
                audio_list[k]["value"] = audio["value"].to(
                    'cpu').detach().numpy().copy()
                audio_list[k]["value"] = self.transform(
                    audio, sample_rate=self.eeg_sample_rate)
                audio_list[k]["value"] = torch.from_numpy(
                    audio.astype(np.float32)).clone()

        padding = (int((3**7-self.eeg_clip_length)/2),
                   int((3**7-self.eeg_clip_length)/2))
        eeg = F.pad(eeg, padding)
        if eeg.shape[1] != 3**7:
            padding = (0, 3**7-eeg.shape[1])
            eeg = F.pad(eeg, padding)

        padding2 = (int((3**11-self.audio_length)/2), int((3**11-self.audio_length)/2))
        for k, audio in enumerate(audio_list):
            audio_list[k]["value"] = F.pad(audio["value"], padding2)
            if audio["value"].shape[1] != 3**11:
                padding2 = (0, 3**11-audio["value"].shape[1])
                audio_list[k]["value"] = F.pad(audio["value"], padding2)

        audio0 = audio_list[0]["value"]
        audio1 = audio_list[1]["value"]
        audio2 = audio_list[2]["value"]
        audio3 = audio_list[3]["value"]

        return eeg, audio0, audio1, audio2, audio3, task, attention_score, subject, song

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:

        eeg, audio0, audio1, audio2, audio3, task, attention_score, subject, song = self.getitem(
            n)
        return eeg, audio0, audio1, audio2, audio3, task, attention_score, subject, song

    def __len__(self) -> int:
        return len(self.df_subset)

    def normalize_EEG(self, eeg):
        eeg_mean = torch.mean(eeg, 1)
        eeg = eeg-eeg_mean.unsqueeze(1)
        max_eeg = torch.max(abs(eeg), 1)
        eeg = eeg/max_eeg.values.unsqueeze(1)
        return eeg

    def normalize_EEG_2(self, eeg):
        eeg_mean = torch.mean(eeg)*torch.ones(eeg.shape[0])
        eeg = eeg-eeg_mean.unsqueeze(1)
        max_eeg = torch.max(abs(eeg), 1)
        eeg = eeg/max_eeg.values.unsqueeze(1)
        return eeg

    def normalize_EEG_3(self, eeg):
        eeg = 100*eeg
        return eeg

    def normalize_EEG_4(self, eeg, clamp_value):
        for idx, ch_eeg in enumerate(eeg):
            transformer = RobustScaler().fit(ch_eeg.view(-1, 1))
            ch_eeg = transformer.transform(ch_eeg.view(-1, 1))
            ch_eeg = torch.from_numpy(ch_eeg.astype(np.float32)).clone()
            eeg[idx] = ch_eeg.view(1, -1)

        eeg = torch.clamp(eeg, min=int(-1*clamp_value), max=int(clamp_value))
        return eeg

    def _get_file_list(self, root, subset):
        BASE = os.path.join(root, "eeg_within_sub")
        
        if not os.path.exists(BASE):
            raise RuntimeError('BASE folder is not found')

        EEG_path_list = [f for f in Path(BASE).rglob('*.pkl') if f.is_file()]
        EEG_path_list = sorted(EEG_path_list)

        BASE = os.path.join(root, "audio")
        if not os.path.exists(BASE):
            raise RuntimeError('BASE folder is not found')

        Audio_path_list = [{'task': int(f.parts[-2]), 'name': self._get_song_id(
            f.name), 'path': f} for f in Path(BASE).rglob('*.wav') if f.is_file()]

        df = pd.DataFrame(columns=['subject', 'song', 'task', 'attention_score',
                                   'eeg_path', 'audio_path0', 'audio_path1', 'audio_path2', 'audio_path3'])

        for idx, r_path in enumerate(EEG_path_list):
            r_part = r_path.parts

            r_subset = r_part[-5]
            r_subject = r_part[-6]
            r_song = self._get_song_id(r_part[-4])
            r_task = int(r_part[-3])
            attention_score = int(r_part[-2])

            # --- CV gate ---
            # See the cv_mode docstring in __init__ for the full semantics.
            # All gates short-circuit by `continue` so they are easy to reason
            # about; the original within-subject behaviour is preserved
            # bit-for-bit when cv_mode == "within".
            if self.cv_mode == "within":
                if subset != r_subset:
                    continue
            elif self.cv_mode == "leave_song_out":
                if subset in ("train", "valid"):
                    if subset != r_subset:
                        continue
                    if int(r_song) == self.cv_held_out_id:
                        continue
                elif subset == "test":
                    if int(r_song) != self.cv_held_out_id:
                        continue
                else:
                    continue
            elif self.cv_mode == "leave_subject_out":
                if subset in ("train", "valid"):
                    if subset != r_subset:
                        continue
                    if int(r_subject) == self.cv_held_out_id:
                        continue
                elif subset == "test":
                    if int(r_subject) != self.cv_held_out_id:
                        continue
                else:
                    continue
            else:
                raise ValueError(
                    f"Unknown cv_mode={self.cv_mode!r}; expected one of "
                    "{'within','leave_song_out','leave_subject_out'}."
                )

            c_audio_list = list(filter(lambda x: self._get_song_id(
                x["name"]) == self._get_song_id(r_song), Audio_path_list))
            c_path0 = list(filter(lambda x: x["task"] == 0, c_audio_list))[
                0]["path"]
            c_path1 = list(filter(lambda x: x["task"] == 1, c_audio_list))[
                0]["path"]
            c_path2 = list(filter(lambda x: x["task"] == 2, c_audio_list))[
                0]["path"]
            c_path3 = list(filter(lambda x: x["task"] == 3, c_audio_list))[
                0]["path"]

            df.loc[idx] = [r_subject, r_song, r_task, attention_score,
                           r_path, c_path0, c_path1, c_path2, c_path3]

        return df

    def _get_song_id(self, name):
        return re.sub(r'.wav$', '', name)


class Preprocessing_EEGMusic_Test_dataset(Preprocessing_EEGMusic_dataset):
    logger = get_logger("test_dataloader_debug")

    def set_test_data_length(self, test_data_length):
        self.test_data_length = test_data_length

    def getitem(self, n, isClip=True):
        p = self.start_position
        eeg_path = self.df_subset_test.iloc[n, 4]
        window = self.df_subset_test.iloc[n, 9]
        with open(eeg_path, 'rb') as f:
            eeg = pickle.load(f)
       
        audio_path_list = [
            {"key": x-5, "value": self.df_subset_test.iloc[n, x]} for x in range(5, 9)]

        audio_list = []
        for audio_path in audio_path_list:
            with open(audio_path["value"], 'rb') as f:
                audio, audio_sample_rate = torchaudio.load(f)
                audio = audio[:, int(window*self.stride*44100/256)-int((self.audio_clip-3)/2*44100):int((window*self.stride+self.window_size)*44100/256)+int((self.audio_clip+3)/2*44100)] 
                audio_list.append({"key": audio_path["key"], "value": audio})

        task = int(self.df_subset_test.iloc[n, 2])
        attention_score = int(self.df_subset_test.iloc[n, 3])
        subject = (int(self.df_subset_test.iloc[n, 0]))
        song = int(self.df_subset_test.iloc[n, 1])

        eeg = eeg[:, int(int((window)*self.stride)+(0.256*self.shifting_time)):int(int((window)*self.stride+self.window_size+(0.256*self.shifting_time)))]
    
        if self.eeg_normalization == "channel_mean":
            eeg = self.normalize_EEG(eeg)
        elif self.eeg_normalization == "all_mean":
            eeg = self.normalize_EEG_2(eeg)
        elif self.eeg_normalization == "constant_multiple":
            eeg = self.normalize_EEG_3(eeg)
        elif self.eeg_normalization == "MetaAI":
            eeg = self.normalize_EEG_4(eeg, self.clamp_value)

        if self.transform != None:
            eeg = eeg.to('cpu').detach().numpy().copy()
            eeg = self.transform(eeg, sample_rate=self.eeg_sample_rate)
            eeg = torch.from_numpy(eeg.astype(np.float32)).clone()

            for k, audio in enumerate(audio_list):
                audio_list[k]["value"] = audio["value"].to(
                    'cpu').detach().numpy().copy()
                audio_list[k]["value"] = self.transform(
                    audio, sample_rate=self.eeg_sample_rate)
                audio_list[k]["value"] = torch.from_numpy(
                    audio.astype(np.float32)).clone()

        padding = (int((3**7-self.eeg_clip_length)/2),
                   int((3**7-self.eeg_clip_length)/2))
        eeg = F.pad(eeg, padding)
        if eeg.shape[1] != 3**7:
            padding = (0, 3**7-eeg.shape[1])
            eeg = F.pad(eeg, padding)
        padding2 = (int((3**11-self.audio_length)/2), int((3**11-self.audio_length)/2))
        for k, audio in enumerate(audio_list):
            audio_list[k]["value"] = F.pad(audio["value"], padding2)
            if audio["value"].shape[1] != 3**11:
                padding2 = (0, 3**11-audio["value"].shape[1])
                audio_list[k]["value"] = F.pad(audio["value"], padding2)

        audio0 = audio_list[0]["value"]
        audio1 = audio_list[1]["value"]
        audio2 = audio_list[2]["value"]
        audio3 = audio_list[3]["value"]

        return eeg, audio0, audio1, audio2, audio3, task, attention_score, subject, song

    def __len__(self) -> int:
        return len(self.df_subset_test)

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int, int, int, str]:
        eeg, audio0, audio1, audio2, audio3, task, attention_score, subject, song_id = self.getitem(
            n)
        return eeg, audio0, audio1, audio2, audio3, task, attention_score, subject, song_id

    def _normalize(self, eeg_data, normalize_type, clamp_value=None):
        if normalize_type == "channel_mean":
            eeg_data = self.normalize_EEG(eeg_data)
        elif normalize_type == "all_mean":
            eeg_data = self.normalize_EEG_2(eeg_data)
        elif normalize_type == "constant_multiple":
            eeg_data = self.normalize_EEG_3(eeg_data)
        elif normalize_type == "MetaAI":
            eeg_data = self.normalize_EEG_4(
                eeg_data, clamp_value)

        return eeg_data

