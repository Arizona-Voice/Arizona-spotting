# -*- coding: utf-8 -*-

import os
import torch
import random
import librosa
import numpy as np
import soundfile as sf

from typing import Any, Dict, Union, List
from fairseq.data.audio.raw_audio_dataset import *

class Wav2KWSDataset(RawAudioDataset):
    def __init__(
        self,
        mode: str='train',
        root: str='',
        sample_rate: int=16000,
        loudest_section: bool=True,
        silence_percentage: float=0.1,
        silence_token: str='_silence_',
        noise_level: float=0.7,
        noise_prob: Union[float, bool]=0.5,
        shift_prob: Union[float, bool]=0.5,
        mask_prob: Union[float, bool]=0.5,
        mask_len: float=0.1,
        tf_audio_processor: Any=None,
        classes: List=None
    ):
        super(Wav2KWSDataset, self).__init__(sample_rate, pad=False)

        self.mode = mode
        self.root = root
        self.mode_root = os.path.join(root, self.mode)
        self.sample_rate = sample_rate        
        self.ap = tf_audio_processor
        self.loudest_section = loudest_section
        self.silence_perecentage = silence_percentage
        self.silence_token = silence_token
        self.data = list()
        self.classes = []
        if classes:
            self.classes = classes
        else:
            for _, dir, _ in os.walk(self.mode_root):
                classes = dir
                break
            self.classes = classes

        self.idx2label = {}
        self.label2idx = {}
        for i, cl in enumerate(self.classes):
            self.idx2label[i] = cl
            self.label2idx[cl] = i

        self.prep_dataset()

        if self.mode.lower() == 'train':
            self.noise_data = list()
            self.noise_prop = noise_prob
            self.noise_level = noise_level
            self.shift_prob = shift_prob
            self.mask_prob = mask_prob
            self.mask_len = mask_len
            if self.noise_prop:
                self.prep_noise_dataset()
    
    def prep_dataset(self):
        if self.ap is None:
            self.id = 0
            for c in self.classes:
                for root, dir, files in os.walk(os.path.join(self.mode_root, c)):
                    for file in files:
                        f_path, cmd = os.path.join(root, file), c
                        self.data.append((f_path, cmd, self.id))
                        self.id += 1
        else:
            self.id = 0
            tf_data = self.ap.data_index[self.mode]
            for td in tf_data:
                f_path, cmd = td['file'], td['label']
                if cmd == self.silence_token:
                    self.data.append((f_path, 'silence', self.id))
                elif cmd in self.classes:
                    self.data.append((f_path, cmd, self.id))
                elif not cmd in self.classes:
                    self.data.append((f_path, 'unknown', self.id))
                self.id += 1

    def prep_noise_dataset(self):
        noise_path = os.path.join(self.root, '_background_noise_')
        if not os.path.exists(noise_path):
            print(f"Path to folder background noise not exists. "
                  f"The name of this folder must be `_background_noise_`. ")
        else:
            samples = []
            for root, dir, files in os.walk(noise_path):
                for file in files:
                    if file.endswith('wav'):
                        f_path = os.path.join(root, file)
                        wav, _ = sf.read(f_path)
                        samples.append(wav)

            samples = np.hstack(samples)
            c = int(self.sample_rate)
            r = len(samples) // c
            self.noise_data = samples[: r*c].reshape(-1, c)

    def __getitem__(self, idx):
        f_path, cmd, id = self.data[idx]

        if f_path and os.path.exists(f_path):
            wav, curr_sample_rate = sf.read(f_path)
            if curr_sample_rate != self.sample_rate:
                wav, curr_sample_rate = librosa.resample(wav, curr_sample_rate, self.sample_rate), self.sample_rate

            if len(wav.shape) == 2:
                wav = librosa.to_mono(wav.transpose(1, 0))
            
            if self.loudest_section:
                wav = self.extract_loudest_section(wav)

            wav_len = len(wav)
            if wav_len < self.sample_rate:
                pad_size = self.sample_rate - wav_len
                wav = np.pad(wav, (round(pad_size / 2) + 1, round(pad_size / 2) + 1), 'constant', constant_values=0)
        else:
            wav, curr_sample_rate = np.zeros(self.sample_rate, dtype=np.float32), self.sample_rate

        wav_len = len(wav)
        mid = int(len(wav) / 2)
        cut_off = int(self.sample_rate / 2)
        wav = wav[mid - cut_off: mid + cut_off]

        if self.mode == 'train':
            if self.shift_prob and random.random() < self.shift_prob:
                percentage = random.uniform(-self.shift_prob, self.shift_prob)
                d = int(self.sample_rate * percentage)
                wav = np.roll(wav, d)
                if d > 0:
                    wav[: d] = 0
                else:
                    wav[d: ] = 0
            
            if self.mask_prob and random.random() < self.mask_prob:
                t = int(self.mask_len * self.sample_rate)
                t0 = random.randint(0, self.sample_rate - t)
                wav[t0: t + t0] = 0

            if self.noise_prop and random.random() < self.noise_prop:
                noise = random.choice(self.noise_data)
                if cmd == 'silence':
                    percentage = random.uniform(0, 1)
                    wav = wav * (1 - percentage) + noise * percentage
                else:
                    percentage = random.uniform(0, self.noise_level)
                    wav = wav * (1 - percentage) + noise * percentage

        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats, curr_sample_rate)
        y = self.label2idx.get(cmd)

        return {'id': id, 'target': y, 'source': feats}

    def extract_loudest_section(self, wav, win_len=30):
        wav_len = len(wav)
        temp = abs(wav)
        st, et = 0, 0
        max_dec = 0

        for ws in range(0, wav_len, win_len):
            cur_dec = temp[ws: ws + 16000].sum()
            if cur_dec >= max_dec:
                max_dec = cur_dec
                st, et = ws, ws + 16000
                
            if ws + 16000 > wav_len:
                break

        return wav[st: et]

    def __len__(self):
        return len(self.data)

    def make_weights_for_balanced_classes(self):
        nclasses = len(self.classes)
        count = np.zeros(nclasses)
        for item in self.data:
            count[self.label2idx.get(item[1])] += 1

        N = float(sum(count))
        weight_per_class = N / count
        weight = np.zeros(len(self.data))
        for idx, item in enumerate(self.data):
            weight[idx] = weight_per_class[self.label2idx.get(item[1])]

        return weight

    def _collate_fn(self, samples):
        sub_samples = [s for s in samples if s['target'] is not None]
        if len(sub_samples) == 0:
            return {}

        batch = self.collater(samples)
        batch['target'] = torch.LongTensor([s['target'] for s in sub_samples])

        return batch
        