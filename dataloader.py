from os import path
from torch.utils.data import Dataset, DataLoader
from glob import glob
import torch
import numpy as np
import librosa as rosa
import random

from scipy.signal import sosfiltfilt
from scipy.signal import butter, cheby1, cheby2, ellip, bessel
from scipy.signal import resample_poly


def create_vctk_dataloader(hparams, cv, sr=24000):
    def collate_fn(batch):
        wav_list = list()
        wav_l_list = list()
        band_list = list()
        for wav, wav_l, band in batch:
            wav_list.append(wav)
            wav_l_list.append(wav_l)
            band_list.append(band)
        wav_list = torch.stack(wav_list, dim=0).squeeze(1)
        wav_l_list = torch.stack(wav_l_list, dim=0).squeeze(1)
        band_list = torch.stack(band_list, dim=0)

        return wav_list, wav_l_list, band_list

    if cv == 0:
        return DataLoader(dataset=VCTKMultiSpkDataset(hparams, cv),
                          batch_size=hparams.train.batch_size,
                          shuffle=True,
                          num_workers=hparams.train.num_workers,
                          collate_fn=collate_fn,
                          pin_memory=True,
                          drop_last=True,
                          sampler=None,
                          prefetch_factor=4)
    else:
        return DataLoader(dataset=VCTKMultiSpkDataset(hparams, cv) if cv == 1 else VCTKMultiSpkDataset(hparams, cv, sr),
                          collate_fn=collate_fn,
                          batch_size=hparams.train.batch_size if cv == 1 else 1,
                          drop_last=True if cv == 1 else False,
                          shuffle=False,
                          num_workers=hparams.train.num_workers,
                          prefetch_factor=4)


class VCTKMultiSpkDataset(Dataset):
    def __init__(self, hparams, cv=0, sr=24000):  # cv 0: train, 1: val, 2: test
        def _get_datalist(folder, file_format, spk_list, cv):
            _dl = []
            len_spk_list = len(spk_list)
            s = 0
            print(f'full speakers {len_spk_list}')
            for i, spk in enumerate(spk_list):
                if cv == 0:
                    if not (i < int(len_spk_list * self.cv_ratio[0])): continue
                elif cv == 1:
                    if not (int(len_spk_list * self.cv_ratio[0]) <= i and
                            i <= int(len_spk_list * (self.cv_ratio[0] + self.cv_ratio[1]))):
                        continue
                else:
                    if not (int(len_spk_list * self.cv_ratio[0]) <= i and
                            i <= int(len_spk_list * (self.cv_ratio[0] + self.cv_ratio[1]))):
                        continue
                _full_spk_dl = sorted(glob(path.join(spk, file_format)))
                _len = len(_full_spk_dl)
                if (_len == 0): continue
                s += 1
                _dl.extend(_full_spk_dl)

            print(cv, s)
            return _dl

        def _get_spk(folder):
            return sorted(glob(path.join(folder, '*')))  # [1:])

        self.hparams = hparams
        self.cv = cv
        self.cv_ratio = eval(hparams.data.cv_ratio)
        self.sr = sr
        self.directory = hparams.data.dir
        self.dataformat = hparams.data.format

        self.data_list = _get_datalist(self.directory, self.dataformat,
                                       _get_spk(self.directory), self.cv)

        assert len(self.data_list) != 0, "no data found"

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        wav, _ = rosa.load(self.data_list[index], self.hparams.audio.sampling_rate)
        wav /= np.max(np.abs(wav))

        if wav.shape[0] < self.hparams.audio.length:
            padl = self.hparams.audio.length - wav.shape[0]
            r = random.randint(0, padl) if self.cv < 2 else padl // 2
            wav = np.pad(wav, (r, padl - r), 'constant', constant_values=0)
        else:
            start = random.randint(0, wav.shape[0] - self.hparams.audio.length)
            wav = wav[start:start + self.hparams.audio.length] if self.cv < 2 \
                else wav[:len(wav) - len(wav) % self.hparams.audio.hop_length]
        wav *= random.random() / 2 + 0.5 if self.cv < 2 else 1

        if self.cv == 0:
            order = random.randint(1, 11)
            ripple = random.choice([1e-9, 1e-6, 1e-3, 1, 5])
            highcut = random.randint(self.hparams.audio.sr_min // 2, self.hparams.audio.sr_max // 2)
        else:
            order = 8
            ripple = 0.05
            if self.cv == 1:
                highcut = random.choice([8000 // 2, 12000 // 2, 16000 // 2, 24000 // 2])
            elif self.cv == 2:
                highcut = self.sr // 2

        nyq = 0.5 * self.hparams.audio.sampling_rate
        hi = highcut / nyq

        if hi == 1:
            wav_l = wav
        else:
            sos = cheby1(order, ripple, hi, btype='lowpass', output='sos')
            wav_l = sosfiltfilt(sos, wav)

            # downsample to the low sampling rate
            wav_l = resample_poly(wav_l, highcut * 2, self.hparams.audio.sampling_rate)
            # upsample to the original sampling rate
            wav_l = resample_poly(wav_l, self.hparams.audio.sampling_rate, highcut * 2)

        if len(wav_l) < len(wav):
            wav_l = np.pad(wav, (0, len(wav) - len(wav_l)), 'constant', constant_values=0)
        elif len(wav_l) > len(wav):
            wav_l = wav_l[:len(wav)]

        fft_size = self.hparams.audio.filter_length // 2 + 1
        band = torch.zeros(fft_size, dtype = torch.int64)
        band[:int(hi * fft_size)] = 1
        return torch.from_numpy(wav).float(), torch.from_numpy(wav_l.copy()).float(), band
