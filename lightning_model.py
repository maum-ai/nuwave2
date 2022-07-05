import torch
import torch.nn as nn
import pytorch_lightning as pl

import dataloader
from diffusion import Diffusion


class NuWave2(pl.LightningModule):
    def __init__(self, hparams, train=True):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = Diffusion(hparams)

        self.loss = nn.L1Loss()

    def forward(self, wav, wav_l, band, t):
        z = torch.randn(wav.shape, dtype=wav.dtype, device=wav.device)
        _, _, diffusion = self.model.diffusion(wav, z, t)

        estim, logsnr, _ = self.model(diffusion, wav_l, band, t)
        return estim, z, logsnr, wav, diffusion, logsnr

    def common_step(self, wav, wav_l, band, t):
        noise_estimation, z, logsnr, wav, wav_noisy, logsnr = self(wav, wav_l, band, t)

        loss = self.loss(noise_estimation, z)
        return loss, wav, wav_noisy, z, noise_estimation, logsnr

    @torch.no_grad()
    def inference(self, wav_l, band, step, noise_schedule=None):
        signal = torch.randn(wav_l.shape, dtype=wav_l.dtype, device=wav_l.device)
        signal_list = []
        for i in range(step):
            if noise_schedule == None:
                t = (1.0 - (i+0.5) * 1/step) * torch.ones(signal.shape[0], dtype=signal.dtype, device=signal.device)
                signal = self.model.denoise(signal, wav_l, band, t, 1/step)
            else:
                logsnr_t = noise_schedule[i] * torch.ones(signal.shape[0], dtype=signal.dtype, device=signal.device)
                if i == step-1:
                    logsnr_s = self.hparams.logsnr.logsnr_max * torch.ones(signal.shape[0], dtype=signal.dtype, device=signal.device)
                else:
                    logsnr_s = noise_schedule[i+1] * torch.ones(signal.shape[0], dtype=signal.dtype, device=signal.device)
                signal, recon = self.model.denoise_ddim(signal, wav_l, band, logsnr_t, logsnr_s)
            signal_list.append(signal)
        wav_recon = torch.clamp(signal, min=-1, max=1-torch.finfo(torch.float16).eps)
        return wav_recon, signal_list

    def training_step(self, batch, batch_idx):
        wav, wav_l, band = batch
        t = ((1 - torch.rand(1, dtype=wav.dtype, device=wav.device))
             + torch.arange(wav.shape[0], dtype=wav.dtype, device=wav.device)/wav.shape[0])%1
        loss, *_ = \
            self.common_step(wav, wav_l, band, t)

        self.log('train/loss', loss, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        wav, wav_l, band = batch

        t = ((1 - torch.rand(1, dtype=wav.dtype, device=wav.device))
            + torch.arange(wav.shape[0], dtype=wav.dtype, device=wav.device) / wav.shape[0]) % 1
        loss, wav, wav_noisy, z, z_recon, logsnr = self.common_step(wav, wav_l, band, t)

        self.log('val/loss', loss, sync_dist=True)
        if batch_idx == 0:
            i = torch.randint(0, wav.shape[0], (1,)).item()
            logsnr_t, *_ = self.model.snr(t)
            _, wav_recon = self.model.denoise_ddim(wav_noisy[i].unsqueeze(0), wav_l[i].unsqueeze(0),
                                                   band[i].unsqueeze(0), logsnr_t[i].unsqueeze(0),
                                                   torch.tensor(self.hparams.logsnr.logsnr_min, device=logsnr_t.device).unsqueeze(0),
                                                   z_recon[i].unsqueeze(0))
            signal = torch.randn(wav.shape[-1], dtype=wav.dtype, device=wav.device).unsqueeze(0)
            h = 1/1000
            wav_l_i, band_i = wav_l[i].unsqueeze(0), band[i].unsqueeze(0)
            for step in range(1000):
                timestep = (1.0 - (step + 0.5) * h) * torch.ones(signal.shape[0], dtype=signal.dtype,
                                                       device=signal.device)
                signal = self.model.denoise(signal, wav_l_i, band_i, timestep, h)
                signal = signal.clamp(-10.0, 10.0)
            wav_recon_allstep = signal.clamp(-1.0, 1.0)
            z_error = z - z_recon
            self.trainer.logger.log_spectrogram(wav[i], wav_noisy[i], z_error[i],
                                                wav_recon_allstep[0], wav_recon[0], wav_l[i],
                                                t[i].item(), logsnr[i].item(),
                                                self.global_step)
            self.trainer.logger.log_audio(wav[i], wav_noisy[i], wav_recon[0], wav_recon_allstep[0], wav_l[i], self.current_epoch)

        return {
            'val_loss': loss,
        }

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(),
                               lr=self.hparams.train.lr,
                               eps=self.hparams.train.opt_eps,
                               betas=(self.hparams.train.beta1,
                                      self.hparams.train.beta2),
                               weight_decay=self.hparams.train.weight_decay)
        return opt

    def train_dataloader(self):
        return dataloader.create_vctk_dataloader(self.hparams, 0)

    def val_dataloader(self):
        return dataloader.create_vctk_dataloader(self.hparams, 1)

    def test_dataloader(self, sr):
        return dataloader.create_vctk_dataloader(self.hparams, 2, sr)
