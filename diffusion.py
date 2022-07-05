import torch
import torch.nn as nn
import torch.nn.functional as F
from math import atan, exp

from model import NuWave2 as model


class Diffusion(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = model(hparams)

        self.logsnr_min = hparams.logsnr.logsnr_min
        self.logsnr_max = hparams.logsnr.logsnr_max

        self.logsnr_b = atan(exp(-self.logsnr_max / 2))
        self.logsnr_a = atan(exp(-self.logsnr_min / 2)) - self.logsnr_b

    def snr(self, time):
        logsnr = - 2 * torch.log(torch.tan(self.logsnr_a * time + self.logsnr_b))
        norm_nlogsnr = (self.logsnr_max - logsnr) / (self.logsnr_max - self.logsnr_min)

        alpha_sq, sigma_sq = torch.sigmoid(logsnr), torch.sigmoid(-logsnr)
        return logsnr, norm_nlogsnr, alpha_sq, sigma_sq

    def forward(self, y, y_l, band, t, z=None):
        logsnr, norm_nlogsnr, alpha_sq, sigma_sq = self.snr(t)

        if z == None:
            noise = self.model(y, y_l, band, norm_nlogsnr)
        else:
            noise = z
        return noise, logsnr, (alpha_sq, sigma_sq)

    def denoise(self, y, y_l, band, t, h):
        noise, logsnr_t, (alpha_sq_t, sigma_sq_t) = self(y, y_l, band, t)
        
        f_t = - self.logsnr_a * torch.tan(self.logsnr_a * t + self.logsnr_b)
        g_t_sq = 2 * self.logsnr_a * torch.tan(self.logsnr_a * t + self.logsnr_b)

        dzt_det = (f_t * y - 0.5 * g_t_sq * (-noise / torch.sqrt(sigma_sq_t))) * h

        denoised = y - dzt_det
        return denoised

    def denoise_ddim(self, y, y_l, band, logsnr_t, logsnr_s, z=None):
        norm_nlogsnr = (self.logsnr_max - logsnr_t) / (self.logsnr_max - self.logsnr_min)
        
        alpha_sq_t, sigma_sq_t = torch.sigmoid(logsnr_t), torch.sigmoid(-logsnr_t)

        if z == None:
            noise = self.model(y, y_l, band, norm_nlogsnr)
        else:
            noise = z

        alpha_sq_s, sigma_sq_s = torch.sigmoid(logsnr_s), torch.sigmoid(-logsnr_s)

        pred = (y - torch.sqrt(sigma_sq_t) * noise) / torch.sqrt(alpha_sq_t)

        denoised = torch.sqrt(alpha_sq_s) * pred + torch.sqrt(sigma_sq_s) * noise
        return denoised, pred

    def diffusion(self, signal, noise, s, t=None):
        bsize = s.shape[0]
        
        time = s if t is None else torch.cat([s, t], dim=0)
        
        _, _, alpha_sq, sigma_sq = self.snr(time)
        if t is not None:
            alpha_sq_s, alpha_sq_t = alpha_sq[:bsize], alpha_sq[bsize:]
            sigma_sq_s, sigma_sq_t = sigma_sq[:bsize], sigma_sq[bsize:]
            
            alpha_sq_tbars = alpha_sq_t / alpha_sq_s
            sigma_sq_tbars = sigma_sq_t - alpha_sq_tbars * sigma_sq_s
            
            alpha_sq, sigma_sq = alpha_sq_tbars, sigma_sq_tbars
        
        alpha = torch.sqrt(alpha_sq)
        sigma = torch.sqrt(sigma_sq)
        
        noised = alpha.unsqueeze(-1) * signal + sigma.unsqueeze(-1) * noise
        return alpha, sigma, noised
