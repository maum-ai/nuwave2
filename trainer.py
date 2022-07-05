from lightning_model import NuWave2
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf as OC
import os
import argparse
import datetime
from glob import glob
from pytorch_lightning.callbacks.base import Callback
import torch
from pytorch_lightning.utilities import rank_zero_only
from copy import deepcopy
from utils.tblogger import TensorBoardLoggerExpanded

# Other DDPM/Score-based model applied EMA
# In our works, there are no significant difference
class EMACallback(Callback):
    def __init__(self, filepath, alpha=0.999, k=3):
        super().__init__()
        self.alpha = alpha
        self.filepath = filepath
        self.k = 3 #max_save
        self.queue = []
        self.last_parameters = None

    @rank_zero_only
    def _del_model(self, removek):
        if os.path.exists(self.filepath.format(epoch=removek)):
            os.remove(self.filepath.format(epoch=removek))

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module,batch, batch_idx,dataloader_idx):
        if hasattr(self, 'current_parameters'):
            self.last_parameters = self.current_parameters
        else:
            self.last_parameters = deepcopy(pl_module.state_dict())

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module,outputs, batch, batch_idx,dataloader_idx):
        self.current_parameters = deepcopy(pl_module.state_dict())
        for k, v in self.current_parameters.items():
            self.current_parameters[k].copy_(self.alpha * v +
                                             (1. - self.alpha) *
                                             self.last_parameters[k])
        del self.last_parameters
        return

    @rank_zero_only
    def on_epoch_end(self, trainer, pl_module):
        if hasattr(self, 'current_parameters'):
            self.queue.append(trainer.current_epoch)
            torch.save(self.current_parameters,
                       self.filepath.format(epoch=trainer.current_epoch))
            pl_module.print(
                f'{self.filepath.format(epoch = trainer.current_epoch)} is saved')

            while len(self.queue) > self.k:
                self._del_model(self.queue.pop(0))

        else:
            self.current_parameters = deepcopy(pl_module.state_dict())

        return


def train(args):
    hparams = OC.load('hparameter.yaml')
    now = datetime.datetime.now().strftime('%m_%d_%H')
    hparams.name = f"{hparams.log.name}_{now}"
    os.makedirs(hparams.log.tensorboard_dir, exist_ok=True)
    os.makedirs(hparams.log.checkpoint_dir, exist_ok=True)
    model = NuWave2(hparams)
    tblogger = TensorBoardLoggerExpanded(hparams)
    ckpt_path = f'{hparams.log.name}_{now}_{{epoch}}'
    checkpoint_callback = ModelCheckpoint(dirpath=hparams.log.checkpoint_dir,
                                          filename=ckpt_path,
                                          verbose=True,
                                          save_last=True,
                                          save_top_k=3,
                                          monitor='val/loss',
                                          mode='min',
                                          prefix='')

    if args.restart:
        ckpt = torch.load(glob(
            os.path.join(hparams.log.checkpoint_dir,
                         f'*_epoch={args.resume_from}.ckpt'))[-1],
                          map_location='cpu')
        ckpt_sd = ckpt['state_dict']
        sd = model.state_dict()
        for k, v in sd.items():
            if k in ckpt_sd:
                if ckpt_sd[k].shape == v.shape:
                    sd[k].copy_(ckpt_sd[k])
    if args.ema:
        ckpt = torch.load(glob(
            os.path.join(hparams.log.checkpoint_dir,
                         f'*_epoch={args.resume_from}_EMA'))[-1],
                          map_location='cpu')
        print(ckpt.keys())
        sd = model.state_dict()
        for k, v in sd.items():
            if k in ckpt:
                if ckpt[k].shape == v.shape:
                    sd[k].copy_(ckpt[k])
        args.resume_from = None


    trainer = Trainer(
        checkpoint_callback=checkpoint_callback,
        gpus=hparams.train.gpus,
        accelerator='ddp' if hparams.train.gpus > 1 else None,
        # plugins='ddp_sharded',
        amp_backend='apex',  #
        amp_level='O2',  #
        #num_sanity_val_steps = -1,
        check_val_every_n_epoch=1,
        # val_check_interval=0.1,
        gradient_clip_val = 0.5,
        max_epochs=200000,
        logger=tblogger,
        progress_bar_refresh_rate=4,
        callbacks=[
            EMACallback(os.path.join(hparams.log.checkpoint_dir,
                         f'{hparams.name}_epoch={{epoch}}_EMA'))
                   ],
        resume_from_checkpoint=None
        if args.resume_from == None or args.restart else sorted(
            glob(
                os.path.join(hparams.log.checkpoint_dir,
                             f'*_epoch={args.resume_from}.ckpt')))[-1])
    trainer.fit(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume_from', type =int,\
            required = False, help = "Resume Checkpoint epoch number")
    parser.add_argument('-s', '--restart', action = "store_true",\
            required = False, help = "Significant change occured, use this")
    parser.add_argument('-e', '--ema', action = "store_true",\
            required = False, help = "Start from ema checkpoint")
    args = parser.parse_args()
    #torch.backends.cudnn.benchmark = False
    train(args)
