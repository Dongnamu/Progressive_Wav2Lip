from logging import setLoggerClass
from os import sync
import torch
import torch.nn as nn
import pytorch_lightning as pl
from models import TrainDiscriminator, PWav2Lip
from torch.utils.data import DataLoader
from .dataset import Wav2LipDataset
from .hparams import hparams
import numpy as np

logloss = nn.BCELoss()
recon_loss = nn.L1Loss()

class TrainWav2Lip(pl.LightningModule):
    def __init__(self, psyncnet_checkpoint_path, update_interval=10, step=1):
        super().__init__()
        self.psync_model = TrainDiscriminator.load_from_checkpoint(checkpoint_path=psyncnet_checkpoint_path)
        self.psync_model.freeze()
        self.model = PWav2Lip()
        self.l1Losses = []
        self.synclosses = []
        self.validation_l1Losses = []
        self.validation_synclosses = []
        self.step = step
        self.validation_max_batches = 200
        self.firstCalled = True
        self.update_interval = update_interval
        self.syncnet_T = 5

    def forward(self, audio_sequences, video_sequences, step):
        return self.model(audio_sequences, video_sequences, step - 1)
    
    def cosine_loss(self,a,v,y):
        d = nn.functional.cosine_similarity(a, v)
        loss = logloss(d.unsqueeze(1), y)

        return loss

    def get_sync_loss(self, mel, g):
        g = g[:, :, :, g.size(3)//2:]
        g = torch.cat([g[:,:,i] for i in range(self.syncnet_T)], dim=1)
        a, v = self.psync_model(mel, g, self.step)
        y = torch.ones(g.size(0), 1).float().type_as(mel)
        return self.cosine_loss(a, v, y)

    # def update_image(self, x, g, gt):
    #     x = (x.detach().cpu().numpy().transpose(0,2,3,4,1) * 255.).astype(np.uint8)
    #     g = (g.detach().cpu().numpy().transpose(0,2,3,4,1) * 255.).astype(np.uint8)
    #     gt = (gt.detach().cpu().numpy().transpose(0,2,3,4,1) * 255.).astype(np.uint8)

    #     refs, inps = x[..., 3:], x[..., :3]
    #     collage = np.concatenate((refs, inps, g, gt), axis=-2)

        

    def training_step(self, batch, batch_idx):
        x, indiv_mels, mel, gt = batch

        g = self(indiv_mels, x, self.step)

        sync_loss = self.get_sync_loss(mel, g)
        
        l1loss = recon_loss(g, gt)

        loss = hparams.syncnet_wt * sync_loss + (1-hparams.syncnet_wt) * l1loss

        self.l1Losses.append(loss.item())
        self.synclosses.append(sync_loss.item())

        # if self.current_epoch % self.update_interval == 0:
        #     self.update_image(x, g, gt)

        return loss

    def training_epoch_end(self, _):
        tensorboard = self.logger.experiment
        average_l1_loss = sum(self.l1Losses) / len(self.l1Losses)
        average_sync_loss = sum(self.synclosses) / len(self.synclosses)
        tensorboard.add_scalar('Training/Wav2Lip L1 Loss', average_l1_loss, self.current_epoch)
        tensorboard.add_scalar('Training/Wav2Lip Sync Loss', average_sync_loss, self.current_epoch)
        self.l1Losses = []
        self.synclosses = []

        if average_sync_loss < 1.:
            hparams.set_hparam('syncnet_wt', 0.01)

    # def validation_step(self, batch, batch_idx):

    #     if batch_idx < self.validation_max_batches:
    #         x, indiv_mels, mel, gt = batch

    #         g = self(indiv_mels, x, self.step)

    #         sync_loss = self.get_sync_loss(mel, g)
    #         l1loss = recon_loss(g, gt)

    #         self.validation_l1Losses.append(l1loss.item())
    #         self.validation_synclosses.append(sync_loss.item())

    # def validation_epoch_end(self, _):
    #     tensorboard = self.logger.experiment
        
    #     average_l1_loss = sum(self.validation_l1Losses) / len(self.validation_l1Losses)
    #     average_sync_loss = sum(self.validation_synclosses) / len(self.validation_synclosses)
    #     tensorboard.add_scalar('Validation/Wav2Lip L1 Loss', average_l1_loss, self.current_epoch)
    #     tensorboard.add_scalar('Validation/Wav2Lip Sync Loss', average_sync_loss, self.current_epoch)

    #     self.validation_l1Losses, self.validation_synclosses = [], []

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.model.parameters(), lr=hparams.initial_learning_rate)

        return [opt_g], []

    def train_dataloader(self):
        if not self.firstCalled:
            self.step += 1
        else:
            self.firstCalled = False
        
        if self.step > 6:
            self.step = 6
        
        image_size = 4 * 2 ** (self.step - 1)

        print("Loading train dataloader with image size {}".format(image_size))
        train_dataset = Wav2LipDataset('train', image_size)
        train_dataloader = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.num_workers, persistent_workers=True, drop_last=True)
        
        return train_dataloader

    def val_dataloader(self):
        image_size = 4 * 2 ** (self.step - 1)
        print("Loading val dataloader with image size {}".format(image_size))
        val_dataset = Wav2LipDataset('val', image_size)
        val_dataloader = DataLoader(val_dataset, batch_size=hparams.batch_size, num_workers=40, persistent_workers=True, drop_last=True)

        return val_dataloader

