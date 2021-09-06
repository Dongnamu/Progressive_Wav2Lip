from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import tensorboard
import torch
import torch.nn as nn
import pytorch_lightning as pl
from models import PSyncNet_color
from torch.utils.data import DataLoader
from .dataset import SyncDataset
from .hparams import hparams

logloss = nn.BCELoss()

class TrainDiscriminator(pl.LightningModule):
    def __init__(self, step=1) -> None:
        super().__init__()
        self.model = PSyncNet_color()
        # self.update_interval = update_interval
        self.training_losses = []
        self.validation_losses = []
        # self.average_loss = float("inf")
        self.validation_max_batches = 200
        self.step = step
        self.firstCalled = True
        # self.dataloaderUpdated = False

        # self.model.apply(self.initialise_weights)

    def initialise_weights(self, model):
        if isinstance(model, nn.Conv2d):
            nn.init.xavier_normal_(model.weight.data)
            nn.init.zeros_(model.bias.data)

    def forward(self, audio_sequences, video_sequences, step):
        return self.model(audio_sequences, video_sequences, step - 1)

    def cosine_loss(self,a, v, y):
        d = nn.functional.cosine_similarity(a, v)
        loss = logloss(d.unsqueeze(1), y)

        return loss

    def training_step(self, batch, batch_idx):
        x, mel, y = batch

        a, v = self(mel, x, self.step)

        loss = self.cosine_loss(a, v, y)

        self.training_losses.append(loss)

        # tensorboard = self.logger.experiment
        # tensorboard.add_scalar('Training/Expert Discriminator Loss Step', loss, self.global_step)

        return loss

    def training_epoch_end(self, _):
        tensorboard = self.logger.experiment
        average_loss = sum(self.training_losses) / len(self.training_losses)
        tensorboard.add_scalar('Training/Expert Discriminator Loss Epoch', average_loss, self.current_epoch)
        self.training_losses = []

    def validation_step(self, batch, batch_idx):
        
        if batch_idx < self.validation_max_batches:
        
            x, mel, y = batch

            a, v = self(mel, x, self.step)

            self.validation_losses.append(self.cosine_loss(a, v, y))

    
    def validation_epoch_end(self, _):
        tensorboard = self.logger.experiment
        tensorboard.add_scalar('Validation/Expert Discriminator Loss', sum(self.validation_losses) / len(self.validation_losses))
        self.validation_losses = []


    def configure_optimizers(self):
        opt_d = torch.optim.Adam(self.model.parameters(), lr=hparams.syncnet_lr)
        
        return [opt_d], []

    def train_dataloader(self):
        
        if not self.firstCalled:
            self.step += 1
        else:
            self.firstCalled = False

        if self.step > 6:
            self.step = 6

        image_size = 4 * 2 ** (self.step - 1)
        print("Loading train dataloader with image size {}".format(image_size))
        train_dataset = SyncDataset('train', image_size)
        train_dataloader = DataLoader(train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=True, num_workers=hparams.num_workers, persistent_workers=True, drop_last=True)
        # val_dataset = SyncDataset('val', image_size)
        # val_dataloader = DataLoader(val_dataset, batch_size=hparams.syncnet_batch_size, num_workers=40, persistent_workers=True, drop_last=True)
        # self.val_dataloader=val_dataloader
        return train_dataloader

    def val_dataloader(self):
        
        image_size = 4 * 2 ** (self.step - 1)
        print("Loading val dataloader with image size {}".format(image_size))
        val_dataset = SyncDataset('val', image_size)
        val_dataloader = DataLoader(val_dataset, batch_size=hparams.syncnet_batch_size, num_workers=40, persistent_workers=True, drop_last=True)
        
        return val_dataloader        

