import argparse

from models import TrainDiscriminator
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import os

parser = argparse.ArgumentParser(description='Wav2Lip Expert Discrininator training code')

# parser.add_argument('--data_root', help="Root folder of the preprocessed dataset", required=True, type=str)
parser.add_argument('--checkpoint_dir', help="Path to save model checkpoints", required=True, type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)
parser.add_argument('--gpus', type=int, default=4, help='Number of gpus to run')
parser.add_argument('--update_every', type=int, default=1, help='Tensorboard update frequency')
parser.add_argument('--epochs', type=int, default=999999999, help='Maximum epochs to run')
parser.add_argument('--step', type=int, default=1, help='Starting resolution (1 for 4x4, 2 for 8x8 ... 6 for 128 x 128')

opt = parser.parse_args()

if __name__ == '__main__':
    logger = TensorBoardLogger(opt.checkpoint_dir)

    if opt.checkpoint_path:
        model = TrainDiscriminator.load_from_checkpoint(checkpoint_path=opt.checkpoint_path, update_interval=opt.update_every, step=opt.step)
    else:
        model = TrainDiscriminator(opt.update_every, opt.step)

    trainer = Trainer(gpus=opt.gpus, max_epochs=opt.epochs, logger=logger, progress_bar_refresh_rate=1, check_val_every_n_epoch=10, reload_dataloaders_every_n_epochs=100)
    trainer.fit(model)