import argparse

from models import TrainWav2Lip
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

parser = argparse.ArgumentParser(description='Wav2Lip training code')

parser.add_argument('--checkpoint_dir', help='Path to save model checkpoints', required=True, type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)
parser.add_argument('--gpus', type=int, default=4, help='Number of gpus to run')
parser.add_argument('--epochs', type=int, default=999999999, help='Maximum epochs to run')
parser.add_argument('--step', type=int, default=1, help='Starting resolution (1 for 4x4, 2 for 8x8 .... 6 for 128 x 128')
parser.add_argument('--psyncnet_checkpoint_path', type=str, required=True, help='Path of psyncnet save file')


opt = parser.parse_args()

if __name__ == '__main__':
    logger = TensorBoardLogger(opt.checkpoint_dir)

    if not opt.checkpoint_path:
        model = TrainWav2Lip(psyncnet_checkpoint_path=opt.psyncnet_checkpoint_path, step=opt.step)
    else:
        model = TrainWav2Lip.load_from_checkpoint(opt.checkpoint_path, psyncnet_checkpoint_path=opt.psyncnet_checkpoint_path, step=opt.step)
    
    trainer = Trainer(gpus=opt.gpus, max_epochs=opt.epochs, logger=logger, progress_bar_refresh_rate=1, check_val_every_n_epoch=100, reload_dataloaders_every_n_epochs=100)
    trainer.fit(model)
