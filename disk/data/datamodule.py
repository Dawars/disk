"""This datamodule is responsible for loading and setting up dataloaders for DocAligner dataset"""
from pathlib import Path

import lightning.pytorch as pl
import torchvision
import torch
from torch.utils.data import DataLoader

from disk.data import get_datasets


class DataModuleTraining(pl.LightningDataModule):

    def __init__(self, args, batch_size):
        """
        Datamodule
        Args:
        """
        super().__init__()
        self.args = args
        self.batch_size = batch_size  # max(batch_size, 4)  # use 4 when searching batch size

    def setup(self, stage: str):
        # get training datasets. They will yield Images as defined in
        # disk/common/image.py. This structure contains the actual bitmap,
        # camera position and intrinsics (focal length, etc) and optionally
        # depth maps.
        self.train_chunk_iter, self.test_iter = get_datasets(
            self.args.data_path,
            no_depth=self.args.reward == 'epipolar',
            batch_size=self.batch_size,  # bs finder
            chunk_size=self.args.chunk_size,
            substep=self.args.substep,
            n_epochs=self.args.num_epochs,
            train_limit=self.args.train_scene_limit,
            test_limit=self.args.test_scene_limit,

            crop_size=(self.args.height, self.args.width),
        )

    def train_dataloader(self):
        return self.train_chunk_iter

    def val_dataloader(self):
        return self.test_iter
