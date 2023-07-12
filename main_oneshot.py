"""Main File
"""

# Python standard imports
from argparse import ArgumentParser

# Pytorch and lightning imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
import os

import logging

logging.basicConfig(level=logging.INFO)

# Model
from model.one_shot import OneShot

# Datamodule
from data.dataloader import FewShotDataModule

parser = ArgumentParser()

if __name__=='__main__':
    parser = FewShotDataModule.add_model_specific_args(parser)
    parser = OneShot.add_model_specific_args(parser)
    args = parser.parse_args()

    # TODO: install WandB logger
    # Neptune logging
    # try:
    #     # Neptune credentials. User-specific. You need to create you Neptune accout!
    #     import neptune_credentials
    #     logger = NeptuneLogger(api_key=neptune_credentials.key,
    #                            project_name=neptune_credentials.project,
    #                            params=vars(args), experiment_name='fewshot',
    #                            offline_mode=args.no_logger)
        
    # except ImportError: # no neptune credentials, no logger
    #     print("No Neptune logging")
    #     logger = NeptuneLogger(offline_mode=True)

    clip_models = ['RN50',
                    'RN101',
                    'RN50x4',
                    'RN50x16',
                    'RN50x64',
                    'ViT-B/32',
                    'ViT-B/16',
                    'ViT-L/14',
                    'ViT-L/14@336px']
    

    clip_model = 'ViT-B/16'
        # try: 
    
    wandb_logger = WandbLogger(project="Deep learning", log_model=True, name=f'Oneshot learning with {clip_model} and data {args.path_to_data}')

    # Model instance
    few_model = OneShot()

    # Datamodule
    dm = FewShotDataModule(ops=few_model.preprocess, path_to_data=args.path_to_data)
    dm.prepare_data()

    # Training step
    dm.setup(stage='fit')
    print(dm.datasets)

    trainer = pl.Trainer(precision=16, logger=wandb_logger,
                        max_epochs=args.max_epochs)

    # Test stage -> loading the best model according to validation accuracy
    dm.setup(stage='test')
    trainer.test(few_model, datamodule=dm)
    wandb.finish()
