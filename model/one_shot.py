"""Implementation of FewShot classifier.
"""
# Standard Python imports
from argparse import ArgumentParser

# Pytorch-related imports
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import numpy as np
# CLIP
import clip
import PIL

# For the confussion matrix
from scikitplot.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

# Utils
from model.utils import classification_block
import wandb

# Name position determines class idx
CLASS_NAMES = ['cucumber', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon',
               'lettuce', 'onion', 'orange', 'pear', 'peas', 'pineapple', 
               'pomegranate', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato',
               'tomato', 'turnip', 'watermelon']

class OneShot(pl.LightningModule):
    """OneShot classification model.
    """

    def __init__(self, backbone: str = "ViT-B/32", num_classes = len(CLASS_NAMES)):
        """Initializing the model.

        Args:
            backbone (str, optional): CLIP model used as backbone. Defaults to ViT-B/16.
            num_classes (int, optional): number of output classes. Defaults to 8.
            learning_rate (float, optional): learning rate used in the training process. Defaults to 1e-3.
            log_freq (int, optional): log stuff each log_freq batches. Defaults to 1.
        """
        super(OneShot, self).__init__()

        print("====> initializing Oneshot classifier model...")
        # self.save_hyperparameters()
        self.backbone_name = backbone.replace("/", "")

        # Instance of CLIP model used as a backbone
        self.backbone, self.preprocess = clip.load(backbone)

        # Freeze the backbone. We don't want to train the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Classification loss
        self.loss = nn.CrossEntropyLoss()

        # Accudarcy metrics
        self.train_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.valid_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

        # For confussion matrix
        self.outs = []
        self.targs = []

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Model specific-hyperparameters.

        Args:
            parent_parser (ArgumentParser): parent parser.

        Returns:
            ArgumentParser: parser updated with the new arguments.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--no_logger', action='store_true',
                            help='if true, log stuff in Neptune.')
        parser.add_argument('--max_epochs', type=int, default=15,
                            help='maximum number of training epochs.')
        parser.add_argument('--learning_rate', type=float, default=1e-3,
                            help='learning rate used for training.')

        return parser


    def test_step(self, batch, batch_idx=4) -> None:
        """Test step.

        Args:
            batch ([type]): input batch of images and its corresponding classes.
            batch_idx ([type]): batch global index.
        """
        self.backbone.eval()
        _x, _y = batch
        image_input = _x
        
        # encode caption & image
        text_descriptions = [f"This is a photo of {label}" for label in CLASS_NAMES]
        text_tokens = clip.tokenize(text_descriptions).cuda()
        with torch.no_grad():
            image_features = self.backbone.encode_image(image_input).float()
            text_features = self.backbone.encode_text(text_tokens).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        # classifier
        text_probs = (text_features @ image_features.T).softmax(dim=-1)
        # top_probs, top_labels = text_probs.T.cpu().topk(5, dim=-1)
        _out = text_probs.T

        self.test_accuracy.update(_out, _y)

        self.targs.extend(_y.cpu().numpy())
        self.outs.extend(torch.argmax(_out, dim=1).cpu().numpy())

    def on_test_epoch_end(self) -> None:
        """Callback executed at the end of testing epoch.
        """
        # get the accuracy over all batches
        _global_acc = self.test_accuracy.compute()
        print(f"Global test accuracy: {_global_acc}")
        self.logger.experiment.log({'test/acc_per_epoch': _global_acc})

        ## Get the confussion matrix
        _fig, _ax = plt.subplots(figsize=(16, 12))
        plot_confusion_matrix(y_true=self.targs, y_pred=self.outs, normalize=True, ax=_ax)

        plt.savefig(f"Oneshot_Confusion_Of_{self.backbone_name}.jpg")
        # self.save_hyperparameters()

        # self.logger.experiment.log({
        #     "test/confussion_matrix": [wandb.Image(img) 
        #     for (img) in _fig]
        # })
        # self.log_image('test/confussion_matrix', _fig)
