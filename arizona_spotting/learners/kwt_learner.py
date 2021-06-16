# -*- coding: utf-8 -*-

import os
import torch
import librosa
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from typing import Any, Union, List
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from arizona_spotting.models.kwt_model import KWT

class KWTLearner():
    def __init__(
        self,
        model: KWT=None,
        device: str=None
    ) -> None:
        super(KWTLearner, self).__init__()

        self.model = model
        self.num_classes = self.model.num_classes

        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def train(self):

        raise NotImplementedError()

    def load_model(self, model_path: str=None):

        raise NotImplementedError()

    def evaluate(self):

        raise NotImplementedError()

    def inference(self, input: Any=None):

        raise NotImplementedError()
