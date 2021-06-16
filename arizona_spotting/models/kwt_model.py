# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn

from typing import Any

class KWT(nn.Module):
    def __init__(
        self,
        num_classes: int=2,
    ) -> None:
        super(KWT, self).__init__()

        self.num_classes = num_classes

    def forward(self, x:Any):
        
        return 