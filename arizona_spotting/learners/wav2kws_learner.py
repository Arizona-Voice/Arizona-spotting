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

from arizona_spotting.models import Wav2KWS
from arizona_spotting.utils.print_utils import *
from arizona_spotting.datasets import Wav2KWSDataset
from arizona_spotting.utils.misc_utils import extract_loudest_section
from arizona_spotting.utils.visualize_utils import plot_confusion_matrix

class Wav2KWSLearner():
    def __init__(
        self,
        model: Wav2KWS=None,
        device: str=None
    ) -> None:
        super(Wav2KWSLearner, self).__init__()

        self.model = model
        self.num_classes = self.model.num_classes

        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def train(
        self,
        train_dataset: Wav2KWSDataset,
        test_dataset: Wav2KWSDataset,
        batch_size: int=48,
        encoder_learning_rate: float=1e-5,
        decoder_learning_rate: float=5e-4,
        weight_decay: float=1e-5,
        max_steps: int=10,
        n_epochs: int=100,
        shuffle: bool=True,
        num_workers: int=4,
        view_model: bool=True,
        save_path: str='./models',
        model_name: str='wav2kws_model',
        **kwargs
    ):
        print_line(text="Dataset Info")
        print(f"Length of Training dataset: {len(train_dataset)}")
        print(f"Length of Test dataset: {len(test_dataset)} \n")
        
        _collate_fn = test_dataset._collate_fn
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle,
            pin_memory=True, collate_fn=_collate_fn, num_workers=num_workers
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            pin_memory=True, collate_fn=_collate_fn, num_workers=num_workers
        )

        self.label2idx = train_dataset.label2idx
        self.idx2label = train_dataset.idx2label

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam([
            {'params': self.model.w2v_encoder.parameters(), 'lr': encoder_learning_rate},
            {'params': self.model.decoder.parameters(), 'lr': decoder_learning_rate},
        ], weight_decay=weight_decay)

        criterion.to(self.device)
        self.model.to(self.device)

        # View the architecture of the model
        if view_model:
            print_line(text="Model Info")
            print(self.model)

        print(f"Using the device: {self.device}")

        step = 0
        best_acc = 0
        
        print_line(text="Training the model")
        
        # Check save_path exists
        if not os.path.exists(save_path):
            print(f"\nCreate a folder {save_path}")
            os.makedirs(save_path)

        for epoch in range(n_epochs):
            train_loss, train_acc = self._train(train_dataloader, optimizer, criterion)
            valid_loss, valid_acc = self._validate(test_dataloader, criterion)

            print_free_style(
                message=f"Epoch {epoch + 1}/{n_epochs}: \n"
                        f"\t- Train: loss = {train_loss:.4f}; acc = {train_acc:.4f} \n"
                        f"\t- Valid: loss = {valid_loss:.4f}; acc = {valid_acc:.4f} \n"
            )

            if valid_acc > best_acc:
                best_acc = valid_acc
                step = 0
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'label2idx': self.label2idx,
                        'idx2label': self.idx2label,
                        'model_type': None,
                        'loss': valid_loss,
                        'acc': valid_acc
                    },
                    os.path.join(save_path, f"{model_name}.pt")
                )
                print_free_style(f"Save the best model!")
            else:
                step += 1
                if step >= max_steps:
                    break

    def _train(
        self,
        train_dataloader,
        optimizer,
        criterion,
        model_type='binary'
    ):
        self.model.train()

        correct = 0
        train_loss = []
        total_sample = 0

        labels = []
        preds = []

        for item in tqdm(train_dataloader):
            x, y = item['net_input'], item['target']
            y = y.to(self.device)
            for k in x.keys():
                x[k] = x[k].to(self.device)

            optimizer.zero_grad()
            output = self.model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            total_sample += y.size(0)

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).sum()

        loss = np.mean(train_loss)
        acc = correct / total_sample

        return loss, acc
    
    def _validate(
        self,
        valid_dataloader,
        criterion=None,
        model_type='binary'
    ):
        self.model.eval()
        
        correct = 0
        valid_loss = []
        total_sample = 0
        preds = []
        labels = []
        
        with torch.no_grad():
            for item in tqdm(valid_dataloader):
                x, y = item['net_input'], item['target']
                y = y.to(self.device)
                for k in x.keys():
                    x[k] = x[k].to(self.device)

                output = self.model(x)
                if criterion:
                    loss = criterion(output, y)
                    valid_loss.append(loss.item())

                total_sample += y.size(0)

                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(y.data.view_as(pred)).sum()

                labels.extend(y.view(-1).data.cpu().numpy())
                preds.extend(pred.view(-1).data.cpu().numpy())
        
        loss = np.mean(valid_loss)
        acc = correct / total_sample

        return loss, acc

    def load_model(self, model_path):
        """Load the pretained model. """
        # Check the model file exists
        if not os.path.isfile(model_path):
            raise ValueError(f"The model file `{model_path}` is not exists or broken! ")

        checkpoint = torch.load(model_path)
        self.model_type = checkpoint['model_type']
        self.label2idx = checkpoint['label2idx']
        self.idx2label = checkpoint['idx2label']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
    def evaluate(
        self,
        test_dataset: Wav2KWSDataset=None,
        batch_size: int=48,
        num_workers: int=4,
        criterion: Any=None,
        model_type: str=None,
        view_classification_report: bool=True
    ):
        _collate_fn = test_dataset._collate_fn
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            pin_memory=True, collate_fn=_collate_fn, num_workers=num_workers
        )

        self.model.eval()
        
        correct = 0
        test_loss = []
        total_sample = 0
        preds = []
        labels = []

        print_line(f"Evaluate the model")

        with torch.no_grad():
            for item in tqdm(test_dataloader):
                x, y = item['net_input'], item['target']
                y = y.to(self.device)
                for k in x.keys():
                    x[k] = x[k].to(self.device)

                output = self.model(x)
                if criterion:
                    loss = criterion(output, y)
                    test_loss.append(loss.item())

                total_sample += y.size(0)

                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(y.data.view_as(pred)).sum()

                labels.extend(y.view(-1).data.cpu().numpy())
                preds.extend(pred.view(-1).data.cpu().numpy())
        
        loss = np.mean(test_loss)
        acc = correct / total_sample

        labels = [self.idx2label.get(i) for i in labels]
        preds = [self.idx2label.get(i) for i in preds]
        classes = list(self.label2idx.keys())

        cm = confusion_matrix(y_true=labels, y_pred=preds, labels=classes)
        
        # View classification report
        if view_classification_report:
            report = classification_report(y_true=labels, y_pred=preds, labels=classes)
            print(report)

        # Save confusion matrix image
        try:
            plot_confusion_matrix(cm, target_names=classes, title='confusion_matrix', save_dir='./evaluation')
        except Exception as e:
            print(f"Warning: {e}")

        return loss, acc

    def inference(
        self,
        input: Union[str, List],
        loudest_section: bool=True,
        sample_rate: int=1600
    ):
        """Inference a given sample from a file path or a List's float values. """

        self.model.eval()

        if isinstance(input, str):
            # Check file input exists
            if input and os.path.exists(input):
                wav, curr_sample_rate = sf.read(input)

            else:
                print(f"Warning: The `input` not exists or broken. Set the values of the `input` is zero. ")
                wav, curr_sample_rate = np.zeros(self.sample_rate, dtype=np.float32), self.sample_rate

        else:
            wav, curr_sample_rate = input, sample_rate
            
        if curr_sample_rate !=  self.sample_rate:
            wav, curr_sample_rate = librosa.resample(wav, curr_sample_rate, self.sample_rate), self.sample_rate

        if len(wav.shape) == 2:
            wav = librosa.to_mono(wav.transpose(1, 0))

        if loudest_section:
            wav = extract_loudest_section(wav, win_len=30)
        
        wav_len = len(wav)
        if wav_len < self.sample_rate:
            pad_size = self.sample_rate - wav_len
            wav = np.pad(wav, (round(pad_size / 2) + 1, round(pad_size / 2) + 1), 'constant', constant_values=0)

        wav_len = len(wav)
        mid = int(len(wav) / 2)
        cut_off = int(self.sample_rate / 2)
        wav = wav[mid - cut_off: mid + cut_off]

        feats = torch.from_numpy(wav).float()
        # feats = self.postprocess(feats, curr_sample_rate)
        x = {
            'source': feats.unsqueeze(0)
        }
        for k in x.keys():
            x[k] = x[k].to(self.device)

        with torch.no_grad():
            output = self.model(x)

        if self.model_type == 'binary':
            pred = torch.sigmoid(output)
            pred = pred[:, -1]
            pred = pred.view(-1).data.cpu().numpy()[0]
        else:
            pred = output.data.max(0, keepdim=True)[1]
            
        return (pred, self.idx2label.get(int(pred)))
