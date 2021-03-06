import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import TriStageLRSchedule
from utils import extract_notes_wo_velocity, transcription_accuracy
from utils.text_processing import GreedyDecoder
import fastwer
import contextlib
import numpy as np

# from nnAudio.Spectrogram import MelSpectrogram
import pandas as pd

class ASR(pl.LightningModule):
    def __init__(self,
                 model,
                 text_transform,
                 lr, num_epochs):
        super().__init__()
        self.text_transform = text_transform        
        self.lr = lr
        self.model = model
        self.num_epochs = num_epochs

    def training_step(self, batch, batch_idx):
        x = batch['waveforms']
        # print('x shape:', x.shape)
        if batch_idx == 0:
          self.example_input_array = x
          # print('orig shape', x.shape)
        output = self.model(x)
        pred = output["prediction"]
        pred = torch.log_softmax(pred, -1) # CTC loss requires log_softmax
        loss = F.ctc_loss(pred.transpose(0, 1),
                          batch['labels'],
                          batch['input_lengths'],
                          batch['label_lengths'])        
        self.log("train_ctc_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['waveforms']
        with torch.no_grad():
            output = self.model(x)
            pred = output["prediction"]
            pred = torch.log_softmax(pred, -1) # CTC loss requires log_softmax            
            spec = output["spectrogram"]
            loss = F.ctc_loss(pred.transpose(0, 1),
                              batch['labels'],
                              batch['input_lengths'],
                              batch['label_lengths'])
            valid_metrics = {"valid_ctc_loss": loss}

            pred = pred.cpu().detach()
            decoded_preds, decoded_targets = GreedyDecoder(pred,
                                                           batch['labels'],
                                                           batch['label_lengths'],
                                                           self.text_transform)
            PER_batch = fastwer.score(decoded_preds, decoded_targets)/100            
            valid_metrics['valid_PER'] = PER_batch
            if batch_idx==0:
                self.log_images(spec, f'Valid/spectrogram')
                self._log_text(decoded_preds, 'Valid/texts_pred', max_sentences=4)
                if self.current_epoch==0: # log ground truth
                    self._log_text(decoded_targets, 'Valid/texts_label', max_sentences=4)

            self.log_dict(valid_metrics)
            
    def test_step(self, batch, batch_idx):
        x = batch['waveforms']
        with torch.no_grad():
            output = self.model(x)
            pred = output["prediction"]
            pred = torch.log_softmax(pred, -1) # CTC loss requires log_softmax
            spec = output["spectrogram"]
            loss = F.ctc_loss(pred.transpose(0, 1),
                              batch['labels'],
                              batch['input_lengths'],
                              batch['label_lengths'])
            valid_metrics = {"test_ctc_loss": loss}

            pred = pred.cpu().detach()
            decoded_preds, decoded_targets = GreedyDecoder(pred,
                                                           batch['labels'],
                                                           batch['label_lengths'],
                                                           self.text_transform)
            PER_batch = fastwer.score(decoded_preds, decoded_targets)/100            
            valid_metrics['test_PER'] = PER_batch
            if batch_idx<4:
                self.log_images(spec, f'Test/spectrogram')
                self._log_text(decoded_preds, 'Test/texts_pred', max_sentences=1)
                if batch_idx==0: # log ground truth
                    self._log_text(decoded_targets, 'Test/texts_label', max_sentences=1)

            self.log_dict(valid_metrics)     

    def _log_text(self, texts, tag, max_sentences=4):
        text_list=[]
        for idx in range(min(len(texts),max_sentences)): # visualize 4 samples or the batch whichever is smallest
            # Avoid using <> tag, which will have conflicts in html markdown
            text_list.append(texts[idx])
        s = pd.Series(text_list, name="IPA")
        self.logger.experiment.add_text(tag, s.to_markdown(), global_step=self.current_epoch)

    def log_images(self, tensor, key):
        for idx, spec in enumerate(tensor):
            fig, ax = plt.subplots(1,1)
            ax.imshow(spec.cpu().detach().t(), aspect='auto', origin='lower')    
            self.logger.experiment.add_figure(f"{key}/{idx}", fig, global_step=self.current_epoch)         


    def log_activations(self, tensor, key):
        # Add activations in a grid
        fig, ax = plt.subplots(4,4, figsize= (12,12))
        for idx, spec in enumerate(tensor):
            # print(idx)
            ax[(idx-1)//4][(idx-1)%4].imshow(spec.cpu().detach(), aspect='auto', origin='lower')            
        self.logger.experiment.add_figure(f"{key}", fig, global_step=self.current_epoch) 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
#         scheduler = TriStageLRSchedule(optimizer,
#                                        [1e-8, self.lr, 1e-8],
#                                        [0.2,0.6,0.2],
#                                        max_update=len(self.train_dataloader.dataloader)*self.trainer.max_epochs)   
#         scheduler = MultiStepLR(optimizer, [1,3,5,7,9], gamma=0.1, last_epoch=-1, verbose=False)

#         return [optimizer], [{"scheduler":scheduler, "interval": "step"}]
        return [optimizer]
    
    # Histogram of weights and biases
    def custom_histogram_adder(self):
        # Iterating through all parameters
        for name,params in self.model.named_parameters():
            self.logger.experiment.add_histogram(name,params,self.current_epoch)

    # Vizualizing activations
    def showActivations(self,x):
      # Layer 1
      out = self.model.spec_layer(x)
      out = torch.log(out+1e-8)
      self.log_activations(out,'Spec_Layer')
      # Layer 2
      out = out.transpose(1,2)
      out = self.model.norm_layer(out)
      self.log_activations(out,'Norm_Layer')
      # Layer 3
      out = out.unsqueeze(1)
      out = self.model.cnn(out)
      out = out.transpose(1,2).flatten(2)
      self.log_activations(out,'CNN_Layer')
      # Layer 4
      out = self.model.fc(out)
      self.log_activations(out,'FC_Layer')
      # Layer 5
      out = self.model.classifier(out)
      self.log_activations(out,'Classifier_Layer')
    
    def training_epoch_end(self,outputs):
      if self.current_epoch==0:
        self.showActivations(self.example_input_array)
      
      self.custom_histogram_adder()

