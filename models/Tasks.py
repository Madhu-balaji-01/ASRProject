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
                 lr):
        super().__init__()
        self.text_transform = text_transform        
        self.lr = lr
        self.model = model

    def training_step(self, batch, batch_idx):
        x = batch['waveforms']
        # print('x shape:', x.shape)
        if batch_idx == 0:
          self.reference_img = x[0].unsqueeze(0)
          print('ref shape', self.reference_img.shape)
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


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
#         scheduler = TriStageLRSchedule(optimizer,
#                                        [1e-8, self.lr, 1e-8],
#                                        [0.2,0.6,0.2],
#                                        max_update=len(self.train_dataloader.dataloader)*self.trainer.max_epochs)   
#         scheduler = MultiStepLR(optimizer, [1,3,5,7,9], gamma=0.1, last_epoch=-1, verbose=False)

#         return [optimizer], [{"scheduler":scheduler, "interval": "step"}]
        return [optimizer]
    
    # Vizualizing activations

    def showActivations(self,x):
      print(x.shape)
      plt.imshow(torch.Tensor.cpu(x))

      # Logging the input image
      self.logger.experiment.add_image("input",torch.Tensor.cpu(x),self.current_epoch,dataformats="HW")
      plt.show()
      plt.clf()
      out = self.model.spec_layer(x)
      # outer=(torch.Tensor.cpu(out).detach())
      # print(outer.shape)
      # plt.figure(figsize=(20,5))
      # b=np.array([]).reshape(0,outer.shape[2])
      # c=np.array([]).reshape(4*outer.shape[2],0)
        
      # # Plotting for layer 1
      # i=0
      # j=0
      # while(i<32):
      #     img=outer[0][i]
      #     print(img.shape)
      #     b=np.concatenate((img,b),axis=0)
      #     j+=1
      #     if(j==4):
      #         c=np.concatenate((c,b),axis=1)
      #         b=np.array([]).reshape(0,outer.shape[2])
      #         j=0
              
      #     i+=1
      # plt.imshow(c)
      # plt.show()
      # plt.clf()
      self.logger.experiment.add_image("Spec_layer",out[0],self.current_epoch,dataformats="HW")
      plt.imshow(torch.Tensor.cpu(out[0]))
      plt.show()
      plt.clf()

      out = out.transpose(1,2)
      out = self.model.norm_layer(out)
      # outer=(torch.Tensor.cpu(out).detach())
      # plt.figure(figsize=(10,10))
      # b=np.array([]).reshape(0,outer.shape[2])
      # c=np.array([]).reshape(8*outer.shape[2],0)
      
      # # Plotting for layer2
      # i=0
      # j=0
      # while(i<64):
      #     img=outer[0][i]
      #     b=np.concatenate((img,b),axis=0)
      #     j+=1
      #     if(j==8):
      #         c=np.concatenate((c,b),axis=1)
      #         b=np.array([]).reshape(0,outer.shape[2])
      #         j=0   
      #     i+=1

      self.logger.experiment.add_image("Norm layer",out[0],self.current_epoch,dataformats="HW")
      print('norm out0', out[0].shape)
      plt.imshow(torch.Tensor.cpu(out[0]))
      plt.show()
      plt.clf()
      
      # print(out.shape)
      out = out.unsqueeze(1)
      out = self.model.cnn(out)
      print(out.shape)
      print('out [0]', out[0].shape)
      # outer=(torch.Tensor.cpu(out).detach())
      # plt.figure(figsize=(20,5))
      # b=np.array([]).reshape(0,outer.shape[2])
      # c=np.array([]).reshape(8*outer.shape[2],0)
      
      # # Plotting for layer3
      # j=0
      # i=0
      # while(i<128):
      #     img=outer[0][i]
      #     b=np.concatenate((img,b),axis=0)
      #     j+=1
      #     if(j==8):
      #         c=np.concatenate((c,b),axis=1)
      #         b=np.array([]).reshape(0,outer.shape[2])
      #         j=0
              
      #     i+=1
      # # print(c.shape)

      self.logger.experiment.add_image("CNN layer",out[0][0],self.current_epoch,dataformats="HW")
      plt.imshow(torch.Tensor.detach(out[0][0]).cpu().numpy())
      plt.show()

    
    def training_epoch_end(self,outputs):
      self.showActivations(self.reference_img)

