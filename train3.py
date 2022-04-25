from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import time
from tqdm.notebook import tqdm
import numpy as np
from pathlib import Path
import pandas as pd
import random
import cv2
import matplotlib.pyplot as plt

# Torch imports 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms
from torchvision.ops.boxes import box_iou
from torchvision.models.detection._utils import Matcher
from torchvision.ops import nms, box_convert
import torch.nn.functional as F

# Albumentations is used for the Data Augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Pytorch import
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from collections import OrderedDict
from torchvision.models.detection.image_list import ImageList
from typing import List, Tuple

import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
#from model.utils.config import cfg
import torch.nn as nn
from torch.autograd import Function
import cv2

seed_everything(25081992)

class CarDataset(Dataset):


    def __init__(self, root, csv_file, image_set, domain, transform):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional data augmentation to be applied
                on a sample.
        """

        #self.root_dir = Path(root_dir)
        self.root = root
        annotations = pd.read_csv(csv_file)
        self.image_set = image_set
        self.image_path = annotations["image_name"]
        self.boxes = [self.decodeString(item) for item in annotations["BoxesString"]]
        self.domain = domain
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        
        imgp = self.root+self.image_path[idx]
        bboxes = self.boxes[idx]
        img = cv2.imread(imgp)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image,bboxes=bboxes,class_labels=["car"]*len(bboxes)) #Albumentations can transform images and boxes
        image = transformed["image"]/255.0
        bboxes = transformed["bboxes"]
          
        if len(bboxes) > 0:
          bboxes = torch.stack([torch.tensor(item) for item in bboxes])
        else:
          bboxes = torch.zeros((0,4))
          
               
        return image, bboxes, torch.tensor(self.domain)

              
    def decodeString(self,BoxesString):
      """
      Small method to decode the BoxesString
      """
      if BoxesString == "no_box":
          return np.zeros((0,4))
      else:
          try:
              boxes =  np.array([np.array([float(i) for i in box.split(" ")])
                              for box in BoxesString.split(";")])
              return boxes.astype(np.float32).clip(min=0)
          except:
              print(BoxesString)
              print("Submission is not well formatted. empty boxes will be returned")
              return np.zeros((0,4))



train_transform = A.Compose(
        [
        A.Resize(height=600, width=1200,p=1.0),
        #A.HorizontalFlip(p=0.5),
        ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels'],min_visibility=0.2)
    )


valid_transform = A.Compose([
    A.Resize(height=600, width=1200,p=1.0),
    ToTensorV2(p=1.0),
],p=1.0,bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels'],min_visibility=0.2)
)


#For results in the second column (S, C) --> B please uncomment the following three lines
tr_dataset1 = CarDataset(csv_file='./datasets/Annots/sim10k_train_car.csv', root='./datasets/sim10k/VOC2012/JPEGImages/',  image_set = 'train', domain=0, transform=train_transform)
tr_dataset2 = CarDataset(csv_file='./datasets/Annots/cityscapes_train_car.csv', root='./datasets/cityscapes/train/', image_set = 'train', domain=1, transform=train_transform) 
vl_dataset = CarDataset(csv_file='./datasets/Annots/bdd100k_val_car.csv', root='./datasets/bdd100k/images/100k/val/', image_set = 'val', domain=0, transform=valid_transform)

#For results in the third column (S, B) --> C please uncomment the following three lines
#tr_dataset1 = CarDataset(csv_file='./datasets/Annots/sim10k_train_car.csv', root='./datasets/sim10k/VOC2012/JPEGImages/',  image_set = 'train', domain=0, transform=train_transform)
#tr_dataset2 = CarDataset(csv_file='./datasets/Annots/bdd100k_train_car.csv', root='./datasets/bdd100k/images/100k/train/', image_set = 'train', domain=1, transform=train_transform)
#vl_dataset = CarDataset(csv_file='./datasets/Annots/cityscapes_val_car.csv', root='./datasets/cityscapes/val/', image_set = 'val', domain=0, transform=valid_transform)

#For results in the fourth column (B, C) --> S please uncomment the following three lines
#tr_dataset1 = CarDataset(csv_file='./datasets/Annots/bdd100k_train_car.csv', root='./datasets/bdd100k/images/100k/train/', image_set = 'train', domain=0, transform=train_transform)
#tr_dataset2 = CarDataset(csv_file='./datasets/Annots/cityscapes_train_car.csv', root='./datasets/cityscapes/train/', image_set = 'train', domain=1, transform=train_transform) 
#vl_dataset = CarDataset(csv_file='./datasets/Annots/sim10k_val_car.csv', root='./datasets/sim10k/VOC2012/JPEGImages/', image_set = 'val', domain=0, transform=valid_transform)


tr_dataset = torch.utils.data.ConcatDataset([tr_dataset1, tr_dataset2]) 
val_dataloader = torch.utils.data.DataLoader(vl_dataset, batch_size=1, shuffle=False,  collate_fn=collate_fn)

def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
    """

    images = list()
    targets=list()
    
    domain_labels = list()
    for i, t, d in batch:
        images.append(i)
        targets.append(t)
        
        domain_labels.append(d)
    images = torch.stack(images, dim=0)

    return images, targets, domain_labels


class GRLayer(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.alpha=0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output=grad_outputs.neg() * ctx.alpha
        return output

def grad_reverse(x):
    return GRLayer.apply(x)
    
class _InstanceDA(nn.Module):
    def __init__(self, num_domains):
        super(_InstanceDA,self).__init__()
        self.num_domains = num_domains
        self.dc_ip1 = nn.Linear(1024, 512)
        self.dc_relu1 = nn.ReLU()
        #self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(512, 256)
        self.dc_relu2 = nn.ReLU()
        #self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer=nn.Linear(256,self.num_domains)
        

    def forward(self,x):
        x=grad_reverse(x)
        x=self.dc_relu1(self.dc_ip1(x))
        x=self.dc_ip2(x)
        x=torch.sigmoid(self.classifer(x))

        return x

class _InsClsPrime(nn.Module):
    def __init__(self, num_cls):
        super(_InsClsPrime,self).__init__()
        self.num_cls = num_cls
        self.dc_ip1 = nn.Linear(1024, 512)
        self.dc_relu1 = nn.ReLU()
        #self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(512, 256)
        self.dc_relu2 = nn.ReLU()
        #self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer=nn.Linear(256,self.num_cls)
        

    def forward(self,x):
        x=grad_reverse(x)
        x=self.dc_relu1(self.dc_ip1(x))
        x=self.dc_ip2(x)
        x=torch.sigmoid(self.classifer(x))

        return x

class _InsCls(nn.Module):
    def __init__(self, num_cls):
        super(_InsCls,self).__init__()
        self.num_cls = num_cls
        self.dc_ip1 = nn.Linear(1024, 512)
        self.dc_relu1 = nn.ReLU()
        #self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(512, 256)
        self.dc_relu2 = nn.ReLU()
        #self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer=nn.Linear(256,self.num_cls)
        

    def forward(self,x):
        x=self.dc_relu1(self.dc_ip1(x))
        x=self.dc_ip2(x)
        x=torch.sigmoid(self.classifer(x))

        return x
                     
class _ImageDA(nn.Module):
    def __init__(self,dim,num_domains):
        super(_ImageDA,self).__init__()
        self.dim=dim  # feat layer          256*H*W for vgg16
        self.num_domains = num_domains
        self.Conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=(2,4))
        self.Conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=4)
        self.Conv3 = nn.Conv2d(256, 256, kernel_size=2, stride=4)
        self.Conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=3)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, self.num_domains)
        self.reLu=nn.ReLU(inplace=False)
        

 
        
    def forward(self,x):
        x=grad_reverse(x)
        x=self.reLu(self.Conv1(x))
        x=self.reLu(self.Conv2(x))
        x=self.reLu(self.Conv3(x))
        x=self.reLu(self.Conv4(x))
        x=self.flatten(x)
        x=self.reLu(self.linear1(x))
        x=torch.sigmoid(self.linear2(x))
        
        return x

 
import fasterrcnn
class myDAFasterRCNN(LightningModule):
    def __init__(self,n_classes):
        super(myDAFasterRCNN, self).__init__()
        
        self.detector = fasterrcnn.fasterrcnn_resnet50_fpn(min_size=600, max_size=1200, pretrained=True,trainable_backbone_layers=1) 
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)


        self.ImageDA = _ImageDA(256, 2)
        self.InsDA = _InstanceDA(2)       
        self.InsCls = nn.ModuleList([_InsCls(2) for i in range(2)])
        self.InsClsPrime = nn.ModuleList([_InsClsPrime(2) for i in range(2)])

        self.best_val_acc = 0
        self.val_acc_stack = [[] for i in range(1)]
        self.freq = torch.tensor(np.zeros(n_classes))
        self.log('val_loss', 100000)
        self.log('val_acc', self.best_val_acc)
        self.n_classes = n_classes

        self.base_lr = 2e-3 #Original base lr is 1e-4
        self.momentum = 0.9
        self.weight_decay=0.0005

        self.detector.backbone.register_forward_hook(self.store_backbone_out)
        self.detector.roi_heads.box_head.register_forward_hook(self.store_ins_features)

        self.mode = 0
        self.sub_mode = 0

    def store_ins_features(self, module, input1, output):
      self.box_features = output
      self.box_labels = input1[1] #Torch tensor of size 512
      
            
    def store_backbone_out(self, module, input1, output):
      self.base_feat = output
      
    def forward(self, imgs,targets=None):
      # Torchvision FasterRCNN returns the loss during training 
      # and the boxes during eval
      self.detector.eval()
      return self.detector(imgs)
    
    def configure_optimizers(self):
      
      optimizer = torch.optim.SGD([{'params': self.detector.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay },
                                    {'params': self.ImageDA.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay },
                                    {'params': self.InsDA.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay },
                                    {'params': self.InsCls.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay },
                                    {'params': self.InsClsPrime.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay}
                                      ],) 
      
      lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, threshold=0.0001, min_lr=0, eps=1e-08),
                      'monitor': 'val_loss'}
      
      
      return [optimizer], [lr_scheduler]
      
     
    
    def train_dataloader(self):
      num_train_sample_batches = len(tr_dataset)//2
      temp_indices = np.array([i for i in range(len(tr_dataset))])
      np.random.shuffle(temp_indices)
      sample_indices = []
      for i in range(num_train_sample_batches):
  
        batch = temp_indices[2*i:2*(i+1)]
  
        for index in batch:
          sample_indices.append(index)  #This is for mode 0
  
        for index in batch:		   #This is for mode 1
          sample_indices.append(index)

        
      
      return torch.utils.data.DataLoader(tr_dataset, batch_size=2, sampler=sample_indices, shuffle=False, collate_fn=collate_fn)
      
    def training_step(self, batch, batch_idx):
      
      
      
      imgs = list(image.cuda() for image in batch[0]) 
      

      targets = []
      for boxes, domain in zip(batch[1], batch[2]):
        target= {}
        target["boxes"] = boxes.float().cuda()
        target["labels"] = torch.ones(len(target["boxes"])).long().cuda()
        targets.append(target)

      # fasterrcnn takes both images and targets for training, returns
      #Detection using source images
      
      if self.mode == 0:
        temp_loss = []
        for index in range(len(imgs)):
          detections = self.detector([imgs[index]], [targets[index]])
          #total_main_loss = sum(loss1 for loss1 in detections[0]['losses'].values())
          
          

          ImgDA_scores = self.ImageDA(self.base_feat['0'])
          detections[0]['losses']['DA_img_loss'] = F.cross_entropy(ImgDA_scores, torch.unsqueeze(batch[2][index], 0))
          IDA_out = self.InsDA(self.box_features)
          detections[0]['losses']['DA_ins_loss'] = 0.1*F.cross_entropy(IDA_out, batch[2][index].repeat(IDA_out.shape[0]).long())
          detections[0]['losses']['Cst_loss'] = F.mse_loss(IDA_out, ImgDA_scores[0].repeat(IDA_out.shape[0],1))

                   
          temp_loss.append(sum(loss1 for loss1 in detections[0]['losses'].values()))

               
        loss = torch.mean(torch.stack(temp_loss))
        

        
        if(self.sub_mode == 0):
          self.mode = 1
          self.sub_mode = 1
        elif(self.sub_mode == 1):
          self.mode = 2
          self.sub_mode = 2
        elif(self.sub_mode == 2):
          self.mode = 3
          self.sub_mode = 3
        else:
          self.sub_mode = 0
          self.mode = 0
        
                 

      elif(self.mode == 1): #Without recording the gradients for detector, we need to update the weights for classifier weights
        loss_dict = {}
        loss = []

        
        for index in range(len(self.InsCls)):
          for param in self.InsCls[index].parameters(): param.requires_grad = True

        for index in range(len(imgs)):
          with torch.no_grad():
            _ = self.detector([imgs[index]], [targets[index]])
          cls_scores = self.InsCls[batch[2][index].item()](self.box_features)
          loss.append(F.cross_entropy(cls_scores, self.box_labels[0])) 

        loss_dict['cls'] = 0.05*(torch.mean(torch.stack(loss)))
        loss = sum(loss for loss in loss_dict.values())

        self.mode = 0
      elif(self.mode == 2): #Only the GRL Classification should influence the updates but here we need to update the detector weights as well
        loss_dict = {}
        loss = []
    
        for index in range(len(imgs)):
          _ = self.detector([imgs[index]], [targets[index]])
          cls_scores = self.InsClsPrime[batch[2][index].item()](self.box_features)
          loss.append(F.cross_entropy(cls_scores, self.box_labels[0]))
  	  
        loss_dict['cls_prime'] = 0.001*(torch.mean(torch.stack(loss)))
        loss = sum(loss for loss in loss_dict.values())

        self.mode = 0
        
      else: #For Mode 3
      
        loss_dict = {}
        loss = []
        consis_loss = []
        
        for index in range(len(self.InsCls)):
          for param in self.InsCls[index].parameters(): param.requires_grad = False
        
        for index in range(len(imgs)):
          _ = self.detector([imgs[index]], [targets[index]])
          temp = []
          for i in range(len(self.InsCls)):
            if(i != batch[2][index].item()):
              cls_scores = self.InsCls[i](self.box_features)
              temp.append(cls_scores)
              loss.append(F.cross_entropy(cls_scores, self.box_labels[0]))
          consis_loss.append(torch.mean(torch.abs(torch.stack(temp, dim=0) - torch.mean(torch.stack(temp, dim=0), dim=0))))

        loss_dict['cls'] = 0.05*(torch.mean(torch.stack(loss))) + torch.mean(torch.stack(consis_loss))
        loss = sum(loss for loss in loss_dict.values())
        
        self.mode = 0
        self.sub_mode = 0
 	 
      return {"loss": loss}#, "log": torch.stack(temp_loss).detach().cpu()}

    def validation_step(self, batch, batch_idx):
      img, boxes, domain = batch
      
      preds = self.forward(img)
      preds[0]['boxes'] = preds[0]['boxes'][preds[0]['scores'] > 0.5]
      #self.val_acc = torch.mean(torch.stack([self.accuracy(b,pb["boxes"],iou_threshold=0.5) for b,pb in zip(boxes,pred_boxes)]))
      self.val_acc_stack[domain[0]].append(torch.stack([self.accuracy(b,pb["boxes"],iou_threshold=0.5) for b,pb in zip(boxes,preds)]))

      #return val_acc_stack
    
    def validation_epoch_end(self, validation_step_outputs):

         
      temp = 0
      non_zero_domains = 0
      for item in range(len(self.val_acc_stack)):
        
        if(self.val_acc_stack[item]):
          temp = temp + torch.mean(torch.stack(self.val_acc_stack[item]))
          non_zero_domains = non_zero_domains + 1
          print(torch.mean(torch.stack(self.val_acc_stack[item])))
          
      temp = temp/non_zero_domains 
      self.log('val_loss', 1 - temp)  #Logging for model checkpoint
      self.log('val_acc', temp)
      if(self.best_val_acc < temp):
              self.best_val_acc = temp
              self.best_val_acc_epoch = self.trainer.current_epoch
      

      self.val_acc_stack = [[] for i in range(1)]
      print('Validation ADA: ',temp)
      self.mode = 0
      self.sub_mode=0
      
      
    def test_step(self, batch, batch_idx):
      img, boxes, metadata = batch
      pred_boxes = self.forward(img) # in validation, faster rcnn return the boxes
      self.test_loss = torch.mean(torch.stack([self.accuracy(b,pb["boxes"],iou_threshold=0.5) for b,pb in zip(boxes,pred_boxes)]))
      return self.test_loss

   

    def accuracy(self, src_boxes,pred_boxes ,  iou_threshold = 1.):
      """
      #The accuracy method is not the one used in the evaluator but very similar
      """
      total_gt = len(src_boxes)
      total_pred = len(pred_boxes)
      if total_gt > 0 and total_pred > 0:


        # Define the matcher and distance matrix based on iou
        matcher = Matcher(iou_threshold,iou_threshold,allow_low_quality_matches=False) 
        match_quality_matrix = box_iou(src_boxes,pred_boxes)

        results = matcher(match_quality_matrix)
        
        true_positive = torch.count_nonzero(results.unique() != -1)
        matched_elements = results[results > -1]
        
        #in Matcher, a pred element can be matched only twice 
        false_positive = torch.count_nonzero(results == -1) + ( len(matched_elements) - len(matched_elements.unique()))
        false_negative = total_gt - true_positive

            
        return  true_positive / (true_positive + false_positive) #mAP for cityscapes

      elif total_gt == 0:
          if total_pred > 0:
              return torch.tensor(0.).cuda()
          else:
              return torch.tensor(1.).cuda()
      elif total_gt > 0 and total_pred == 0:
          return torch.tensor(0.).cuda()
          
    


          
import os
detector = myDAFasterRCNN(2)
NET_FOLDER = 'CAR_Res50FPN_cyclic'
weights_file = 'best_prop'

if(os.path.exists(NET_FOLDER+'/'+weights_file+'.ckpt')):
  detector.load_state_dict(torch.load(NET_FOLDER+'/'+weights_file+'.ckpt')['state_dict'])
else:
  if not os.path.exists(NET_FOLDER):
    mode = 0o777
    os.mkdir(NET_FOLDER, mode)

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
early_stop_callback= EarlyStopping(monitor='val_acc', min_delta=0.00, patience=10, verbose=False, mode='max')

checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=NET_FOLDER, filename=weights_file)
trainer = Trainer(gpus=1, progress_bar_refresh_rate=1, max_epochs=100, deterministic=False, callbacks=[checkpoint_callback, early_stop_callback], reload_dataloaders_every_n_epochs=1)
trainer.fit(detector, val_dataloaders=val_dataloader)


detector.load_state_dict(torch.load(NET_FOLDER+'/'+weights_file+'.ckpt')['state_dict'])


detector.freeze()
detector.detector.eval()
test_dataloader = torch.utils.data.DataLoader(vl_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
def acc_new(src_boxes, pred_boxes, iou_threshold = 1.):
      
      #The accuracy method is not the one used in the evaluator but very similar
      
      total_gt = len(src_boxes)
      total_pred = len(pred_boxes)
      if total_gt > 0 and total_pred > 0:


        # Define the matcher and distance matrix based on iou
        matcher = Matcher(iou_threshold,iou_threshold,allow_low_quality_matches=False) 
        match_quality_matrix = box_iou(src_boxes,pred_boxes)

        results = matcher(match_quality_matrix)
        
        true_positive = torch.count_nonzero(results.unique() != -1)
        matched_elements = results[results > -1]
        
        #in Matcher, a pred element can be matched only twice 
        false_positive = torch.count_nonzero(results == -1) + ( len(matched_elements) - len(matched_elements.unique()))
        false_negative = total_gt - true_positive

            
        return  true_positive / ( true_positive + false_positive)

      elif total_gt == 0:
          if total_pred > 0:
              return torch.tensor(0.).cuda()
          else:
              return torch.tensor(1.).cuda()
      elif total_gt > 0 and total_pred == 0:
          return torch.tensor(0.).cuda()
          

detector.to('cuda')
val_acc_stack = [[] for i in range(1)]
test_dataset = iter(test_dataloader)
domain = torch.zeros(1)
for index, data_sample in enumerate(test_dataset):

  images, boxes, labels = data_sample
   
  preds = detector(images.cuda())
  
  preds[0]['boxes'] = preds[0]["boxes"].detach().cpu()
  preds[0]['labels'] = preds[0]["labels"].detach().cpu()
  preds[0]['scores'] = preds[0]["scores"].detach().cpu()
  
  preds[0]['boxes'] = preds[0]['boxes'][preds[0]['scores'] > 0.5]
  val_acc_stack[labels[0]].append(torch.stack([acc_new(b,pb["boxes"],iou_threshold=0.5) for b,pb in zip(boxes,preds)]).detach().cpu())
      
  domain[labels[0]] = domain[labels[0]] + 1
  #print(index)
  
      
#ADA_whole = torch.mean(torch.stack(val_acc_stack))

#print(ADA_whole)
weights = [1/domain[i] for i in range(1)]
temp = 0
test_acc = []
for index in range(len(val_acc_stack)):

  if(len(val_acc_stack[index]) == 0):
    print(str(index)+'  is empty')
  else:
    temp = temp + weights[index]*torch.sum(torch.stack(val_acc_stack[index]))
    test_acc.append(torch.mean(torch.stack(val_acc_stack[index])).item())
    print(torch.mean(torch.stack(val_acc_stack[index])))
    
np.savetxt(NET_FOLDER+'/test_acc.txt',np.array(test_acc))
print('WAD:', temp)
