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
from torch.autograd import Variable
from torch.autograd import Function

from torchvision.models import ResNet50_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms
from torchvision.ops.boxes import box_iou
from torchvision.models.detection._utils import Matcher
import torchvision.models as models
from torchvision.ops import nms, box_convert
from torchvision.ops import generalized_box_iou_loss 
# Albumentations is used for the Data Augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Pytorch import
from pytorch_lightning.core.module import LightningModule
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


class WheatDataset(Dataset):
    """A dataset example for GWC 2021 competition."""

    def __init__(self, csv_file, root_dir, image_set, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional data augmentation to be applied
                on a sample.
        """

        annotations = pd.read_csv(csv_file)
        self.image_set = image_set
        self.image_path = root_dir+annotations["image_name"]
        self.boxes = [self.decodeString(item) for item in annotations["BoxesString"]]
        self.domains_str = annotations['domain']
        
        if(image_set == 'train'):
          self._domains = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17')
        elif(image_set == 'val'):
          self._domains = ('18', '19', '20', '21', '22', '23', '24', '25')
        else:
          self._domains = ('26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46')
        
        self.num_domains = len(self._domains)
        self._domain_to_ind = dict(zip(self._domains, range(len(self._domains))))
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        
        imgp = self.image_path[idx]
        bboxes = self.boxes[idx]
        img = cv2.imread(imgp)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Opencv open images in BGR mode by default
        
        try:
          domain = torch.tensor(self._domain_to_ind[str(self.domains_str[idx])])
        except:
          domain = torch.tensor(-1)
        
        bboxes = np.clip(np.array(bboxes), a_min=0, a_max=1024)
        bboxes[:, 0][bboxes[:, 0] == bboxes[:, 2]] = bboxes[:, 0][bboxes[:, 0] == bboxes[:, 2]] - 1
        bboxes[:, 1][bboxes[:, 1] == bboxes[:, 3]] = bboxes[:, 1][bboxes[:, 1] == bboxes[:, 3]] - 1
      
        transformed = self.transform(image=image, bboxes=list(bboxes), class_labels=["wheat_head"]*len(bboxes)) 
        image_tr = transformed["image"]/255.0
        bboxes = transformed["bboxes"]
        
        if len(bboxes) > 0:
          bboxes = torch.stack([torch.tensor(item) for item in bboxes])
        else:
          bboxes = torch.zeros((0,4))
          
                      
        return image_tr, bboxes, domain, image
              
    def decodeString(self,BoxesString):
      """
      Small method to decode the BoxesString
      """
      if BoxesString == "no_box":
          return np.zeros((0,4))
      else:
          try:
              boxes =  np.array([np.array([int(i) for i in box.split(" ")])
                              for box in BoxesString.split(";")])
              return boxes.astype(np.int32).clip(min=0)
          except:
              print(BoxesString)
              print("Submission is not well formatted. empty boxes will be returned")
              return np.zeros((0,4))


seed_everything(25081992)


#SIZE = 512
train_transform = A.Compose(
        [
        #A.HorizontalFlip(p=0.5),
        #A.VerticalFlip(p=0.5),
        #A.Transpose(p=0.5),
        #A.RandomRotate90(p=0.5),
        #A.RandomRotate90(A.RandomRotate90(p=1.0), p=0.5),
        #A.RandomRotate90(A.RandomRotate90(A.RandomRotate90(p=1.0), p=1.0), p=0.5),
        ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels'],min_area=20)
    )


valid_transform = A.Compose([
    ToTensorV2(p=1.0),
],p=1.0,bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels'],min_area=20))



def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

    """

    images = list()
    targets=list()
    orig_img = list()
    domain_labels = list()
    for i, t, d, io in batch:
        images.append(i)
        targets.append(t)
        orig_img.append(io)
        domain_labels.append(d)
    images = torch.stack(images, dim=0)

    return images, targets, domain_labels, orig_img


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


class _ImageDA(nn.Module):
    def __init__(self,num_domains):
        super(_ImageDA,self).__init__()
        self.num_domains = num_domains
        self.Conv1 = nn.Conv2d(2048, 2048, kernel_size=3, stride=4)
        self.Conv2 = nn.Conv2d(2048, 2048, kernel_size=3, stride=4)
        self.maxpool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(2048, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, self.num_domains)
        self.reLu=nn.ReLU(inplace=False)
        
    def forward(self,x):
        x=grad_reverse(x)
        x=self.reLu(self.Conv1(x))
        x=self.reLu(self.Conv2(x))
        x=self.reLu(self.maxpool(x))
        x=self.flatten(x)
        x=self.reLu(self.linear1(x))
        x=self.reLu(self.linear2(x))
        x=torch.sigmoid(self.linear3(x))
        
        return x
   
class _InstanceDA(nn.Module):
    def __init__(self, num_domains):
        super(_InstanceDA,self).__init__()
        self.num_domains = num_domains
        self.dc_ip1 = nn.Linear(256, 128)
        self.dc_relu1 = nn.ReLU()

        self.classifer=nn.Linear(128,self.num_domains)
        

    def forward(self,x):
        x=grad_reverse(x)
        x=self.dc_relu1(self.dc_ip1(x))
        x=torch.sigmoid(self.classifer(x))

        return x

class _InsClsPrime(nn.Module):
    def __init__(self, num_cls):
        super(_InsClsPrime,self).__init__()
        self.num_cls = num_cls
        self.dc_ip1 = nn.Linear(256, 128)
        self.dc_relu1 = nn.ReLU()
        #self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(128, 64)
        self.dc_relu2 = nn.ReLU()
        #self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer=nn.Linear(64,self.num_cls)
        

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
        self.dc_ip1 = nn.Linear(256, 128)
        self.dc_relu1 = nn.ReLU()
        #self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(128, 64)
        self.dc_relu2 = nn.ReLU()
        #self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer=nn.Linear(64,self.num_cls)
        

    def forward(self,x):
        x=self.dc_relu1(self.dc_ip1(x))
        x=self.dc_ip2(x)
        x=torch.sigmoid(self.classifer(x))

        return x

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    #if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        #_log_api_usage_once(sigmoid_focal_loss)
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss                    


import fcos
class myDAFasterRCNN(LightningModule):
    def __init__(self, n_classes, batchsize, n_tdomains, n_vdomains):
        super(myDAFasterRCNN, self).__init__()
        self.n_classes = n_classes
        self.n_tdomains = n_tdomains
        self.n_vdomains = n_vdomains
        self.batchsize = batchsize
        
        self.detector = fcos.fcos_resnet50_fpn(min_size=1024, max_size=1024, num_classes=self.n_classes, pretrained_backbone=True)
        #self.detector = torchvision.models.detection.fcos_resnet50_fpn(min_size=1024, max_size=1024, num_classes=self.n_classes, pretrained_backbone=True)
	
        self.ImageDA = _ImageDA(self.n_tdomains)
        self.InsDA = _InstanceDA(self.n_tdomains)       
        self.InsCls = nn.ModuleList([_InsCls(self.n_classes) for i in range(self.n_tdomains)])
        self.InsClsPrime = nn.ModuleList([_InsClsPrime(self.n_classes) for i in range(self.n_tdomains)])
        
        self.best_val_acc = 0
        self.val_acc_stack = [[] for i in range(self.n_vdomains)]
        self.freq = torch.tensor(np.zeros(n_classes))
        self.log('val_loss', 100000)
        self.log('val_acc', self.best_val_acc)

        self.base_lr = 1e-5 #Original base lr is 1e-4
        self.momentum = 0.9
        self.weight_decay=0.0001
        
        self.detector.backbone.body.register_forward_hook(self.store_backbone_out)        
        self.detector.head.register_forward_hook(self.store_head_input)  #For instance level features
        
        self.mode = 0
        self.sub_mode = 0
    
      
    def store_backbone_out(self, module, input1, output):
      self.base_feat = output['2']  #Output is a dict at three levels. We consider the top level feature as representative of image level feature
    
    def store_head_input(self, module, input1, output):
      
      temp0 = torch.reshape(input1[0][0], (input1[0][0].shape[0], input1[0][0].shape[1], input1[0][0].shape[2]*input1[0][0].shape[3]))
      temp1 = torch.reshape(input1[0][1], (input1[0][1].shape[0], input1[0][1].shape[1], input1[0][1].shape[2]*input1[0][1].shape[3]))
      temp2 = torch.reshape(input1[0][2], (input1[0][2].shape[0], input1[0][2].shape[1], input1[0][2].shape[2]*input1[0][2].shape[3]))
      temp3 = torch.reshape(input1[0][3], (input1[0][3].shape[0], input1[0][3].shape[1], input1[0][3].shape[2]*input1[0][3].shape[3]))
      temp4 = torch.reshape(input1[0][4], (input1[0][4].shape[0], input1[0][4].shape[1], input1[0][4].shape[2]*input1[0][4].shape[3]))
      
      self.ins_feat = torch.cat((temp0, temp1, temp2, temp3, temp4), -1).permute(0, 2, 1)
      
      
    def forward(self, imgs,targets=None):
      # Torchvision FasterRCNN returns the loss during training 
      # and the boxes during eval
      self.detector.eval()
      return self.detector(imgs)
    
    def configure_optimizers(self):
      
      optimizer = torch.optim.Adam([{'params': self.detector.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay},
                                    {'params': self.ImageDA.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay },
                                    {'params': self.InsDA.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay },
                                    {'params': self.InsCls.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay },
                                    {'params': self.InsClsPrime.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay}
                                      ],) 
      
      lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, threshold=0.0001, min_lr=0, eps=1e-08),
                      'monitor': 'val_loss'}
      
      
      return [optimizer], [lr_scheduler]
      
    def train_dataloader(self):
      num_train_sample_batches = len(tr_dataset)//self.batchsize 
      temp_indices = np.array([i for i in range(len(tr_dataset))])
      np.random.shuffle(temp_indices)
      sample_indices = []
      for i in range(num_train_sample_batches):
  
        batch = temp_indices[self.batchsize*i:self.batchsize*(i+1)]
  
        for index in batch:
          sample_indices.append(index)  
  
        for index in batch:		 
          sample_indices.append(index)

      return torch.utils.data.DataLoader(tr_dataset, batch_size=self.batchsize, sampler=sample_indices, shuffle=False, collate_fn=collate_fn, num_workers=32)
      
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
      
      
      
      if(self.mode == 0):
        loss_dict = self.detector(imgs, targets)
        
        loss = loss_dict['classification'] + loss_dict['bbox_regression'] + loss_dict['bbox_ctrness'] #sum(loss for loss in loss_dict.values())
        
        #print(loss_dict['gt_classes'].shape)
        #print(self.ins_feat.shape)
        
        
        if(self.sub_mode == 0):
          self.mode = 1
          self.sub_mode = 1
        elif(self.sub_mode == 1):
          self.mode = 2
          self.sub_mode = 2
        elif(self.sub_mode == 2):
          self.mode = 3
          self.sub_mode = 3
        elif(self.sub_mode == 3):
          self.mode = 4
          self.sub_mode = 4  
        else:
          self.sub_mode = 0
          self.mode = 0
	
      elif(self.mode == 1):
        
        loss_dict = {}
        
        _ = self.detector(imgs, targets)
        
        ImgDA_scores  = self.ImageDA(self.base_feat)
        #loss_dict['DA_img_loss'] = F.cross_entropy(ImgDA_scores, torch.tensor(batch[2]).cuda().long())
        one_hot_labels = torch.zeros(ImgDA_scores.shape)
        for index in range(len(imgs)):
          one_hot_labels[index, batch[2][index]] = 1
        
        #loss_dict['DA_img_loss'] = sigmoid_focal_loss(ImgDA_scores, torch.tensor(batch[2]).cuda().long(), reduction="mean")
        loss_dict['DA_img_loss'] = 0.05*sigmoid_focal_loss(ImgDA_scores, one_hot_labels.cuda(), reduction="mean")
        
        IDA_out = self.InsDA(self.ins_feat)
        #loss_dict['DA_ins_loss'] = F.cross_entropy(IDA_out.permute(0,2,1), torch.tensor(batch[2]).repeat(IDA_out.shape[1],1).permute(1,0).cuda())
        inp = IDA_out.permute(0, 2, 1)
        one_hot_labels = torch.zeros(inp.shape)
        for index in range(len(imgs)):
          one_hot_labels[index, batch[2][index], :] = 1
        
        loss_dict['DA_ins_loss'] = 0.5*sigmoid_focal_loss(inp, one_hot_labels.cuda(), reduction="mean")
             
        loss_dict['Cst_loss'] = F.mse_loss(IDA_out, ImgDA_scores.unsqueeze(dim=-1).repeat(1, 1, IDA_out.shape[1]).permute(0, 2, 1))
        loss = sum(loss for loss in loss_dict.values())  
        
        self.mode = 0
    
        
      elif(self.mode == 2): #Without recording the gradients for detector, we need to update the weights for classifier weights
        loss_dict = {}
        loss = []

        
        for index in range(len(self.InsCls)):
          for param in self.InsCls[index].parameters(): param.requires_grad = True

        for index in range(len(imgs)):
          with torch.no_grad():
            temp_dict = self.detector([imgs[index]], [targets[index]])
          cls_logits = self.InsCls[batch[2][index].item()](self.ins_feat)
          loss.append(sigmoid_focal_loss(cls_logits, temp_dict['gt_classes'], reduction="mean")) 

        loss_dict['cls'] = 0.05*(torch.mean(torch.stack(loss)))
        loss = sum(loss for loss in loss_dict.values())

        self.mode = 0
        
      elif(self.mode == 3): #Only the GRL Classification should influence the updates but here we need to update the detector weights as well
        loss_dict = {}
        loss = []
    
        for index in range(len(imgs)):
          temp_dict = self.detector([imgs[index]], [targets[index]])
          cls_logits = self.InsClsPrime[batch[2][index].item()](self.ins_feat)
          loss.append(sigmoid_focal_loss(cls_logits, temp_dict['gt_classes'], reduction="mean")) 
  	  
        loss_dict['cls_prime'] = 0.5*(torch.mean(torch.stack(loss)))
        loss = sum(loss for loss in loss_dict.values())

        self.mode = 0
        
      else: #For Mode 4
      
        loss_dict = {}
        loss = []
        consis_loss = []
        
        for index in range(len(self.InsCls)):
          for param in self.InsCls[index].parameters(): param.requires_grad = False
        
        for index in range(len(imgs)):
          temp_dict = self.detector([imgs[index]], [targets[index]])
          temp = []
          for i in range(len(self.InsCls)):
            if(i != batch[2][index].item()):
              cls_logits = self.InsCls[i](self.ins_feat)
              #temp.append(cls_logits)
              loss.append(sigmoid_focal_loss(cls_logits, temp_dict['gt_classes'], reduction="mean")) 
          #consis_loss.append(torch.mean(torch.abs(torch.stack(temp, dim=0) - torch.mean(torch.stack(temp, dim=0), dim=0))))

        loss_dict['cls'] = 0.001*(torch.mean(torch.stack(loss)))# + torch.mean(torch.stack(consis_loss)))
        loss = sum(loss for loss in loss_dict.values())
        
        self.mode = 0
        self.sub_mode = 0
 	 
      return {"loss": loss}#, "log": torch.stack(temp_loss).detach().cpu()}

    def validation_step(self, batch, batch_idx):
      img, boxes, domain, _ = batch
      
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
          
      temp = temp/non_zero_domains #8 Validation domains 
      self.log('val_loss', 1 - temp)  #Logging for model checkpoint
      self.log('val_acc', temp)
      if(self.best_val_acc < temp):
        self.best_val_acc = temp
        self.best_val_acc_epoch = self.trainer.current_epoch
      

      self.val_acc_stack = [[] for i in range(self.n_vdomains)]
      
      print('Validation ADA : ',temp)
      self.mode = 0
      
      
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

            
        return  true_positive / (true_positive + false_positive + false_negative) 

      elif total_gt == 0:
          if total_pred > 0:
              return torch.tensor(0.).cuda()
          else:
              return torch.tensor(1.).cuda()
      elif total_gt > 0 and total_pred == 0:
          return torch.tensor(0.).cuda()
          
tr_dataset = WheatDataset('../datasets/gwhd_2021/official_train.csv', root_dir='../datasets/gwhd_2021/images/', image_set = 'train', transform=train_transform)
vl_dataset = WheatDataset('../datasets/gwhd_2021/official_val.csv', root_dir='../datasets/gwhd_2021/images/', image_set = 'val', transform=valid_transform)
val_dataloader = torch.utils.data.DataLoader(vl_dataset, batch_size=1, shuffle=False,  collate_fn=collate_fn, num_workers=4)

           
import os
detector = myDAFasterRCNN(n_classes=2, batchsize=2, n_tdomains=tr_dataset.num_domains, n_vdomains=vl_dataset.num_domains)


NET_FOLDER = 'GWHD_FCOS'
weights_file = 'best_baseline_focalloss_final' #(v1--> (0.5, 1, 1)) v2--> (0.1, 1, 1), (0.05, 1, 1), (0.001, 1, 1)
if(os.path.exists(NET_FOLDER+'/'+weights_file+'.ckpt')):
  detector.load_state_dict(torch.load(NET_FOLDER+'/'+weights_file+'.ckpt')['state_dict'])
else:	
  if not os.path.exists(NET_FOLDER):
    mode = 0o777
    os.mkdir(NET_FOLDER, mode)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
early_stop_callback= EarlyStopping(monitor='val_acc', min_delta=0.00, patience=10, verbose=False, mode='max')


checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=NET_FOLDER, filename=weights_file)
trainer = Trainer(accelerator='gpu', devices=1, max_epochs=100, deterministic=False, callbacks=[checkpoint_callback, early_stop_callback], reload_dataloaders_every_n_epochs=1)
#trainer.fit(detector, val_dataloaders=val_dataloader)


detector.load_state_dict(torch.load(NET_FOLDER+'/'+weights_file+'.ckpt')['state_dict'])
detector.freeze()
test_dataset = WheatDataset('../datasets/gwhd_2021/official_test.csv', root_dir='../datasets/gwhd_2021/images/', image_set = 'test', transform=valid_transform)

detector.detector.eval()
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
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

            
        return  true_positive / ( true_positive + false_positive + false_negative )

      elif total_gt == 0:
          if total_pred > 0:
              return torch.tensor(0.).cuda()
          else:
              return torch.tensor(1.).cuda()
      elif total_gt > 0 and total_pred == 0:
          return torch.tensor(0.).cuda()
          

detector.to('cuda')
val_acc_stack = [[] for i in range(test_dataset.num_domains)]
domain = torch.zeros(test_dataset.num_domains)
for index, data_sample in enumerate(iter(test_dataloader)):

  images, boxes, labels, orig_img = data_sample
   
  preds = detector(images.cuda())
  
  preds[0]['boxes'] = preds[0]["boxes"].detach().cpu()
  preds[0]['labels'] = preds[0]["labels"].detach().cpu()
  preds[0]['scores'] = preds[0]["scores"].detach().cpu()
  
  preds[0]['boxes'] = preds[0]['boxes'][preds[0]['scores'] > 0.5]
  val_acc_stack[labels[0]].append(torch.stack([acc_new(b,pb["boxes"],iou_threshold=0.5) for b,pb in zip(boxes,preds)]).detach().cpu())
      
  domain[labels[0]] = domain[labels[0]] + 1
  
  for box in preds[0]['boxes']:
    cv2.rectangle(orig_img[0], (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 3)
  
  for box in boxes[0]:
    cv2.rectangle(orig_img[0], (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 3)
  
  path = './baseline_GWHD/'+str(labels[0].item())+'/'+str(index)+'.png'
  cv2.imwrite(path, cv2.cvtColor(orig_img[0], cv2.COLOR_RGB2BGR))
  domain[labels[0]] = domain[labels[0]] + 1
  #print(index)
  
      
#ADA_whole = torch.mean(torch.stack(val_acc_stack))

#print(ADA_whole)
weights = [1/domain[i] for i in range(test_dataset.num_domains)]
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
print('WAD:', torch.mean(torch.tensor(test_acc)))

