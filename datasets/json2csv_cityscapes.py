#This file is the helper code used to convert all the annotations into csv format
import json
import cv2
import numpy as np
from os import walk
import pandas as pd
import random
	
#Have only car object when a csv file contaning car object is needed. Cross dataset generalisation of our code use only car. 
#categories = {'car': 1}
categories = {'person':1, 'rider': 2,'car': 3,'truck': 4, 'bus':5, 'train':6, 'motorcycle':7, 'bicycle':8}

def encode_boxes(boxes):

  if len(boxes) >0:
    boxes = [" ".join([str(int(i)) for i in item]) for item in boxes]
    BoxesString = ";".join(boxes)
  else:
    BoxesString = "no_box"
  return BoxesString

def encode_labels(labels):

  if len(labels) >0:
    labels = [" ".join([str(item)]) for item in labels]
    LabelsString = ";".join(labels)
  else:
    LabelsString = "no_label"
  return LabelsString
  
train_df = pd.DataFrame(columns=['image_name', 'BoxesString', 'LabelsString'])  
val_df = pd.DataFrame(columns=['image_name', 'BoxesString', 'LabelsString'])

train_obj_freq = {'person':0, 'rider': 0,'car': 0,'truck': 0, 'bus':0, 'train':0, 'motorcycle':0, 'bicycle':0}
val_obj_freq = {'person':0, 'rider': 0,'car': 0,'truck': 0, 'bus':0, 'train':0, 'motorcycle':0, 'bicycle':0}

#weather parameter can take one of the {'foggy', 'rain', 'clear'}  
weather = 'clear' 

for (dirpath, dirnames, filenames) in walk('./leftImg8bit_'+weather): 
  split = dirpath.split('/')
  if(len(split) == 4):
    for f in filenames:
    
      sub_split = f.split('/')
      imagename_split = sub_split[-1].split('_')
      
      imagename = imagename_split[0] + '_' + imagename_split[1] + '_' + imagename_split[2]  

      json_name = './gtFine/'+split[2]+'/'+imagename_split[0]+'/'+imagename+'_gtFine_polygons.json'
      f = open(json_name)
      data = json.load(f)
      bboxes = []
      labels = []
      for item in data['objects']:
        if(item['label'] in categories.keys()):
            
          polygon_coods = np.array(item['polygon'])
          x_min = np.min(polygon_coods[:,0])
          x_max = np.max(polygon_coods[:,0])
          y_min = np.min(polygon_coods[:,1])
          y_max = np.max(polygon_coods[:,1])
              
          bboxes.append([x_min, y_min, x_max, y_max])
          #labels.append(categories[item['label']])
          labels.append(item['label'])
          if(split[2] == 'train'):
            train_obj_freq[item['label']] = train_obj_freq[item['label']] + 1
          elif(split[2] == 'val'):
            val_obj_freq[item['label']] = val_obj_freq[item['label']] + 1
          else:
            continue
          
      BoxesString = encode_boxes(bboxes)
      LabelsString = encode_labels(labels)
      image_path = imagename_split[0] + '/' + sub_split[-1]
      
      new_row = {'image_name':image_path, 'BoxesString': BoxesString, 'LabelsString': LabelsString}  

      if(weather == 'foggy'):
        if(imagename_split[6] == '0.02.png'):
          if(split[2] == 'train'):
            train_df = train_df.append(new_row, ignore_index=True)
          elif(split[2] == 'val'):
            val_df = val_df.append(new_row, ignore_index=True)
          else:
            continue
      else:
        if(split[2] == 'train'):
          train_df = train_df.append(new_row, ignore_index=True)
        elif(split[2] == 'val'):
          val_df = val_df.append(new_row, ignore_index=True)
        else:
          continue
        
          
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

train_df.to_csv('./Annots/cityscapes_'+weather+'_train.csv')  
val_df.to_csv('./Annots/cityscapes_'+weather+'_val.csv')

print(train_obj_freq)
print(val_obj_freq)
print(train_df.head())
print(val_df.head())
