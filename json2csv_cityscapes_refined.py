#This file has the helper code used to convert all the annotations into csv format for refined cityscapes
#Not all images of cityscapes are used in refined cityscapes

import json
import cv2
import numpy as np
from os import walk
import pandas as pd
import random
import os, fnmatch


weather = 'rain' #Can be rain or foggy based on the requirement

def find_all(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def encode_boxes(boxes):

  if len(boxes) >0:
    boxes = [" ".join([str(int(i)) for i in item]) for item in boxes]
    BoxesString = ";".join(boxes)
  else:
    BoxesString = "no_box"
  return BoxesString

def encode_labels(labels):

  if len(labels) >0:
    labels = [" ".join([str(int(item))]) for item in labels]
    LabelsString = ";".join(labels)
  else:
    LabelsString = "no_label"
  return LabelsString
  
train_df = pd.DataFrame(columns=['image_name', 'BoxesString', 'LabelsString'])  
val_df = pd.DataFrame(columns=['image_name', 'BoxesString', 'LabelsString'])

train_obj_freq = {'person':0, 'rider': 0,'car': 0,'truck': 0, 'bus':0, 'train':0, 'motorcycle':0, 'bicycle':0}
val_obj_freq = {'person':0, 'rider': 0,'car': 0,'truck': 0, 'bus':0, 'train':0, 'motorcycle':0, 'bicycle':0}
    	
#Have only car object when a csv file contaning car object is needed. Cross dataset generalisation of our code use only car. 
#categories = {'car': 1}
categories = {'person':1, 'rider': 2,'car': 3,'truck': 4, 'bus':5, 'train':6, 'motorcycle':7, 'bicycle':8}

df = pd.read_csv(weather+'_trainval_refined_filenames.txt', names = ['image_name'])

for index in range(len(df)):
  [image_set, city, image_name] = df['image_name'][index].split('/')
  path = 'leftImg8bit_'+weather+'/'+image_set+'/'+city+'/'
  result = find_all(image_name+'*', path)  #Has all the file names matching the string 
  
  for item in result:
    [_, image_set, city, image_name] =  item.split('/')
    temp = image_name.split('_')
    if(weather == 'foggy'):
      if(temp[-1] != '0.02.png'):
        continue
      
    json_path = './gtFine/'+image_set+'/'+city+'/'+city+'_'+temp[1]+'_'+temp[2]+'_gtFine_polygons.json'
    f = open(json_path)
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
        labels.append(categories[item['label']])
        if(image_set == 'train'):
          train_obj_freq[item['label']] = train_obj_freq[item['label']] + 1
        elif(image_set == 'val'):
          val_obj_freq[item['label']] = val_obj_freq[item['label']] + 1
        else:
          continue
          
    BoxesString = encode_boxes(bboxes)
    LabelsString = encode_labels(labels)
    image_path = city + '/' + image_name
    new_row = {'image_name':image_path, 'BoxesString': BoxesString, 'LabelsString': LabelsString}  
    
    if(image_set == 'train'):
      train_df = train_df.append(new_row, ignore_index=True)
    elif(image_set == 'val'):
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

