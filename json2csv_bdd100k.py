import json
import cv2
import numpy as np
from os import walk
import pandas as pd
import random
	
image_set = 'train' #Change to 'val' to generate csv file for validation set
#Here we extract the annotation for car object as it is the only common object for demonstrating cross dataset generalisation
categories = {'car': 0} 

def encode_boxes(boxes):

  if len(boxes) >0:
    boxes = [" ".join([str(float(i)) for i in item]) for item in boxes]
    BoxesString = ";".join(boxes)
  else:
    BoxesString = "no_box"
  return BoxesString

df = pd.DataFrame(columns=['image_name', 'BoxesString'])

f = open('../datasets/bdd100k/labels/det_20/det_'+image_set+'.json') #Path for the annotation file
data = json.load(f)

df = pd.DataFrame(columns=['image_name', 'BoxesString'])

for item in data:
  img_name = item['name']
  
  bboxes = []
  if('labels' in item.keys()):
    for obj in item['labels']:
      if(obj['category'] == 'car'):
        bboxes.append([obj['box2d']['x1'], obj['box2d']['y1'], obj['box2d']['x2'], obj['box2d']['y2']])
        
  BoxesString = encode_boxes(bboxes)
  
  new_row = {'image_name':img_name, 'BoxesString': BoxesString}
  
  df = df.append(new_row, ignore_index=True)

df.to_csv('./Annots/bdd100k_'+image_set+'_car.csv')
print(df.head())


