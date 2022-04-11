import json
import cv2
import numpy as np
from os import walk
import pandas as pd
import random
	
#BDD100K dataset is used for demonstrating the cross dataset generalisation
# and hence we extract the annotation corresponding to car alone
categories = {'car': 0}  
sets = ['train', 'val']

def encode_boxes(boxes):

  if len(boxes) >0:
    boxes = [" ".join([str(float(i)) for i in item]) for item in boxes]
    BoxesString = ";".join(boxes)
  else:
    BoxesString = "no_box"
  return BoxesString


for image_set in sets:
  df = pd.DataFrame(columns=['image_name', 'BoxesString'])

  f = open('./bdd100k/labels/det_20/det_'+image_set+'.json')
  data = json.load(f)



  for item in data:
    img_name = item['name']
  
    bboxes = []
    if('labels' in item.keys()):
      for obj in item['labels']:
        if(obj['category'] in categories.keys()):
          bboxes.append([obj['box2d']['x1'], obj['box2d']['y1'], obj['box2d']['x2'], obj['box2d']['y2']])
        
    BoxesString = encode_boxes(bboxes)
  
    new_row = {'image_name':img_name, 'BoxesString': BoxesString}
    df.append(new_row, ignore_index=True)

  df = df.reset_index(drop=True)
  df.to_csv('./Annots/bdd100k_car_'+image_set+'.csv')
  f.close()

