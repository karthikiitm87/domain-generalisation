#This file has the code necessary for converting the XML annotations to CSV format
import pandas as pd
from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

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
    
mypath = './sim10k/VOC2012'

imagefiles = [f for f in listdir(mypath+'/JPEGImages/') if isfile(join(mypath+'/JPEGImages/', f))]
xmlfiles = [f for f in listdir(mypath+'/Annotations/') if isfile(join(mypath+'/Annotations/', f))]

#Only car is needed for sim10k as that is the only object used for demonstrating the cross dataset generalisability. 
#obj_freq = {'car':0, 'motorbike':0, 'person':0} 
categories = {'car':0}  #by inlcuding the keys of other objects we can include their annotations as well.
annotations = []
for image in imagefiles:

    tree = ET.parse(mypath+'/Annotations/'+image.split('.')[0]+'.xml')
    root = tree.getroot()
    
    bbox = []
    labels = []
    
    for obj in root.findall('object'):
      if(obj.find('name').text in categories.keys()):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        bbox.append([xmin, ymin, xmax, ymax])
        labels.append(obj.find('name').text)
        
    annotations.append([image, encode_boxes(bbox), encode_labels(labels)])
  
  
results = pd.DataFrame(annotations,columns =["image_name", "BoxesString", "LabelsString"])


#There is no seperate test file and hence we are generating this sim10k test split 
train, val = train_test_split(results, test_size=0.2) 

train = train.reset_index(drop=True)
val = val.reset_index(drop=True)
  
train.to_csv('./Annots/sim10k_train_car.csv', index = False)
val.to_csv('./Annots/sim10k_val_car.csv', index = False)

#print(train.head())
#print(val.head())


