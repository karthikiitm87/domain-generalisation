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

    
mypath = 'VOC2012'

imagefiles = [f for f in listdir(mypath+'/JPEGImages/') if isfile(join(mypath+'/JPEGImages/', f))]
xmlfiles = [f for f in listdir(mypath+'/Annotations/') if isfile(join(mypath+'/Annotations/', f))]

#Only car is needed for sim10k as that is the only object used for demonstrating the cross dataset generalisability. 
#obj_freq = {'car':0, 'motorbike':0, 'person':0} 
annotations = []
for image in imagefiles:

    tree = ET.parse(mypath+'/Annotations/'+image.split('.')[0]+'.xml')
    root = tree.getroot()
    
    bbox = []
    
    for obj in root.findall('object'):
      if(obj.find('name').text == 'car'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        bbox.append([xmin, ymin, xmax, ymax])
    annotations.append([image, encode_boxes(bbox)])
  
  
results = pd.DataFrame(annotations,columns =["image_name","BoxesString"])


#There is no seperate test file and hence we are generating this sim10k test split 
train, test = train_test_split(results, test_size=0.2) 
train.to_csv('sim10k_train_car.csv', index = False)
test.to_csv('sim10k_test_car.csv', index = False)


