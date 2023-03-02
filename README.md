# Domain generalisation for Object Detection

This repository is the collection of all the works done for the project on '[Domain Generalisation for Object Detection']. Here three popular object detectors (Faster R-CNN, FCOS, YoloV3) are explored and the codes related to each of the detector are present in their respective folders. 

Till date, the work done under this project has been published in two premier conference publications. 
1. K. Seemakurthy, C. Fox, E. Aptoula, and P. Bosilj, "Domain Generalised Faster R-CNN", AAAI 2023 (Codes present in DGFRCNN folder)
2. K. Seemakurthy, P.Bosilj, E. Aptoula, and C. Fox, "Domain Generalised Fully Convolutional One Stage Object Detection", ICRA 2023 (Codes present in DGFCOS folder)
The links for camera ready publications will be soon updated in this page. 

However, the first version of ArXiv paper can be found at '[Domain Generalisation for Object Detection'](https://arxiv.org/abs/2203.05294).  

We used the following datasets for our experiments. 

1. [Global Wheat Head Detection (GWHD)](https://zenodo.org/record/5092309#.YjeR1zynzJU) 
2. [Cityscapes](https://www.cityscapes-dataset.com/) 
3. [Sim10K](https://fcav.engin.umich.edu/projects/driving-in-the-matrix) 
4. [Berkely Deep Drive 100K (BDD100K)](https://bdd-data.berkeley.edu/) 

Our code expects the input data format to be in csv format and we provide necessary helper functions to convert the annotations from JSON (Cityscapes, BDD100K), XML (Sim10K) formats into csv format for all the datasets. But we expect the users to download the datasets from respective websites and follow the file structure mentioned below so that the code can directly access the datasets. The file requirements.txt has all the dependencies needed to run this code. 

```
./downloads.sh  #GWHD and Sim10K can be downloaded in 'datasets' folder by execuitng this command 
```

BDD100K, Cityscapes, Foggy-cityscapes, Rain-Cityscapes need to be downloaded manually from their respective websites and arrange them in the following directory structure. 

# Directory structure for datasets

```
.
├── datasets
│   ├── bdd100k
    |     ├── images
    |     |     ├── 100k
    |     |           ├── train
    |     |           ├── val
    |     |           ├── test
    |     ├── labels
    |     |     ├── det20
    |     |           ├── det_train.json
    |     |           ├── det_val.json
    ├── sim10k
    |     ├── VOC2012
    |     |       ├── Annotations
    |     |       ├── JPEGImages
    ├── cityscapes_clear
    |     ├── train
    |     ├── val
    |     ├── test
    ├── cityscapes_foggy
    |     ├── train
    |     ├── val
    |     ├── test
    ├── cityscapes_rain
    |     ├── train
    |     ├── val
    |     ├── test
    ├── gwhd_2021
    |     ├── images
    ├── downloads.sh
    ├── to_csv_conversion.sh
    ├── Annots
```

Once above directory structure is ensured, the following command needs to be executed to convert all the annotations into csv format and place them in Annots as needed by our code. 

```
./to_csv_conversion.sh
```

The above command will generate the following 12 csv files in Annots folder where a subset of them will be used by the detector during training. 

```
1. bdd100k_car_train.csv
2. bdd100k_car_val.csv
3. cityscapes_clear_all_train.csv
4. cityscapes_clear_all_val.csv
5. cityscapes_clear_car_train.csv
6. cityscapes_clear_car_val.csv
7. cityscapes_foggy_train.csv
8. cityscapes_foggy_val.csv
9. cityscapes_rain_train.csv
10. cityscapes_rain_val.csv
11. sim10k_train_car.csv
12. sim10k_val_car.csv
```


# Using Faster-RCNN detector

In this code, we need to train additional domain specific classifiers for which we need the access to ground truth labels of each identified region proposal. We have made minor changes to the Faster-RCNN implementation in [WilDS](https://github.com/p-lambda/wilds/tree/main/examples/models/detection) to obtain the ground truth labels of each region proposal. We initialise our ResNet backbone with COCO pretrained weights for experimenting with Cityscapes, Sim10K, BDD100K while we obtained better accuracy for GWHD using ImageNet pretrained weights. We use the Pytorch-Lightning framework to train our model. 

The following command can be executed to replicate the results in Table 1. Appropriate lines of train_Cityscapes.py needs to be uncommented based on which results in  Table 1 can be replicated. 
```
python DGFRCNN/train_cityscapes.py
```

The following commands can be executed to replicate the results in Table 2.  Appropriate lines in train_BSC.py needs to be uncommented based on the column that is of interest for replication. 
```
python DGFRCNN/train_GWHD.py
python DGFRCNN/train_BSC.py
```

The following command can be executed to replicate the results in Table 3.

```
python DGFRCNN/train_DA.py
```

