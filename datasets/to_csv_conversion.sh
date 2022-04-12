#!/bin/bash

python xml2csv_sim10k.py
echo Sim10k annotations are converted.
python json2csv_bdd100k_full.py
echo BDD100K annotations are converted
python json2csv_cityscapes.py --category car
echo Cityscapes car annotations are converted
python json2csv_cityscapes.py --category all
echo Cityscapes full annotations are converted
python json2csv_cityscapes_refined.py --weather foggy
echo Cityscapes foggy refined annotations are converted
python json2csv_cityscapes_refined.py --weather rain
echo Cityscapes rain refined annotations are converted
