#!/bin/bash

wget http://droplab-files.engin.umich.edu/repro_10k_images.tgz
wget http://droplab-files.engin.umich.edu/repro_10k_annotations.tgz
mkdir sim10k
scp *.tgz ./sim10k/
cd sim10k
tar -zxvf repro_10k_images.tgz
tar -zxvf repro_10k_annotations.tgz
cd ..
wget https://worksheets.codalab.org/rest/bundles/0x36e16907b7254571b708b725f8beae52/contents/blob/ -O gwhd_2021.tar.gz
mkdir gwhd_2021
scp gwhd_2021.tar.gz ./gwhd_2021/
cd gwhd_2021
tar -zxvf gwhd_2021.tar.gz
scp ./official_train.csv ../Annots/
scp ./official_val.csv ../Annots/
scp ./official_test.csv ../Annots/
