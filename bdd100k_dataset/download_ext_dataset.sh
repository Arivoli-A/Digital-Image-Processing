#!/bin/bash

wget https://dl.cv.ethz.ch/bdd100k/data/10k_images_test.zip  
# wget https://dl.cv.ethz.ch/bdd100k/data/10k_images_train.zip  
# wget https://dl.cv.ethz.ch/bdd100k/data/10k_images_val.zip  

mkdir -p ./bdd100k_images_10k_test
# mkdir -p ./bdd100k_images_10k_train
# mkdir -p ./bdd100k_images_10k_val

unzip 10k_images_test.zip -d ./bdd100k_images_10k_test
# unzip 10k_images_train.zip -d ./bdd100k_images_10k_train
# unzip 10k_images_val.zip -d ./bdd100k_images_10k_val