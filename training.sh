#! /bin/bash
DIR=/mnt/WD_Blue/Multitask_master/Corpus/ACL/5test/20news
trainData=${DIR}/20news_train.xml
testData=${DIR}/20news_test1.xml

## hyper-params ##
epoch=100
batchSize=32
gpu=0
shuffle=yes
pretrained=0
multilabel=0
model=XML-CNN ## XML-CNN, TRF-Single, TRF-Multi, TRF-Delay-Multi, or TRF-Sequential ##




FP=RESULT_${model}
mkdir -p ${FP}

python program/train.py -itrain ${trainData}  -itest ${testData} -m ${model}  \
 -e ${epoch}  -b ${batchSize} -g ${gpu} --filepath ${FP} \
--shuffle ${shuffle} --pretrained ${pretrained} --multilabel ${multilabel}

