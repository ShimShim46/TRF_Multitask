#! /bin/bash
DIR=/mnt/WD_Blue/Multitask_master/Corpus/ACL/5test/20news
# DIR=/mnt/WD_Blue/Multitask_master/Corpus/ACL/data_drop/RCV1
trainData=${DIR}/20news_train.xml
# trainData=${DIR}/rcv1_5_train.xml


## hyper-params ##
epoch=100
batchSize=32
gpu=0
shuffle=yes
model=XML-CNN

testData=${DIR}/20news_test1.xml
# testData=${DIR}/rcv1_5_train.xml

FP=RESULT_${model}
mkdir -p ${FP}

python program/train.py -itrain ${trainData}  -itest ${testData} -m ${model}  \
 -e ${epoch}  -b ${batchSize} -g ${gpu} --filepath ${FP} \
--shuffle ${shuffle} --pretrained 0 --multilabel 0

