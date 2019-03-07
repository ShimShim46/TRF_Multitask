#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import pdb
import random
import sys
from itertools import chain

import chainer
import chainer.computational_graph as c
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O
import chainer.serializers as S
import cupy as cp

import numpy as np
import optuna
import six
from chainer import cuda

from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from xmlcnn import XMLCnn
from net import Transformer
from sentence_reader import (SentenceReaderDir, convert_numeric_data, load_xml)

chainer.config.cudnn_deterministic = True
chainer.config.use_cudnn = 'never'
MAX_SEN_LEN = 100




def estimate_test(test_data, reader, epoch, path, model_type, model, is_multi_label):
    ## 推論 ##
    print ("")
    print ("-"*50)
    print ("Estimate")
    batch_size = args.batchsize


    test_data_size = len(test_data['indexed_text'])
    netouts_txt = [] ## 文書分類の結果
    netouts_wsd = [] ## WSDの結果
    network_output_order = [i[0] for i in sorted(dict(reader.catgy).items(), key=lambda x: x[1])] ## wsdの予測順(ニューロンの順番)

    is_multi_task = 0
    if "Multi" in model_type.split("-"):
        is_multi_task = 1



    for i in tqdm(six.moves.range(0, test_data_size, batch_size)):
        x = test_data['indexed_text'][i : i + batch_size]
        positions = test_data['positions'][i : i + batch_size]
        labels = test_data['labels'][i : i + batch_size]
        keys = test_data['keys'][i : i + batch_size]

        sent = {"indexed_text":x, "positions": positions, "labels":labels, "keys":keys}

        preds_tc = None
        preds_wsd = None

        if is_multi_task:
            preds_tc,preds_wsd = model(sent,None,epoch,True)
        elif "TRF" in model_type.split("-"):
            if model_type == "TRF-Sequential":
                preds, task_type = model(sent,None,epoch,True)
                if task_type == "wsd":
                    preds_wsd = preds
                elif task_type == "tc":
                    preds_tc = preds
            else:
                preds_tc = model(sent,None,epoch,True)
        elif model_type == "XML-CNN":
            preds_tc = model.estimate(sent)
        
        ## 文書分類ラベルの変換 ##
        if not preds_tc is None:
            preds_tc_label_list = [[] for i in range(preds_tc.shape[0])]

            if is_multi_label == 1:
                preds_tc = F.sigmoid(preds_tc)
                indexes, ind_label = np.where((chainer.cuda.to_cpu(preds_tc.data) >= 0.5)) ##しきい値(0.5)を超えたら
                converted_label = np.array(model.le.classes)[ind_label]
                for i,j in zip(indexes, converted_label):
                    preds_tc_label_list[i].append(j)
                netouts_txt.extend(preds_tc_label_list)
            elif is_multi_label == 0:
                preds_tc = F.softmax(preds_tc) ## 文書ラベルの方をsoftmaxで活性化
                indexes = chainer.cuda.to_cpu(F.argmax(preds_tc,axis=1).data).tolist()
                converted_label = model.le.inverse_transform(indexes).tolist()
                netouts_txt.extend(converted_label)

        # WSDのラベル変換 ##
        if not preds_wsd is None:
            unfold_labels = chain(*labels)
            for t,p in zip(unfold_labels, preds_wsd):
                if t == "<PAD>":
                    continue
                else:
                    try:
                        target_indexes = model.senseid2netout[t]
                        values = p[target_indexes]
                        local_index = int(F.argmax(values).data)
                        predict_index = target_indexes[local_index]
                        netouts_wsd.append(network_output_order[predict_index])
                    except:
                        predict_index = int(F.argmax(p).data)
                        netouts_wsd.append(network_output_order[predict_index])

    if not preds_tc is None:

        ## F値の計算 ##
        ## 文書分類 ##
        if is_multi_label == 1:

            bin_pred_tc = model.le.fit_transform(netouts_txt)
            bin_ans_tc = model.le.fit_transform(test_data['doc_category'])
            micro_f1_tc = f1_score(y_pred=bin_pred_tc, y_true=bin_ans_tc, average="micro")
            macro_f1_tc = f1_score(y_pred=bin_pred_tc, y_true=bin_ans_tc, average="macro")
            weighted_f1_tc = f1_score(y_pred=bin_pred_tc,y_true=bin_ans_tc, average="weighted")
            accuracy_tc = accuracy_score(y_true=bin_ans_tc, y_pred=bin_pred_tc)

        elif is_multi_label == 0:
            pred_tc = netouts_txt
            ans_tc = list(chain(*test_data['doc_category']))
            micro_f1_tc = f1_score(y_pred=pred_tc, y_true=ans_tc, average="micro")
            macro_f1_tc = f1_score(y_pred=pred_tc, y_true=ans_tc, average="macro")
            weighted_f1_tc = f1_score(y_pred=pred_tc,y_true=ans_tc, average="weighted")
            accuracy_tc = accuracy_score(y_true=ans_tc, y_pred=pred_tc)

        print ("-"*50)
        print ("**Text Categorization**")
        print ("Micro F1\t{}".format(micro_f1_tc))
        print ("Macro F1\t{}".format(macro_f1_tc))
        print ("Accuracy\t{}".format(accuracy_tc))


        result_file_txt_cat = open(path + "/RESULT_FILE_" + str(epoch) + "EPOCH_TC", mode="w")
        result_file_fscore = open(path + "/RESULT_FILE_" + str(epoch) + "EPOCH_TC_fscore", mode="w")

        result_file_fscore.write("Micro F1\t" + str(micro_f1_tc) + "\n")
        result_file_fscore.write("Macro F1\t" + str(macro_f1_tc) + "\n")
        result_file_fscore.write("Weighted F1\t" + str(weighted_f1_tc) + "\n")
        result_file_fscore.write("Accuracy\t" + str(accuracy_tc) + "\n")
        result_file_fscore.close()

        result_file_txt_cat.write("Ground Truth\tPrediction\n")
        for i,j in zip(test_data["doc_category"], netouts_txt):
            if is_multi_label == 1:
                result_file_txt_cat.write(",".join(i) + "\t" + ",".join(j)+"\n")
            elif is_multi_label == 0:
                result_file_txt_cat.write(i[0] + "\t" + j + "\n")
        result_file_txt_cat.close()

    if not preds_wsd is None:

        ## WSD ##
        ans_wsd = [label for label in chain(*test_data['labels']) if label != "<PAD>"]
        assert len(ans_wsd) == len(netouts_wsd)
        micro_f1_wsd = f1_score(y_pred=netouts_wsd, y_true=ans_wsd ,average="micro")
        macro_f1_wsd = f1_score(y_pred=netouts_wsd, y_true=ans_wsd ,average="macro")
        weighted_f1_wsd = f1_score(y_pred=netouts_wsd,y_true=ans_wsd, average="weighted")
        accuracy_wsd = accuracy_score(y_true=ans_wsd, y_pred=netouts_wsd)
        print ("")
        print ("**Word Sense Disambiguation**")
        print ("Micro F1\t{}".format(micro_f1_wsd))
        print ("Macro F1\t{}".format(macro_f1_wsd))
        print ("Accuracy\t{}".format(accuracy_wsd))

        result_file_fscore = open(path + "/RESULT_FILE_" + str(epoch) + "EPOCH_WSD_fscore", mode="w")
        result_file_wsd = open(path + "/RESULT_FILE_" + str(epoch) + "EPOCH_WSD", mode="w")
        result_file_fscore.write("Micro F1\t" + str(micro_f1_wsd) + "\n")
        result_file_fscore.write("Macro F1\t" + str(macro_f1_wsd) + "\n")
        result_file_fscore.write("Weighted F1\t" + str(weighted_f1_wsd) + "\n")
        result_file_fscore.write("Accuracy\t" + str(accuracy_wsd) + "\n")
        result_file_fscore.close()

        result_file_wsd.write("Grouned Truth\tPrediction\n")
        for i,k in zip(ans_wsd, netouts_wsd):
            result_file_wsd.write(i + "\t" + k + "\n")
        result_file_wsd.close()

    print ("-"*50)
    print("")
    print("-"*50)
    print("Writing out prediction...")



def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--intraindata', '-itrain',
                        default=None,
                        help='input train corpus directory')

    parser.add_argument('--intestdata', '-itest',
                        default=None,
                        help='input test file name')

    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')

    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='learning minibatch size')

    parser.add_argument('--epoch', '-e', default=10, type=int,
                        help='number of epochs to learn')

    parser.add_argument('--model', '-m', choices=['XML-CNN','TRF-Single','TRF-Multi','TRF-Delay-Multi','TRF-Sequential'])

    parser.add_argument('--filepath', '-fp',default=None)

    parser.add_argument('--shuffle',default='no', type=str, choices=['yes','no'])

    parser.add_argument('--pretrained',choices=[0,1,2], type=int)

    parser.add_argument('--multilabel', '-ml', type=int, choices=[0,1])

    args = parser.parse_args()

    if args.shuffle == 'no':
        args.shuffle = False
    else:
        args.shuffle = True

    print("-"*50)
    print('# GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('# Model: {}'.format(args.model))

    return args




def prepare():
    if args.gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(args.gpu).use()

    reader = SentenceReaderDir(args.intraindata, args.batchsize)
    print("")
    print("-"*50)
    print('n_vocab: %d' % (len(reader.word2index)-3)) # excluding the three special tokens
    print('corpus size: %d' % (reader.total_words))

    max_sen_len = MAX_SEN_LEN
    train_data = convert_numeric_data(reader.data, args.batchsize, reader.word2index, max_sen_len)
    test_data, freq, _ = load_xml(args.intestdata)
    test_data = convert_numeric_data(test_data, args.batchsize, reader.word2index, max_sen_len)
    return reader, train_data, test_data




def objective(args):
    emb_dim = 100
    pre_trained_embedding = None
    use_pretrained = None

    ## 事前学習済みの分散表現を利用するかどうか ##
    if args.pretrained ==1:
        use_pretrained = reader.word2index
        pre_trained_embedding = "./embedding/rcv1_8_org.model.bin"

    print  ("-"*50)
    print (args.model)
    print  ("-"*50)

    if 'TRF' in args.model: 
        if args.model == 'TRF-Single':
            head = 5
            hopping = 1
            wsd_epoch = 0
            weight_decay = 1.4248358746328444e-10


        elif args.model == 'TRF-Multi':
            head = 10
            hopping = 1
            wsd_epoch = 0
            weight_decay = 4.391898194731847e-08

        elif args.model == 'TRF-Delay-Multi' or args.model == "TRF-Sequential":
            head = 5
            hopping = 1
            wsd_epoch = 1 #75
            weight_decay = 9.083672498965683e-08

        if args.model == "TRF-Sequential":
            model_wsd = Transformer(n_layers=hopping,n_source_vocab=len(reader.word2index), 
            n_units=emb_dim, catgy = reader.catgy, doc_catgy = reader.doc_catgy, 
            senseid2netout=reader.senseid2netout,model_type=args.model,
            word2index=reader.word2index, multi_label= args.multilabel, 
            pre_trained_embedding = pre_trained_embedding,
            wsd_epoch=wsd_epoch,h=head, dropout=0.1,max_length=100)
            model_tc = Transformer(n_layers=hopping,n_source_vocab=len(reader.word2index), 
                n_units=emb_dim, catgy = reader.catgy, doc_catgy = reader.doc_catgy, 
                senseid2netout=reader.senseid2netout,model_type=args.model,
                word2index=reader.word2index, multi_label= args.multilabel,
                pre_trained_embedding = pre_trained_embedding,
                wsd_epoch=wsd_epoch,h=head, dropout=0.1,max_length=100,wsd_model=model_wsd)
        else:
            model = Transformer(n_layers=hopping,n_source_vocab=len(reader.word2index), 
                n_units=emb_dim, catgy = reader.catgy, doc_catgy = reader.doc_catgy, 
                senseid2netout=reader.senseid2netout,model_type=args.model,
                word2index=reader.word2index, multi_label = args.multilabel,
                pre_trained_embedding = pre_trained_embedding,
                wsd_epoch=wsd_epoch,h=head, dropout=0.1,max_length=100)



    elif args.model == "XML-CNN":
        ## hyper params ##
        out_channels = 128
        filter_size = (4,5,6)
        weight_decay = 0.000305318639612637
        wsd_epoch = 0
        model = XMLCnn(doc_catgy=reader.doc_catgy, n_vocab=len(reader.word2index), emb_dim=emb_dim, 
        out_channels=out_channels, filter_size=filter_size, word2index=reader.word2index, pre_trained_embedding=pre_trained_embedding,
        multi_label= args.multilabel)
    else:
        raise Exception('Unknown context type: {}'.format(args.model))



    assert args.filepath != None
    file_path = args.filepath

    print("")
    print("-"*50)
    best_mic = 0
    best_mac = 0



    Epoch = args.epoch + wsd_epoch


    if args.model == "TRF-Sequential":
        loop = 2
    else:
        loop = 1

    for i in range(loop):
        if args.model == "TRF-Sequential":
            if i == 0:
                model = model_wsd
                Epoch = wsd_epoch
            elif i == 1:
                model = model_tc
                Epoch = args.epoch

        optimizer = O.Adam()
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

        ## epoch processing ##
        for epoch in tqdm(range(Epoch),desc="Epoch Processing"):

            main_loss = 0.0
            tc_loss = 0.0
            wsd_loss = 0.0
            batch_size = args.batchsize
            print('epoch: {0}'.format(epoch))

            train_data_size = len(train_data['indexed_text'])
            np.random.seed(1023)
            if args.shuffle is True:
                perm = np.random.permutation(train_data_size)
            elif args.shuffle is False:
                perm = np.arange(train_data_size)

            # iteration processing ##  
            with chainer.using_config('train', True):
                for i in tqdm(six.moves.range(0, train_data_size, batch_size)):
                    model.cleargrads()

                    x = np.array(train_data['indexed_text'])[perm[i : i + batch_size]].tolist()
                    pos = np.array(train_data['positions'])[perm[i : i + batch_size]].tolist()
                    labels = np.array(train_data['labels'])[perm[i : i + batch_size]].tolist()
                    keys = np.array(train_data['keys'])[perm[i : i + batch_size]].tolist()
                    doc_category = np.array(train_data['doc_category'])[perm[i : i + batch_size]].tolist()

                    sent = {"indexed_text":x, "positions": pos, "labels":labels, "keys":keys, "doc_category": doc_category}

                    if "TRF" in args.model:
                        mb_loss, mb_tc_loss,mb_wsd_loss = model(sent=sent, opt=optimizer, epoch = epoch, get_prediction=False)
                        tc_loss = mb_tc_loss if mb_tc_loss is None else tc_loss + mb_tc_loss
                        wsd_loss = mb_wsd_loss if mb_wsd_loss is None else wsd_loss + mb_wsd_loss
                    elif args.model == "XML-CNN" :
                        mb_loss = model(sent,optimizer)
                    main_loss += mb_loss

                print("main_loss: {}".format(main_loss))
                print("tc_loss: {}".format(tc_loss))
                print("wsd_loss: {}".format(wsd_loss))
                print ("")

            ## Test ##
            with chainer.using_config('train',False):
                estimate_test(test_data, reader, epoch, file_path, args.model, model, args.multilabel)


if __name__ == "__main__":
    args = parse_arguments()
    reader,train_data, test_data = prepare()
    objective(args)

