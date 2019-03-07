#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Learns context2vec's parametric model
"""
import argparse
import time
import sys
import pdb
from tqdm import tqdm
import random
import six
from itertools import chain

import numpy as np
import cupy as cp
from chainer import cuda
import chainer.links as L
import chainer.optimizers as O
import chainer.serializers as S
import chainer.computational_graph as C
import chainer.functions as F
import chainer
import chainer.computational_graph as c

from dataset_reader import DatasetReader
from sentence_reader import SentenceReaderDir
from sentence_reader import make_test_data
from sentence_reader import convert_numeric_data
from sentence_reader import load_xml
from context2vec.common.context_models import LstmContext
from context2vec.common.context_models import Cnn
from context2vec.common.context_models import XMLCnn
from context2vec.common.context_models import LstmCnnContext
from context2vec.common.context_models import LstmVotingContext
from context2vec.common.context_models import LstmAttenContext
from context2vec.common.context_models import LstmLstmContext
from net import Transformer
# from context2vec.common.context_models import key_output_layers
from model_reader_shimura import ModelReader
from context2vec.common.defs import IN_TO_OUT_UNITS_RATIO, NEGATIVE_SAMPLING_NUM
import math
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
chainer.config.cudnn_deterministic = True
chainer.config.use_cudnn = 'never'
import optuna

#TODO: LOWER AS ARG
def dump_embeddings(filename, w, units, index2word):
    with open(filename, 'w') as f:
        f.write('%d %d\n' % (len(index2word), units))
        for i in range(w.shape[0]):
            v = ' '.join(['%f' % v for v in w[i]])
            f.write('%s %s\n' % (index2word[i], v))
            
def dump_comp_graph(filename, vs):
    g = C.build_computational_graph(vs)
    with open(filename, 'w') as o:
        o.write(g.dump())

def estimate_test(test_data, reader,multitask,task_type,epoch,path, model_type, best_mic, best_mac, model):
    ## 推論 ##
    print ("")
    print ("-"*50)
    print ("Estimate")
    batch_size = args.batchsize

    ## 単純な推論では駄目 ##
    ## テストデータにもキーがあるので, 全てのカテゴリの結果を持ってくるのは駄目. 対象のカテゴリだけに絞る必要がある. ##
    test_data_size = len(test_data['indexed_text'])
    netouts_m = [] ## multi
    netouts_s = [] ## single
    netouts_txt_cat = []
    network_output_order = [i[0] for i in sorted(dict(reader.catgy).items(), key=lambda x: x[1])]

    for i in tqdm(six.moves.range(0, test_data_size, batch_size)):
        x = test_data['indexed_text'][i : i + batch_size]
        positions = test_data['positions'][i : i + batch_size]
        labels = test_data['labels'][i : i + batch_size]
        keys = test_data['keys'][i : i + batch_size]

        # keys = list(chain(*keys))
        # labels = labels
        sent = {"indexed_text":x, "positions": positions, "labels":labels, "keys":keys}
        if model_type == "transformer":
            preds_txt = model(sent,None,epoch,True)
        else:
            preds_txt = model.estimate(sent, multitask)
        
        preds_txt = F.softmax(preds_txt) ## 文書ラベルの方をsigmoidで活性化
        ## 文書分類ラベルの変換 ##
        pred_txt_label_list = [[] for i in range(preds_txt.shape[0])]
        indexes = chainer.cuda.to_cpu(F.argmax(preds_txt,axis=1).data).tolist()
        converted_label = model.le.inverse_transform(indexes).tolist()
        netouts_txt_cat.extend(converted_label)
        

        # if multitask is True and curriculum is True:
        #     preds_wsd = F.concat(preds_wsd, axis=0)
        #     label_indexes = np.where(np.array(list(chain(*labels))) != "<PAD>")[0]
        #     label_names = np.array(list(chain(*labels)))[label_indexes]
        #     keys = list(chain(*keys))
        #     selected_preds_wsd = preds_wsd[label_indexes]
        #     assert len(label_indexes) == len(label_names) == len(keys) == len(selected_preds_wsd)
        #     for p_wsd, key, label in zip(selected_preds_wsd, keys, label_names):
        #         # pdb.set_trace()
        #         if key in reader.key2netout:
        #             pos_of_category = reader.key2netout[key]
        #             names = np.array(network_output_order)[pos_of_category]
        #             # pdb.set_trace()

        #             p = p_wsd[pos_of_category]
        #             single_label = names[np.argmax(chainer.cuda.to_cpu(p.data))]
        #             netouts_s.append(single_label)
        #         else:
        #             # label = "UNKOWN_TARGET"
        #             label_unk = label
        #             netouts_s.append(label_unk)

    
    ## F値の計算 ##
    pred = netouts_txt_cat
    ans = list(chain(*test_data['doc_category']))
    micro_f1 = f1_score(y_pred=pred, y_true=ans, average="micro")
    macro_f1 = f1_score(y_pred=pred, y_true=ans, average="macro")
    weighted_f1 = f1_score(y_pred=pred,y_true=ans, average="weighted")
    accuracy = accuracy_score(y_true=ans, y_pred=pred)

    print ("-"*50)
    print ("Micro F1\t{}".format(micro_f1))
    print ("Macro F1\t{}".format(macro_f1))
    print ("Accuracy\t{}".format(accuracy))
    print ("-"*50)
 
    best_mac = max(best_mac, macro_f1)
    best_mic = max(best_mic, micro_f1)    

    # print("")
    # print("-"*50)
    # print("Writing out prediction...")
    # ## モデル保存 ##
    # if multitask  is True:
    #     task = "MULTI-TASK"
    # else:
    #     task = "SINGLE-TASK"

    # result_file_txt_cat = open(path + "/RESULT_FILE_" + task + "_" + str(epoch) + "EPOCH_single_TXT_CAT", mode="w")
    # result_file_fscore = open(path + "/RESULT_FILE_" + task + "_" + str(epoch) + "EPOCH_fscore", mode="w")

    # result_file_fscore.write("Micro F1\t" + str(micro_f1) + "\n")
    # result_file_fscore.write("Macro F1\t" + str(macro_f1) + "\n")
    # result_file_fscore.write("Weighted F1\t" + str(weighted_f1) + "\n")
    # result_file_fscore.write("Accuracy\t" + str(accuracy) + "\n")
    # result_file_fscore.close()

    # result_file_txt_cat.write("Ground Truth\tPrediction\n")
    # for i,j in zip(test_data["doc_category"], netouts_txt_cat):
    #     result_file_txt_cat.write(",".join(i) + "\t" + ",".join(j)+"\n")
    # result_file_txt_cat.close()

    # if multitask is True and curriculum is True:
    #     result_file_s = open(path + "/RESULT_FILE_" + task + "_" + str(epoch) + "EPOCH_single_WSD", mode="w")
    #     instance_id_flatten = list(chain(*[j for i in test_data['instance_id'] for j in i]))

    #     for i,k in zip(instance_id_flatten, netouts_s):
    #         result_file_s.write(i + " " + k + "\n")
    #     result_file_s.close()

    return best_mic, best_mac
    
def select_function(scores):
    indexes = [int(np.argmax(score.data)) if not score is None else None for score in scores]
    return indexes
        
def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--intraindata', '-itrain',
                        default=None,
                        help='input train corpus directory')
    parser.add_argument('--intestdata', '-itest',
                        default=None,
                        help='input test file name')
  
    parser.add_argument('--trimfreq', '-t', default=0, type=int,
                        help='minimum frequency for word in training')
    parser.add_argument('--dropout', '-o', default=0.0, type=float,
                        help='NN dropout')
    parser.add_argument('--wordsfile', '-w',
                        default=None,
                        help='word embeddings output filename')
    parser.add_argument('--modelfile', '-m',
                        default=None,
                        help='model output filename')
    parser.add_argument('--cgfile', '-cg',
                        default=None,
                        help='computational graph output filename (for debug)')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', default=300, type=int,
                        help='number of units (dimensions) of one context word')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=10, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--context', '-c', choices=['lstm', 'pretrainedLstm', 'cnn','lstmcnn','lstm_voting', 'lstm_attn','lstmlstm','transformer','xmlcnn'],
                        default='lstm',
                        help='context type ("lstm")')
    parser.add_argument('--deep', '-d', choices=['yes', 'no'],
                        default=None,
                        help='use deep NN architecture')
    parser.add_argument('--modelparams', '-mp', default=None, help='input model params file name')
    parser.add_argument('--numoftrfiles', '-ntrf', default=None)
    parser.add_argument('--numoftefiles', '-ntef', default=None)
    parser.add_argument('--multitask', '-mt', default='yes', type=str)
    parser.add_argument('--task', default='context2vec', type=str)
    parser.add_argument('--filepath', '-fp',default=None)
    parser.add_argument('--saveModel',default='no', type=str)
    parser.add_argument('--shuffle',default='no', type=str)
    parser.add_argument('--includeAllWords',default='no', type=str)
    parser.add_argument('--oneline', default='no',type=str)
    parser.add_argument('--nstep',default=1,type=int)
    parser.add_argument('--head',default=8,type=int)
    parser.add_argument('--depth',default=1,type=int)
    parser.add_argument('--mhacnn',type=str)
    parser.add_argument('--wsd',type=int)
    parser.add_argument('--curriculum',type=int)
    parser.add_argument('--cheatWSD',default=0,type=int)
    parser.add_argument('--wsdWeight',default=1.0,type=float)
    parser.add_argument('--storagename',type=str)
    parser.add_argument('--dbname',type=str)
    args = parser.parse_args()
    
    if args.shuffle == 'no':
        args.shuffle = False
    else:
        args.shuffle = True

    if args.includeAllWords == 'no':
        args.includeAllWords = False
    else:
        args.includeAllWords = True

    if args.multitask == 'yes':
        args.multitask = True
    elif args.multitask == 'no':
        args.multitask = False
    else:
        raise Exception("Invalid choice")

    if args.deep == 'yes':
        args.deep = True
    elif args.deep == 'no':
        args.deep = False
    else:
        raise Exception("Invalid deep choice: " + args.deep)
    
    if args.mhacnn == 'yes':
        args.mhacnn = True
    else:
        args.mhacnn = False
    if args.saveModel == 'yes':
        args.saveModel = True
    elif args.saveModel == 'no':
        args.saveModel = False
    else:
        raise Exception("Invalid save model choice:" + args.saveModel)

    if args.oneline == "no":
        args.oneline = False
    elif args.oneline == "yes":
        args.oneline = True
    else:
        raise Exception("Invalid oneline mode choice:" + args.oneline)

    print("-"*50)
    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('Context type: {}'.format(args.context))
    print('Deep: {}'.format(args.deep))
    print('Dropout: {}'.format(args.dropout))
    print('Trimfreq: {}'.format(args.trimfreq))
    if args.context == "pretrainedLstm":
        print('Model params file: {}'.format(args.modelparams))
    print('Multi-task: {}'.format(args.multitask))
    if args.multitask == False:
        print("Target task: {}".format(args.task))

    return args 

def prepare():

    if args.gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(args.gpu).use()
    xp = cuda.cupy if args.gpu >= 0 else np
    
    pretrained = False
    if args.context == 'pretrainedLstm':
        pretrained=True

    if args.context == "cnn" or args.context == "transformer" or args.context == "lstm" or args.context == "xmlcnn":
        FIXED_LEN_PADDING = True
        args.oneline = True
    else:
        FIXED_LEN_PADDING = False

    reader = SentenceReaderDir(args.intraindata, args.trimfreq, args.batchsize, args.includeAllWords, oneline=args.oneline) # inibatchを生成するやつ
    print("")
    print("-"*50)
    print('n_vocab: %d' % (len(reader.word2index)-3)) # excluding the three special tokens
    print('corpus size: %d' % (reader.total_words))
    reader.make_catgy()
    # pdb.set_trace()
    reader.make_key2netout()
    # pdb.set_trace()
    ## csは単語id順に出現頻度が入っているリスト ##

    if args.oneline is True:
        max_sen_len = 100
        train_data = convert_numeric_data(reader.data, args.batchsize, reader.word2index, max_sen_len, fixed_len_padding=FIXED_LEN_PADDING)
        test_data, freq, _ = load_xml(args.intestdata, includeAllWords=args.includeAllWords, oneline=args.oneline)
        # max_sen_len = max(reader.one_sentence_max_len, max_sen_len)
        test_data = convert_numeric_data(test_data, args.batchsize, reader.word2index, max_sen_len, fixed_len_padding=FIXED_LEN_PADDING)
    else:
        train_data = convert_numeric_data(reader.data, args.batchsize, reader.word2index, max_len=None, fixed_len_padding=FIXED_LEN_PADDING)
        test_data, freq = load_xml(args.intestdata,args.intestdataa, includeAllWords=args.includeAllWords, oneline=args.oneline)
        test_data = convert_numeric_data(test_data, args.batchsize, reader.word2index, max_len=None, fixed_len_padding=FIXED_LEN_PADDING)
        
        
    return reader, train_data, test_data

# def create_hyper_parms(trial):
#     out_channels = trial.suggest_categorical('out_channels', [16,32,64,128,256])
#     batch_size = trial.suggest_categorical('batch_size', [8,16,32,64,100,128])
#     unit_size = trial.suggest_categorical('unit_size', [128,256,512,1024,2028])
#     hyper_params = {"out_channels":out_channels, "batch_size":batch_size, "unit_size":unit_size}
#     return hyper_params


def objective(trial):
    # context_word_units = args.unit
    # lstm_hidden_units = IN_TO_OUT_UNITS_RATIO*args.unit
    # target_word_units = IN_TO_OUT_UNITS_RATIO*args.unit
    emb_dim = 100 ##これはHEADの数が関係するので今回は固定
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    if args.context == 'transformer':
        ## hyper params ##
        head = trial.suggest_categorical('num_of_head',[1,2,4,5,10])
        hopping = trial.suggest_categorical('hopping', [1,2,3,4])
        if args.curriculum == 1:
            wsd_epoch = trial.suggest_categorical('wsd_epoch',[25,50,75,100])
            
        else:
            wsd_epoch = 0
        model = Transformer(n_layers=hopping,n_source_vocab=len(reader.word2index), 
        n_target_vocab=len(reader.word2index), n_units=emb_dim, 
        catgy = reader.catgy, doc_catgy = reader.doc_catgy, senseid2netout=reader.senseid2netout, 
        cnn_mode=False, index2word = reader.index2word,key2netout =reader.key2netout, wsd=args.wsd, 
        total_epoch=args.epoch+wsd_epoch, curriculum=args.curriculum, wsd_epoch=wsd_epoch,
        wsdWeight=args.wsdWeight, cheatWSD=args.cheatWSD, 
        mhacnn = args.mhacnn, h=head, dropout=0.1,max_length=100)
    
    elif args.context == "xmlcnn":
        ## hyper params ##
        wsd_epoch=0
        out_channels = trial.suggest_categorical('out_channels', [8,16,32,64,128,256])
        filter_size = trial.suggest_categorical('filter_size', [(1,2,3),(2,3,4),(3,4,5),(4,5,6)])
        model = XMLCnn(doc_catgy=reader.doc_catgy, n_vocab=len(reader.word2index), emb_dim=emb_dim, out_channels=out_channels, filter_size=filter_size)
    else:
        raise Exception('Unknown context type: {}'.format(args.context))

    optimizer = O.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))


    assert args.filepath != None
    file_path = args.filepath


    STATUS_INTERVAL = 1000000
    print("")
    print("-"*50)
    best_mic = 0
    best_mac = 0
    ## epoch processing ##
    for epoch in tqdm(range(args.epoch+wsd_epoch),desc="Epoch Processing"):
        begin_time = time.time()
        cur_at = begin_time
        word_count = 0
        next_count = STATUS_INTERVAL
        accum_loss = 0.0
        tc_loss = 0.0
        wsd_loss = 0.0
        batch_size = args.batchsize
        print('epoch: {0}'.format(epoch))
        
        train_data_size = len(train_data['indexed_text'])
        np.random.seed(1023)
        perm = np.random.permutation(train_data_size)

        # iteration processing ##  
        with chainer.using_config('train', True):
            for i in tqdm(six.moves.range(0, train_data_size, batch_size)):
                model.cleargrads()
        
                if args.shuffle is True:
                    x = np.array(train_data['indexed_text'])[perm[i : i + batch_size]].tolist()
                    pos = np.array(train_data['positions'])[perm[i : i + batch_size]].tolist()
                    labels = np.array(train_data['labels'])[perm[i : i + batch_size]].tolist()
                    keys = np.array(train_data['keys'])[perm[i : i + batch_size]].tolist()
                    doc_category = np.array(train_data['doc_category'])[perm[i : i + batch_size]].tolist()
                else:
                    x = train_data['indexed_text'][i : i + batch_size]
                    pos = train_data['positions'][i : i + batch_size]
                    labels = train_data['labels'][i : i + batch_size]
                    keys = train_data['keys'][i : i + batch_size]
                    doc_category = train_data['doc_category'][i : i + batch_size]
                sent = {"indexed_text":x, "positions": pos, "labels":labels, "keys":keys, "doc_category": doc_category}
                if args.context == "transformer":
                    loss,loss_tc,loss_wsd = model(sent=sent, opt=optimizer, epoch = epoch, get_prediction=False)
                    tc_loss += loss_tc
                    wsd_loss += loss_wsd
                else:
                    loss = model(sent,args.task,args.multitask,optimizer)
                accum_loss += loss
                

            
            
            print("accum_loss: {}".format(accum_loss))
            print("tc_loss: {}".format(tc_loss))
            print("wsd_loss: {}".format(wsd_loss))
            print ("")
            # pdb.set_trace()
        with chainer.using_config('train',False):
            best_mic, best_mac = estimate_test(test_data, reader, args.multitask, args.task, epoch, file_path, args.context,best_mic,best_mac,model)
            
    ## 学習終了 ##
    # print ("Best Micro-F1\t{}".format(best_mic))
    # print ("Best Macro-F1\t{}".format(best_mac))
    return 1 - best_mic ## microが最小になるように


if __name__ == "__main__":
    args = parse_arguments()
    reader,train_data, test_data = prepare()
    study = optuna.Study(study_name=args.dbname, storage=args.storagename)
    if args.curriculum == 1:
        study.optimize(objective, n_trials=20)
    else:
        study.optimize(objective, n_trials=50)
    with open(args.filepath + "/opt_result.txt", mode='w') as f:
        f.write('Number of finished trials: {}'.format(len(study.trials)) + "\n")

        f.write('Best trial:'+"\n")
        trial = study.best_trial

        f.write('  Value: {}'.format(trial.value)+"\n")
        f.write('  Params: ' + "\n")
        for key, value in trial.params.items():
           f.write('    {}: {}'.format(key, value) + "\n")

    hist_df = study.trials_dataframe()
    hist_df.to_csv(args.filepath + "/history.csv")
