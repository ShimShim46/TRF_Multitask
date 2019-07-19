import copy
import math
import pdb
from itertools import chain
import math
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import scipy.sparse as sp
from context2vec.common.defs import Toks
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder ## single label
from sklearn.preprocessing import MultiLabelBinarizer ## multi label encoder

from pyfasttext import FastText
import random
linear_init = chainer.initializers.LeCunUniform()
# -*- coding:utf-8 -*-

# XML-CNN Network
# =========================================================
class XMLCnn(chainer.Chain):

    def __init__(self, doc_catgy, n_vocab, emb_dim, out_channels, filter_size, word2index, pre_trained_embedding, multi_label):
        self.in_channels = 1
        self.out_channels = out_channels
        self.row_dim = emb_dim
        self.hidden_dim = 512 ## fixed
        self.doc_catgy = doc_catgy
        self.n_classes = len(doc_catgy)
        self.n_vocab = n_vocab
        self.filter_size = filter_size
        self.word2index = word2index
        self.mutli_label = multi_label
        self.le = None
        if self.mutli_label == 1:
            self.le = MultiLabelBinarizer(classes=[i[0] for i in sorted(self.doc_catgy.items(), key=lambda x: x[1])],sparse_output=False)
        elif self.mutli_label == 0:
            self.le = LabelEncoder()
            self.le.fit([i[0] for i in sorted(self.doc_catgy.items(), key=lambda x: x[1])])
        self.look_up_table = None
        self.pre_trained_embedding = pre_trained_embedding
        super(XMLCnn, self).__init__()
        self.to_gpu()
        if not self.pre_trained_embedding is None:
                model = FastText(self.pre_trained_embedding)
                dim = len(model['a'])
                n_vocab = len(self.word2index.keys())
                self.look_up_table = self.xp.zeros((n_vocab, dim),dtype=np.float32)
                for word,index in tqdm(self.word2index.items()):
                    try:
                        self.look_up_table[index] = chainer.cuda.to_gpu(model.get_numpy_vector(word))
                    except:
                        self.xp.random.seed(index)
                        self.look_up_table[index][:] = self.xp.random.uniform(-0.25, 0.25, dim)

        self.set_seed_random(123)
        with self.init_scope():
            if self.look_up_table is None:
                self.embedding=L.EmbedID(n_vocab, self.row_dim, ignore_label=-1,initialW=linear_init)
            else:
                self.embedding=L.EmbedID(n_vocab, self.row_dim, ignore_label=-1,initialW=self.look_up_table)
            self.conv1 = L.Convolution2D(self.in_channels,self.out_channels,(filter_size[0],self.row_dim), stride=2,initialW=linear_init)
            self.conv2 = L.Convolution2D(self.in_channels,self.out_channels,(filter_size[1],self.row_dim), stride=2,initialW=linear_init)
            self.conv3 = L.Convolution2D(self.in_channels,self.out_channels,(filter_size[2],self.row_dim), stride=2,initialW=linear_init)
            self.l1=L.Linear(in_size = None, out_size = self.hidden_dim, initialW=linear_init)
            self.l2=L.Linear(in_size = self.hidden_dim, out_size = self.n_classes,initialW=linear_init)
        self.to_gpu()    

    # =========================================================

    def __call__(self, sent, opt):

        return self._calculate_loss(sent, opt)

    def _calculate_loss(self, sent, opt):
        self.set_seed_random(123)
        self.embedding.disable_update()
        with chainer.using_config('use_cudnn', 'never'):
            with chainer.using_config('cudnn_deterministic', True):
                x = self.xp.array(sent['indexed_text']) 

                if self.mutli_label == 1:
                    t_txt = self.le.fit_transform(sent['doc_category'])
                elif self.mutli_label == 0:
                    t_txt = self.le.transform(list(chain(*sent['doc_category'])))

                h_non_static = F.dropout(self.embedding(x),0.25)
                h_non_static = F.reshape(h_non_static, (h_non_static.shape[0], 1, h_non_static.shape[1], h_non_static.shape[2]))

                h1 = self.conv1(h_non_static)
                h2 = self.conv2(h_non_static)
                h3 = self.conv3(h_non_static)

                h1 = F.max_pooling_2d(F.relu(h1), (2,1), stride=1)
                h2 = F.max_pooling_2d(F.relu(h2), (2,1), stride=1)
                h3 = F.max_pooling_2d(F.relu(h3), (2,1), stride=1)

                h = F.concat((h1,h2,h3),axis=2)

                h = F.dropout(F.relu(self.l1(h)), ratio=0.5)
                y = self.l2(h)

                if self.mutli_label == 1:
                    loss = F.sigmoid_cross_entropy(y, self.xp.array(t_txt))
                elif self.mutli_label == 0:
                    loss = F.softmax_cross_entropy(y, self.xp.array(t_txt))
                loss.backward()

                opt.update()

        return loss.data

    def estimate(self, sent):
        self.set_seed_random(123)
        self.embedding.disable_update()
        x = sent['indexed_text']
        try:
            x = self.xp.array(x)
        except:
            pdb.set_trace()

        h_non_static = F.dropout(self.embedding(x),0.25)
        h_non_static = F.reshape(h_non_static, (h_non_static.shape[0], 1, h_non_static.shape[1], h_non_static.shape[2]))
        
        h1 = self.conv1(h_non_static)
        h2 = self.conv2(h_non_static)
        h3 = self.conv3(h_non_static)

        h1 = F.max_pooling_2d(F.relu(h1), (2,1), stride=1)
        h2 = F.max_pooling_2d(F.relu(h2), (2,1), stride=1)
        h3 = F.max_pooling_2d(F.relu(h3), (2,1),stride=1)
    
        h = F.concat((h1,h2,h3),axis=2)
        h = F.dropout(F.relu(self.l1(h)), ratio=0.5)
        
        y = self.l2(h)

        return y

    # The Setting of the seed value for random number generation
    # =========================================================
    def set_seed_random(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        if chainer.cuda.available:
            chainer.cuda.cupy.random.seed(seed)

