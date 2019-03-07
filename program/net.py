# encoding: utf-8

import pdb
import random
from itertools import chain

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import cuda, reporter
from chainer.dataset import convert
from gensim.models import KeyedVectors
from sklearn.preprocessing import LabelEncoder  # # single label encoder
from sklearn.preprocessing import MultiLabelBinarizer  # # multi label encoder

from pyfasttext import FastText

linear_init = chainer.initializers.LeCunUniform()


def sentence_block_embed(embed, x):
    """ Change implicitly embed_id function's target to ndim=2

    Apply embed_id for array of ndim 2,
    shape (batchsize, sentence_length),
    instead for array of ndim 1.

    ただの行列の転置
    """
    batch, length = x.shape
    _, units = embed.W.shape
    e = embed(x.reshape((batch * length, )))
    assert(e.shape == (batch * length, units))
    e = F.transpose(F.stack(F.split_axis(e, batch, axis=0), axis=0), (0, 2, 1))
    assert(e.shape == (batch, units, length))
    return e

def seq_func(func, x, reconstruct_shape=True):
    """ Change implicitly function's target to ndim=3

    Apply a given function for array of ndim 3,
    shape (batchsize, dimension, sentence_length),
    instead for array of ndim 2.

    """

    batch, units, length = x.shape
    e = F.transpose(x, (0, 2, 1)).reshape(batch * length, units)
    e = func(e)
    if not reconstruct_shape:
        return e
    out_units = e.shape[1]
    e = F.transpose(e.reshape((batch, length, out_units)), (0, 2, 1))
    assert(e.shape == (batch, out_units, length))
    return e


class LayerNormalizationSentence(L.LayerNormalization):

    """ Position-wise Linear Layer for Sentence Block

    Position-wise layer-normalization layer for array of shape
    (batchsize, dimension, sentence_length).

    """

    def __init__(self, *args, **kwargs):
        super(LayerNormalizationSentence, self).__init__(*args, **kwargs)

    def __call__(self, x):
        y = seq_func(super(LayerNormalizationSentence, self).__call__, x)
        return y


class ConvolutionSentence(L.Convolution2D):

    """ Position-wise Linear Layer for Sentence Block

    Position-wise linear layer for array of shape
    (batchsize, dimension, sentence_length)
    can be implemented a convolution layer.

    """

    def __init__(self, in_channels, out_channels,
                 ksize=1, stride=1, pad=0, nobias=False,
                 initialW=None, initial_bias=None):
        super(ConvolutionSentence, self).__init__(
            in_channels, out_channels,
            ksize, stride, pad, nobias,
            initialW, initial_bias)

    def __call__(self, x):
        """Applies the linear layer.

        Args:
            x (~chainer.Variable): Batch of input vector block. Its shape is
                (batchsize, in_channels, sentence_length).

        Returns:
            ~chainer.Variable: Output of the linear layer. Its shape is
                (batchsize, out_channels, sentence_length).

        """
        x = F.expand_dims(x, axis=3)
        y = super(ConvolutionSentence, self).__call__(x)
        y = F.squeeze(y, axis=3)
        return y


class MultiHeadAttention(chainer.Chain):

    """ Multi Head Attention Layer for Sentence Blocks

    For batch computation efficiency, dot product to calculate query-key
    scores is performed all heads together.

    """

    def __init__(self, n_units, h=8, dropout=0.1, self_attention=True):
        super(MultiHeadAttention, self).__init__()
        with self.init_scope():
            if self_attention:
                self.W_QKV = ConvolutionSentence(
                    n_units, n_units * 3, nobias=True,
                    initialW=linear_init)
            else:
                self.W_Q = ConvolutionSentence(
                    n_units, n_units, nobias=True,
                    initialW=linear_init)
                self.W_KV = ConvolutionSentence(
                    n_units, n_units * 2, nobias=True,
                    initialW=linear_init)
            self.finishing_linear_layer = ConvolutionSentence(
                n_units, n_units, nobias=True,
                initialW=linear_init)
        self.h = h
        self.scale_score = 1. / (n_units // h) ** 0.5
        self.dropout = dropout
        self.is_self_attention = self_attention

    def __call__(self, x, z=None, mask=None):
        xp = self.xp
        h = self.h

        if self.is_self_attention:
            Q, K, V = F.split_axis(self.W_QKV(x), 3, axis=1)
        else:
            Q = self.W_Q(x)
            K, V = F.split_axis(self.W_KV(z), 2, axis=1)
        batch, n_units, n_querys = Q.shape
        _, _, n_keys = K.shape

        # Calculate Attention Scores with Mask for Zero-padded Areas
        # Perform Multi-head Attention using pseudo batching
        # all together at once for efficiency

        batch_Q = F.concat(F.split_axis(Q, h, axis=1), axis=0)
        batch_K = F.concat(F.split_axis(K, h, axis=1), axis=0)
        batch_V = F.concat(F.split_axis(V, h, axis=1), axis=0)
        assert(batch_Q.shape == (batch * h, n_units // h, n_querys))
        assert(batch_K.shape == (batch * h, n_units // h, n_keys))
        assert(batch_V.shape == (batch * h, n_units // h, n_keys))

        mask = xp.concatenate([mask] * h, axis=0)
        batch_A = F.batch_matmul(batch_Q, batch_K, transa=True) \
            * self.scale_score
        batch_A = F.where(mask, batch_A, xp.full(batch_A.shape, -np.inf, 'f'))
        batch_A = F.softmax(batch_A, axis=2)
        batch_A = F.where(
            xp.isnan(batch_A.data), xp.zeros(batch_A.shape, 'f'), batch_A)
        assert(batch_A.shape == (batch * h, n_querys, n_keys))

        # Calculate Weighted Sum
        batch_A, batch_V = F.broadcast(
        batch_A[:, None], batch_V[:, :, None])
        batch_C = F.sum(batch_A * batch_V, axis=3)
        assert(batch_C.shape == (batch * h, n_units // h, n_querys))

        
           
        C = F.concat(F.split_axis(batch_C, h, axis=0), axis=1)
        assert(C.shape == (batch, n_units, n_querys))
        C = self.finishing_linear_layer(C)
        return C


class FeedForwardLayer(chainer.Chain):
    def __init__(self, n_units):
        super(FeedForwardLayer, self).__init__()
        n_inner_units = n_units * 4
        with self.init_scope():
            self.W_1 = ConvolutionSentence(n_units, n_inner_units,
                                           initialW=linear_init)
            self.W_2 = ConvolutionSentence(n_inner_units, n_units,
                                           initialW=linear_init)
            # self.act = F.relu
            self.act = F.leaky_relu

    def __call__(self, e):
        e = self.W_1(e)
        e = self.act(e)
        e = self.W_2(e)
        return e


class EncoderLayer(chainer.Chain):
    def __init__(self, n_units, h=8, dropout=0.1):
        super(EncoderLayer, self).__init__()
        with self.init_scope():
            self.self_attention = MultiHeadAttention(n_units, h)
            self.feed_forward = FeedForwardLayer(n_units)
            self.ln_1 = LayerNormalizationSentence(n_units, eps=1e-6)
            self.ln_2 = LayerNormalizationSentence(n_units, eps=1e-6)
        self.dropout = dropout

    def __call__(self, e, xx_mask):
        sub = self.self_attention(e, e, xx_mask)
        e = e + F.dropout(sub, self.dropout)
        e = self.ln_1(e)

        sub = self.feed_forward(e)
        e = e + F.dropout(sub, self.dropout)
        e = self.ln_2(e)
        return e





class Encoder(chainer.Chain):
    def __init__(self, n_layers, n_units, h=8, dropout=0.1):
        super(Encoder, self).__init__()
        self.layer_names = []
        for i in range(1, n_layers + 1):
            name = 'l{}'.format(i)
            layer = EncoderLayer(n_units, h, dropout)
            self.add_link(name, layer)
            self.layer_names.append(name)

    def __call__(self, e, xx_mask):
        for name in self.layer_names:
            e = getattr(self, name)(e, xx_mask)
        return e





class Transformer(chainer.Chain):

    def __init__(self, n_layers, n_source_vocab, n_units, catgy, doc_catgy, senseid2netout, word2index, pre_trained_embedding, model_type,multi_label,
                wsd_epoch = 0,
                h=8, dropout=0.1, max_length=500,
                use_label_smoothing=False,
                embed_position=False, wsd_model=None):
        super(Transformer, self).__init__()
        self.to_gpu()
        self.set_random_seed(123)
        self.word2index = word2index
        self.pre_trained_embedding = pre_trained_embedding
        self.model_type = model_type
        self.wsd_model = wsd_model
        self.multi_label = multi_label

        with self.init_scope():
            if not self.pre_trained_embedding is None:
                model = FastText(self.pre_trained_embedding)
                dim = len(model['a'])
                n_vocab = len(self.word2index.keys())
                self.look_up_table = self.xp.zeros((n_vocab, dim),dtype=np.float32)
                for word,index in self.word2index.items():
                    try:
                        self.look_up_table[index] = chainer.cuda.to_gpu(model.get_numpy_vector(word))
                    except:
                        self.xp.random.seed(index)
                        self.look_up_table[index][:] = self.xp.random.uniform(-0.25, 0.25, dim)
                self.embed_x = L.EmbedID(n_source_vocab, n_units, ignore_label=-1,
                                        initialW=self.look_up_table)
            else:
                self.embed_x = L.EmbedID(n_source_vocab, n_units, ignore_label=-1,
                                    initialW=linear_init)

            self.encoder = Encoder(n_layers, n_units, h, dropout)

            self.fc2 = L.Linear(in_size=n_units, out_size=len(doc_catgy),initialW=linear_init)
            self.fc2_wsd = L.Linear(in_size=n_units, out_size=len(catgy),initialW=linear_init)
            self.lookup_table_sense = L.EmbedID(in_size=len(catgy),out_size=n_units,ignore_label=-1, initialW=linear_init)
            self.lookup_table_sense_fixed = self.lookup_table_sense.W.data
            self.senseid2netout = senseid2netout
            self.senseid2netout['<PAD>'] = [-1]
            
            self.wsd_epoch = wsd_epoch
            if embed_position:
                self.embed_pos = L.EmbedID(max_length, n_units,
                                           ignore_label=-1)

        self.n_layers = n_layers
        self.n_units = n_units
        self.dropout = dropout
        self.use_label_smoothing = use_label_smoothing
        self.initialize_position_encoding(max_length, n_units)
        self.scale_emb = self.n_units ** 0.5 ## origin 0.5
        self.doc_catgy = doc_catgy
        self.catgy = catgy
        self.inverse_catgy = {v:k for k,v in self.catgy.items()}

        self.wsd_netout2wordindex = {k:self.word2index[v] for k,v in self.inverse_catgy.items() }
        self.wsd_netout2wordindex[-1] = -1
        self.max_len = max_length
        self.le = None
        if self.multi_label == 1:
            self.le = MultiLabelBinarizer(classes=[i[0] for i in sorted(self.doc_catgy.items(), key=lambda x: x[1])],sparse_output=False)
        elif self.multi_label == 0:
            self.le = LabelEncoder()
            self.le.fit([i[0] for i in sorted(self.doc_catgy.items(), key=lambda x: x[1])])
        self.to_gpu()


    def initialize_position_encoding(self, length, n_units):
        xp = self.xp
        """
        # Implementation described in the paper
        start = 1  # index starts from 1 or 0
        posi_block = xp.arange(
            start, length + start, dtype='f')[None, None, :]
        unit_block = xp.arange(
            start, n_units // 2 + start, dtype='f')[None, :, None]
        rad_block = posi_block / 10000. ** (unit_block / (n_units // 2))
        sin_block = xp.sin(rad_block)
        cos_block = xp.cos(rad_block)
        self.position_encoding_block = xp.empty((1, n_units, length), 'f')
        self.position_encoding_block[:, ::2, :] = sin_block
        self.position_encoding_block[:, 1::2, :] = cos_block
        """

        # Implementation in the Google tensor2tensor repo
        channels = n_units
        position = xp.arange(length, dtype='f')
        num_timescales = channels // 2
        log_timescale_increment = (
            xp.log(10000. / 1.) /
            (float(num_timescales) - 1))
        inv_timescales = 1. * xp.exp(
            xp.arange(num_timescales).astype('f') * -log_timescale_increment)
        scaled_time = \
            xp.expand_dims(position, 1) * \
            xp.expand_dims(inv_timescales, 0)
        signal = xp.concatenate(
            [xp.sin(scaled_time), xp.cos(scaled_time)], axis=1)
        signal = xp.reshape(signal, [1, length, channels])
        self.position_encoding_block = xp.transpose(signal, (0, 2, 1))

    def make_input_embedding(self, embed, block):
        ## 位置エンコーディング
        batch, length = block.shape
        emb_block = sentence_block_embed(embed, block) * self.scale_emb
        emb_block += self.xp.array(self.position_encoding_block[:, :, :length])
        if hasattr(self, 'embed_pos'):
            emb_block += sentence_block_embed(
                self.embed_pos,
                self.xp.broadcast_to(
                    self.xp.arange(length).astype('i')[None, :], block.shape))
        emb_block = F.dropout(emb_block, self.dropout)
        return emb_block

    def make_attention_mask(self, source_block, target_block):
        mask = (target_block[:, None, :] >= 0) * \
            (source_block[:, :, None] >= 0)
        # (batch, source_length, target_length)
        return mask

    def make_history_mask(self, block):
        batch, length = block.shape
        arange = self.xp.arange(length)
        history_mask = (arange[None, ] <= arange[:, None])[None, ]
        history_mask = self.xp.broadcast_to(
            history_mask, (batch, length, length))
        return history_mask

    
    def tc(self,trf_encoded_matrix):
        sum_matrix = F.sum(trf_encoded_matrix,axis=2)
        y_tc = self.fc2(sum_matrix)
        return y_tc

    def wsd_only(self,trf_encoded_matrix,labels):
        ### WSD ###
        wsd = trf_encoded_matrix.reshape(-1,trf_encoded_matrix.shape[-2]) ## [batch_size * word, depth]
        y_wsd = self.fc2_wsd(wsd) ## ここがWSDの予測結果
        conv_list = [self.senseid2netout[i] if i in self.senseid2netout else np.arange(len(self.senseid2netout)).tolist() for i in list(chain(*labels))] ##見る範囲を限定する. testのときに未知の語義ラベルであれば仕方ないのですべての範囲を見る
        mask = F.broadcast_to(self.xp.array([-1024.0]*y_wsd.shape[-1],dtype=y_wsd.dtype),y_wsd.shape) ## 見ないところをマスクする
        cond = chainer.Variable(self.xp.zeros(y_wsd.shape,dtype=bool))
        for i,cl in enumerate(conv_list):
            if cl[0] != -1:
                cond.data[i][cl] = True
            else:
                continue
        y_wsd = F.where(cond,y_wsd,mask)

        return y_wsd

    def wsd_with_tc(self,sent,trf_encoded_matrix,labels):

        ### WSD ###

        if self.model_type == "TRF-Multi" or self.model_type == "TRF-Delay-Multi":
            y_wsd = self.wsd_only(trf_encoded_matrix, labels)
        elif self.model_type == "TRF-Sequential":
            y_wsd,task_type= self.wsd_model(sent, None, None, True) ## 読み込みsequential

        y_wsd_soft = F.softmax(y_wsd) ## 予測結果にSoftmaxをかける
        argmax_wsd = F.argmax(y_wsd_soft,axis=1) ## 最大のインデクス値を取ってくる
        cond = chainer.Variable(self.xp.array([True if i != "<PAD>" else False for i in list(chain(*labels))])) ## 語義のラベルがついていない単語は無視するための条件
        pad_array = chainer.Variable(-1 * self.xp.ones(argmax_wsd.shape,dtype=argmax_wsd.dtype))
        pad_array_argmax_wsd = F.where(cond, argmax_wsd, pad_array)

        sense_label_embed = F.embed_id(x=pad_array_argmax_wsd,W=self.xp.array(self.lookup_table_sense_fixed),ignore_label=-1) ## 固定.

        sense_label_embed = sense_label_embed.reshape(trf_encoded_matrix.shape[0],trf_encoded_matrix.shape[-1],-1)
        origin_shape = sense_label_embed.shape
        sense_label_embed = F.moveaxis(sense_label_embed,1,2)

        ## 置き換え ##
        cond_reshape = cond.reshape(cond.shape[0],-1)
        cond_reshape = F.broadcast_to(cond_reshape,(cond_reshape.shape[0], trf_encoded_matrix.shape[1]))
        cond_reshape = cond_reshape.reshape(origin_shape)
        cond_reshape = F.swapaxes(cond_reshape,1,2)
        replaced_trf_matrix = F.where(cond_reshape,sense_label_embed,trf_encoded_matrix)

        ### WSDの予測をTCに組み入れる ###
        tc = replaced_trf_matrix ## 置換後の文書行列

        ### TC ###
        tc_features = F.sum(tc,axis=2) ## TC特徴
        y_tc = self.fc2(tc_features) ### TCの予測結果

        return (y_tc, y_wsd) if (self.model_type == "TRF-Multi") or (self.model_type == "TRF-Delay-Multi") else y_tc


    def __call__(self, sent, opt, epoch=None, get_prediction=False):
        self.set_random_seed(123)
        self.embed_x.disable_update()
        self.lookup_table_sense.disable_update()
        x_block = convert.concat_examples(sent['indexed_text'],device=0,padding=-1)

        labels = sent['labels']
        if x_block.shape[1] > self.max_len: 
            x_block = x_block[:,:self.max_len]
        elif x_block.shape[1] < self.max_len:
            pdb.set_trace()

        # Make Embedding
        ex_block = self.make_input_embedding(self.embed_x, x_block)

        # Make Masks
        xx_mask = self.make_attention_mask(x_block, x_block)

        # Encode Sources

        ## ここはTRF ##
        trf_sentence_matrix = self.encoder(ex_block, xx_mask) ## 共有特徴
        y_tc = None
        y_wsd = None

        if self.model_type == "TRF-Single":
            y_tc = self.tc(trf_sentence_matrix)
        
        elif self.model_type == "TRF-Multi" or self.model_type == "TRF-Delay-Multi" or self.model_type == "TRF-Sequential":
            if self.model_type == "TRF-Sequential":
                if self.wsd_model is None:
                    y_wsd = self.wsd_only(trf_sentence_matrix,labels) ## WSD
                else:
                    y_tc = self.wsd_with_tc(sent,trf_sentence_matrix,labels) ## TC
            else: ## TRF-Multi or TRF-Delay-Multi
                y_tc,y_wsd = self.wsd_with_tc(sent,trf_sentence_matrix,labels)
   
        ## lossの計算 ##
        loss = chainer.Variable(None)
        loss_tc = chainer.Variable(None)
        loss_wsd = chainer.Variable(None)

        if get_prediction is False:
            if not((self.model_type == "TRF-Sequential") and (self.wsd_model is None)):
                if self.multi_label == 1:
                    t_txt = self.le.fit_transform(sent['doc_category'])
                    loss_tc =  F.sigmoid_cross_entropy(y_tc, self.xp.array(t_txt))
                elif self.multi_label == 0:
                    t_txt = self.le.transform(list(chain(*sent['doc_category'])))
                    loss_tc = F.softmax_cross_entropy(y_tc, self.xp.array(t_txt))

            if self.model_type == "TRF-Single":
                loss = loss_tc
            elif self.model_type == "TRF-Multi" or self.model_type == "TRF-Delay-Multi" or self.model_type == "TRF-Sequential":
                if (self.model_type == "TRF-Sequential") and not(self.wsd_model is None):
                    loss = loss_tc
                else:
                    t_wsd = self.xp.array([-1 if i == "<PAD>" else self.catgy[i] for i in list(chain(*labels))])
                    loss_wsd = F.softmax_cross_entropy(y_wsd, self.xp.array(t_wsd))

                    if self.model_type == "TRF-Multi":
                        loss = loss_tc + loss_wsd
                    elif self.model_type == "TRF-Delay-Multi":
                        if epoch < self.wsd_epoch:
                            ## WSD learning ##
                            loss = loss_wsd
                        elif epoch >= self.wsd_epoch:
                            loss = loss_tc + loss_wsd
                    elif (self.model_type == "TRF-Sequential") and (self.wsd_model is None):
                        loss = loss_wsd

            loss.backward()
            opt.update()
            return loss.data, loss_tc.data, loss_wsd.data
        else: ## get_prediction
            if "Multi" in self.model_type.split("-"):
                ## Multi or Delay-Multi
                return y_tc, y_wsd

            elif self.model_type == "TRF-Sequential":
                    if self.wsd_model is None: 
                        return y_wsd,"wsd"
                    else:
                        return y_tc,"tc"
            else:
                return y_tc


    def set_random_seed(self,seed):
        # set Python random seed
        random.seed(seed)

        # set NumPy random seed
        np.random.seed(seed)

        # set Chainer(CuPy) random seed
        self.xp.random.seed(seed)
