import math
import collections
import numpy as np
import pdb
from collections import defaultdict
import sys
import lxml.etree as et
from tqdm import tqdm
from itertools import chain

from context2vec.common.defs import Toks
from nltk.tokenize import word_tokenize, sent_tokenize
MAX_SENTENCE_LEN = 10

def convert_numeric_data(data, batchsize, word2index,max_len):
    converted_data = {"indexed_text":[], "labels":[],"positions":[], "keys":[], "doc_category":[]}

    for doc in tqdm(sorted(data.items(),key=lambda x: x[0])):
        doc_labels = doc[1][1]
        sent = doc[1][0]
        sent_inds = []
        for word in sent["context"]:
            if word == "<PAD>":
                ind = -1
            elif word in word2index:
                ind = word2index[word]
            else:
                ind = word2index['<UNK>']
            sent_inds.append(ind)

        if len(sent_inds) > max_len:
            sent_inds = sent_inds[:max_len]
            sent['answers'] = sent['answers'][:max_len]
        elif len(sent_inds) < max_len:
            sent_inds += [-1] * (max_len - len(sent_inds))
            sent['answers'] += ['<PAD>'] * (max_len - len(sent['answers'])) 

        converted_data["indexed_text"].append(np.array(sent_inds,dtype=np.int32))
        converted_data["labels"].append(sent["answers"])
        converted_data["positions"].append(sent["positions"])
        converted_data["keys"].append(sent["keys"])
        converted_data["doc_category"].append(doc_labels.split(","))
    
    return converted_data

def convert_sentences(sentences_dic, max_sentence_len=MAX_SENTENCE_LEN):

    joint_context = []
    joint_keys = []
    joint_answers = []
    joint_positions = []

    for i,sent in enumerate(sorted(sentences_dic.items(), key=lambda x:x[0])):
        if len(sent[1]['context']) >= 1: ## 何らかの文があるとき
            joint_context.append(sent[1]['context'])
            joint_keys.append(sent[1]['keys'])
            joint_answers.append(sent[1]['answers'])
            joint_positions.append(sent[1]['positions'])
        if i == max_sentence_len - 1:
            break

    joint_context = list(chain(*joint_context))
    joint_keys = list(chain(*joint_keys))
    joint_answers = list(chain(*joint_answers))

    doc_dic = {"context":joint_context, "keys":joint_keys, "answers":joint_answers, "positions":joint_positions}
    return doc_dic, len(joint_context[0])


def load_xml(path):

    parser = et.XMLParser()
    doc = et.parse(path, parser)
    texts = doc.findall(".//text")
    word_freq = {}

    texts_dic = {}
    max_one_sentence_len = 0
    for text in tqdm(texts):

        sentences_dic = {}
        for sent in tqdm(text):
            context = [] ## lemma or raw
            positions = []
            answers = []
            keys = []
            instance_id = []
            assert sent.tag == "sentence"
            for i,child in tqdm(enumerate(sent)):
                word = child.get("lemma")
                pos = child.get("pos")
                context.append(word)
                try:
                    key = word
                except:
                    pdb.set_trace()
                if word in word_freq: ## counting lemma ##
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1

                if child.tag == "wf":
                    answers.append("<PAD>")

                elif child.tag == "instance":
                    positions.append(i)
                    keys.append(key)
                    answers.append(child.get('goldAns'))
                else:
                    raise ValueError("unknown child tag to sentence")

            sentences_dic[sent.get("id")] = {"context":context, "positions": positions, "answers":answers, "keys":keys}
        if len(sentences_dic) != 0:
            sentences_dic,sen_leng = convert_sentences(sentences_dic, max_sentence_len = MAX_SENTENCE_LEN)
            max_one_sentence_len = max(sen_leng, max_one_sentence_len)
            texts_dic[text.get("id")] = [sentences_dic,text.get("category")]

    return texts_dic, word_freq, max_one_sentence_len

class SentenceReaderDir(object):

    def __init__(self, data_path, batchsize):
        self.data_path = data_path
        self.data,self.word_freq,self.one_sentence_max_len = load_xml(data_path)
       
        self.batchsize = batchsize
        self.trimmed_word2count, self.word2index, self.index2word = self.read_and_trim_vocab()
        self.total_words = sum(self.trimmed_word2count.values())
        self.catgy = defaultdict( lambda: len(self.catgy) ) ##分類カテゴリ名とカテゴリ番号のマッピング
        self.key2sid = {} ## key(activate.v)とsense_id(カテゴリ名)のマッピング
        self.key2netout = {} ## keyとネットワークの出力層のユニット番号の対応
        self.senseid2netout = {}
        self.doc_catgy = defaultdict( lambda: len(self.doc_catgy) ) ##文書分類用のカテゴリ
        self.make_catgy()
        self.make_key2netout()
        self.add_wsdtag_to_vocab()


    def make_catgy(self):

        for doc in tqdm(sorted(self.data.items(),key=lambda x: x[0])):
            labels = doc[1][1]
            sents = doc[1][0]
            [self.doc_catgy[label] for label in labels.split(",")]
            try:
                if len(sents['keys']) > 0:
                    keys = sents['keys']
                    answers = sents['answers']
                    answers = [i for i in answers if i != "<PAD>"]
                    try:
                        assert len(keys) == len(answers)
                    except:
                        pdb.set_trace()
                    for key,ans in zip(keys,answers):
                        if ans == "<PAD>":
                            continue 
                        elif not key in self.key2sid:
                            self.key2sid[key] = [ans]
                        elif (key in self.key2sid) and (not ans in self.key2sid[key]):
                            self.key2sid[key].append(ans)
            except:
                pdb.set_trace()
            [self.catgy[sid] for sid in sents['answers'] if sid != "<PAD>"]

        if len(self.doc_catgy) != 3:
            all_labels = ["GENT","GSCI","GPOL"]
            for l in all_labels:
                self.doc_catgy[l]


    def make_key2netout(self):
        for k,v in self.key2sid.items():
            self.key2netout[k] = [self.catgy[_v] for _v in v]
            for _v in v:
                self.senseid2netout[_v] = self.key2netout[k]



    def read_and_trim_vocab(self, trimfreq=0):
        word2count = self.word_freq
        trimmed_word2count = collections.Counter()
        index2word = {Toks.UNK:'<UNK>', Toks.BOS:'<BOS>', Toks.EOS:'<EOS>'}
        word2index = {'<UNK>':Toks.UNK, '<BOS>':Toks.BOS, '<EOS>':Toks.EOS}
        unknown_counts = 0

        for word, count in sorted(word2count.items()):
            if count >= trimfreq and word.lower() != '<unk>' and word.lower() != '<rw>':    
                ind = len(word2index)
                word2index[word] = ind
                index2word[ind] = word
                trimmed_word2count[ind] = count
            else:
                unknown_counts += count
        trimmed_word2count[word2index['<BOS>']] = 0
        trimmed_word2count[word2index['<EOS>']] = 0
        trimmed_word2count[word2index['<UNK>']] = unknown_counts


        return trimmed_word2count, word2index, index2word

    def add_wsdtag_to_vocab(self):
        for wsdtag in self.catgy.keys():
            ind = len(self.word2index)
            self.word2index[wsdtag] = ind
            self.index2word[ind] = wsdtag


