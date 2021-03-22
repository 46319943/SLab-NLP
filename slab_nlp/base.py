from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from slab_utils.pickle_util import pickle_to_file, unpickle_from_file

# 设置画图支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 22})

from gensim import corpora


def word_segment_list_to_dictionary_corpus(word_segment_list):
    '''
    根据词语列表生成字典与向量语料库

    :param words_ls:
    :return:
    '''
    # 字典
    dictionary = corpora.Dictionary(word_segment_list)
    # 通过字典转为向量语料库
    corpus = [dictionary.doc2bow(word_segment) for word_segment in word_segment_list]

    return dictionary, corpus


class SLabModel():
    def __init__(self, namespace, model_name):
        self.namespace = namespace
        self.model_name = model_name
        Path(namespace).mkdir(exist_ok=True, parents=True)

    def save(self):
        pickle_to_file(f'{self.namespace}/{self.model_name}.pkl')

    def load(self):
        return unpickle_from_file(f'{self.namespace}/{self.model_name}.pkl')
