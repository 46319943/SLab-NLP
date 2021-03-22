import pkuseg
import pymongo
from gensim import models, corpora
from gensim.models.wrappers.dtmmodel import DtmModel
from matplotlib import pyplot as plt
import math
import pandas as pd
import numpy as np
from slab.logger.base_logger import logger

# 设置画图支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 22})


def dict_corpus_comment(words_ls):
    '''
    根据词语列表生成字典与向量语料库

    :param words_ls:
    :return:
    '''
    # 字典
    dictionary = corpora.Dictionary(words_ls)
    # 通过字典转为向量语料库
    corpus = [dictionary.doc2bow(words) for words in words_ls]

    return (dictionary, corpus)


def lda_coherence(corpus, dictionary, num_topics, words_ls, passes=50, iterations=400):
    '''
    根据语料库、词典，训练LDA模型，返回LDA模型与topic coherence score
    :param corpus:
    :param dictionary:
    :param num_topics:
    :param words_ls:
    :param passes:
    :param iterations:
    :return:
    '''

    lda = models.ldamulticore.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics, eval_every=None,
                                           passes=passes,
                                           iterations=iterations)
    # lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, eval_every=None,
    #                                passes=passes,
    #                                iterations=iterations)
    cm = models.coherencemodel.CoherenceModel(model=lda, texts=words_ls, dictionary=dictionary, coherence='c_v')
    return (lda, cm.get_coherence())


def lda_topic_sensitivity(corpus, dictionary, words_ls, pass_num=50, topic_num_list=range(3, 21, 2),
                          iter_num=400) -> list:
    max_coherence = 0
    max_lda = None
    max_num = 0
    lda_coherence_list = []
    for topic_num in topic_num_list:
        lda, coherence = lda_coherence(corpus, dictionary, topic_num, words_ls, pass_num, iter_num)
        if coherence > max_coherence:
            max_coherence = coherence
            max_lda = lda
            max_num = topic_num
        lda_coherence_list.append(coherence)
        print(f'topic num: {topic_num}, coherence: {coherence}')
    plt.xticks(topic_num_list)
    plt.plot(topic_num_list, lda_coherence_list)
    return lda_coherence_list, max_coherence, max_lda, max_num


def lda_pass_sensitivity(corpus, dictionary, words_ls, pass_num_list=[20, 30, 40, 50, 60], topic_num=14, iter_num=400):
    lda_coherence_list = []
    for pass_num in pass_num_list:
        coherence = lda_coherence(corpus, dictionary, topic_num, words_ls, pass_num, iter_num)
        lda_coherence_list.append(coherence)
    plt.plot(pass_num_list, lda_coherence_list)
    return lda_coherence_list


def lda_iter_sensitivity(corpus, dictionary, words_ls, pass_num=50, topic_num=14,
                         iteration_num_list=[500, 600, 700, 800]):
    lda_coherence_list = []
    for iteration_num in iteration_num_list:
        lda, coherence = lda_coherence(corpus, dictionary, topic_num, words_ls, pass_num, iteration_num)
        lda_coherence_list.append(coherence)
    plt.plot(iteration_num_list, lda_coherence_list)
    return lda_coherence_list


def dtm_draw_topic(dtm_model: DtmModel, topic_index: int, time_num: int = None, topn=10):
    # 自动判断主题数量
    if time_num is None:
        time_num = 0
        while True:
            try:
                dtm_model.show_topic(topic_index, time_num, topn)
                time_num += 1
            except:
                break

    x = range(time_num)

    # 统计所有时间的关键词
    word_set = set()
    for time_index in range(time_num):
        for prob, word in dtm_model.show_topic(topic_index, time_index, topn):
            word_set.add(word)
    word_stat = {word: [] for word in word_set}

    # 在各个时间下，根据关键词获取频率
    max_prob = 0

    for time_index in range(time_num):
        word_dict = {word: prob for prob, word in dtm_model.show_topic(topic_index, time_index, topn)}
        for word in word_set:
            if word in word_dict:
                word_stat[word].append(word_dict[word])
                if word_dict[word] > max_prob:
                    max_prob = word_dict[word]
            else:
                word_stat[word].append(0)

    # 统计当前主题文档数量
    current_topic_doc_num = pd.Series(np.argmax(dtm_model.gamma_, axis=1)).value_counts().sort_index()[topic_index]
    total_doc_num = len(np.argmax(dtm_model.gamma_, axis=1))

    # 画图
    subplot_num = len(word_stat)
    subplot_col = 4
    subplot_row = math.ceil(float(subplot_num) / subplot_col)
    plt.figure(figsize=(4 * subplot_col, 4 * subplot_row))
    plt.suptitle(f'主题ID：{topic_index}，共{dtm_model.num_topics}个主题，当前主题文本数量：{current_topic_doc_num}/{total_doc_num}')

    for word_index, (word, prob_list) in enumerate(word_stat.items()):
        plt.subplot(subplot_row, subplot_col, word_index + 1)
        plt.plot(x, prob_list, label=word)
        plt.xticks([*range(0, x[-1], 2), x[-1]])
        plt.ylim(0, max_prob)
        plt.legend()
    plt.show()


def dtm_print_topic_all_time(dtm_model: DtmModel, topic_index, topn=10):
    time_index = 0
    while True:
        try:
            msg = dtm_model.print_topic(topic_index, time_index, topn)
            print(msg)
        except:
            return
        time_index += 1


'''

'''
