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

from dianping_text import filter_text


def query_comment(start_time_str='2020-02', end_time_str='2020-04'):
    '''
    根据时间获取评论文本
    :param start_time_str:
    :param end_time_str:
    :return:
    '''

    # 连接数据库
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    dianping_db = client["dianping"]
    comment_collection = dianping_db['comment']

    doc_list = []

    for comment in comment_collection.find({
        'time': {'$gt': start_time_str, '$lt': end_time_str},
        'timestamp': {'$ne': 'NaN'}
    }, {'text': 1, '_id': 0}):
        doc = comment['text']
        doc_list.append(doc)

    return doc_list


def filter_comment(doc_list):
    '''
    对非法的符号、英文、表情进行过滤
    :param doc_list:
    :return:
    '''

    doc_list_result = []
    for doc in doc_list:
        doc = filter_text(doc)
        if doc is not None:
            doc_list_result.append(doc)
    return doc_list_result


def is_preserved(word, tag, debug=False):
    '''

    :param word:
    :param tag:
    :param debug:
    :return:
    '''
    if debug:
        return True

    word_type_flags = ['a', 'ad', 'j', 'l', 'n', 'ns', 'nt', 'nz', 'v', 'vd', 'vn']
    stop_words = (
        '湖北', '湖北省', '武汉', '武汉市', '长江', '黄鹤楼', '景点', '景区', '交通',
        '觉得', '直接', '可以',
        '还是', '还有', '真的', '非常', '一定', '时候', '现在', '没有', '就是', '小时'

    )

    if tag not in word_type_flags:
        return False
    if word in stop_words:
        return False

    # 去掉一个字的词语
    if len(word) <= 1:
        return False

    return True


def segment_comment(doc_list):
    '''
    对文本进行分词
    返回分词后的词语、词性标记

    :param doc_list:
    :return:
    '''

    # 加载分词模型
    segment_model = pkuseg.pkuseg(postag=True)

    # 单词列表，(n, m)
    words_ls = []
    # 词性列表，(n, m)
    tags_ls = []
    # 单词、词性列表，(n, m, 2)
    words_tags_ls = []
    # 是否保留的标志
    mask_list = []

    for doc in doc_list:
        # (m, 2)
        words_tags = [[word, tag] for word, tag in segment_model.cut(doc) if is_preserved(word, tag, debug=False)]
        # (2, m)。相当于np矩阵转置
        try:
            words, tags = zip(*words_tags)
            mask_list.append(True)
        except:
            # 出现异常说明，分词后为空
            logger.exception(doc)
            mask_list.append(False)
            continue

        words_ls.append(words)
        tags_ls.append(tags)
        words_tags_ls.append(words_tags)

    return words_ls, tags_ls, words_tags_ls, mask_list


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


def lda_coherence(corpus, dictionary, num_topics, words_ls, passes, iterations):
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


def dtm(time_query_list, topic_num):
    '''
    根据查询时间列表训练DTM模型
    返回词典、语料库、模型

    :param time_query_list:
    :param topic_num:
    :return:
    '''
    words_slice = []
    total_words_ls = []
    for time_query in time_query_list:
        words_ls, _, _ = segment_comment(query_comment(*time_query))

        words_slice.append(len(words_ls))
        total_words_ls.extend(words_ls)

    dictionary, corpus = dict_corpus_comment(total_words_ls)

    dtm_model = DtmModel('dtm-win64.exe', corpus, words_slice, num_topics=topic_num,
                         id2word=dictionary, initialize_lda=True,
                         lda_sequence_min_iter=30, lda_sequence_max_iter=100,
                         lda_max_em_iter=50
                         )

    return total_words_ls, dictionary, corpus, dtm_model


def dtm_seq(time_query_list, topic_num):
    '''
    根据查询时间列表训练DTM模型
    返回词典、语料库、模型

    :param time_query_list:
    :param topic_num:
    :return:
    '''
    words_slice = []
    total_words_ls = []
    for time_query in time_query_list:
        words_ls, _, _ = segment_comment(query_comment(*time_query))

        words_slice.append(len(words_ls))
        total_words_ls.extend(words_ls)

    dictionary, corpus = dict_corpus_comment(total_words_ls)

    dtm_model = models.LdaSeqModel(corpus, words_slice, num_topics=topic_num,
                                   id2word=dictionary,
                                   # em_min_iter =30, em_max_iter =100,
                                   # lda_inference_max_iter =50
                                   )

    return total_words_ls, dictionary, corpus, dtm_model


def lda_topic_sensitivity(corpus, dictionary, words_ls, pass_num=50, topic_num_list=range(3, 21, 2),
                          iter_num=400) -> list:
    lda_coherence_list = []
    for topic_num in topic_num_list:
        lda, coherence = lda_coherence(corpus, dictionary, topic_num, words_ls, pass_num, iter_num)
        lda_coherence_list.append(coherence)
        print(f'topic num: {topic_num}, coherence: {coherence}')
    plt.xticks(topic_num_list)
    plt.plot(topic_num_list, lda_coherence_list)
    return lda_coherence_list


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


def dtm_draw_topic(dtm_model: DtmModel, topic_index, time_num=None, topn=10):
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
        try:
            word_dict = {word: prob for prob, word in dtm_model.show_topic(topic_index, time_index, topn)}
        except:
            break
        for word in word_set:
            if word in word_dict:
                word_stat[word].append(word_dict[word])
                if word_dict[word] > max_prob:
                    max_prob = word_dict[word]
            else:
                word_stat[word].append(0)

    # 画图
    subplot_num = len(word_stat)
    subplot_col = 4
    subplot_row = math.ceil(float(subplot_num) / subplot_col)
    plt.figure(figsize=(4 * subplot_col, 4 * subplot_row))
    for word_index, (word, prob_list) in enumerate(word_stat.items()):
        plt.subplot(subplot_row, subplot_col, word_index + 1)
        plt.plot(x, prob_list, label=word)
        plt.ylim(0, max_prob)
        plt.legend()


def dtm_df(dtm_model: DtmModel, words_ls):
    # 各主题数量
    df_topic = pd.DataFrame(np.argmax(dtm_model.gamma_, axis=1), columns=['topic'])
    # 聚合统计列
    df_topic.loc[:, 'count'] = 1
    df_g = df_topic.groupby('topic').count()

    df_words = pd.DataFrame()
    df_words['words'] = words_ls
    df_words['topic'] = df_topic['topic']


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
n   名词
t   时间词
s   处所词
f   方位词
m   数词
q   量词
b   区别词
r   代词
v   动词
a   形容词
z   状态词
d   副词
p   介词
c   连词
u   助词
y   语气词
e   叹词
o   拟声词
i   成语
l   习惯用语
j   简称
h   前接成分
k   后接成分
g   语素
x   非语素字
w   标点符号
nr  人名
ns  地名
nt  机构名称
nx  外文字符
nz  其它专名
vd  副动词
vn  名动词
vx  形式动词
ad  副形词
an  名形词
'''
