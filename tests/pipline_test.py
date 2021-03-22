from slab_nlp.topic_bert import BertTopicSLab
from slab_nlp.topic_dtm import DtmlModelSLab
from slab_nlp.sentiment import PaddleSkepSentiment
from slab_utils.pickle_util import pickle_to_file, unpickle_from_file

import re


def year_process(docs):
    docs_postprocess = list()

    doc_counter = 0
    doc_time_slice = list()

    for doc in docs:
        match_result = re.match(r'(\d{4})年', doc)
        if match_result is not None and len(doc) <= 6:
            year_str = match_result.group(1)
            year_num = int(year_str)
            print(year_num)

            doc_time_slice.append(doc_counter)
            doc_counter = 0

        else:
            docs_postprocess.append(doc)
            doc_counter += 1

    # 循环完之后，处理剩余计数
    if doc_counter != 0:
        doc_time_slice.append(doc_counter)
        doc_counter = 0

    return docs_postprocess, doc_time_slice


if __name__ == '__main__':
    import numpy as np

    docs = open(r'C:\Document\中共嘉兴\text.txt', encoding='UTF-8').readlines()
    docs, time_slice = year_process(docs)

    # 两年合并为一年
    time_slice = [array.sum() for array in np.array_split(np.array(time_slice), 10)]

    # DTM模型
    model = DtmlModelSLab('中共DTM', docs, time_slice)
    df = model.model(11)

    docs = model.docs
    time_slice = model.time_slice

    timestamps = []
    for time_index, time_count in enumerate(time_slice):
        for i in range(time_count):
            timestamps.append(time_index)

    # 地名
    location_rec_list = [
        [word for word, tag in zip(word_segment, tag_segment) if tag in ['ns', 'nt']]
        for word_segment, tag_segment in zip(model.word_segment_list, model.tag_segment_list)
    ]

    df['location'] = location_rec_list

    # 情感分析
    sentiment = PaddleSkepSentiment()
    df['sentiment'] = sentiment.sentiment_docs(df['doc'])

    model.df = df
    model.save()

    bert_topic = BertTopicSLab('中共Bert', docs)
    bert_topic.model_time_series(timestamps)

    print()
