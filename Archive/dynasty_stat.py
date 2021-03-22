import pandas as pd
from dianping_nlp import *
from slab.pickle_util import pickle_to_file, unpickle_from_file
from dynasty_extract import *

dynasty_ordered_list = ['现代', '清朝', '明朝', '元朝', '宋朝', '唐朝', '隋朝',
                        '魏晋南北朝', '三国', '汉代', '秦代', '春秋战国', '西周', '商代', '夏代', '黄帝', ]


def df_to_dummies(df: pd.DataFrame) -> pd.DataFrame:
    dynasty_str_list = []
    for loc_list in df['line_time_result'].values:
        dynasty_str_list.append(
            ','.join(dynasty_extract(loc_list))
        )

    df['dynasty_str'] = dynasty_str_list
    df_dummies = pd.concat([df, df['dynasty_str'].str.get_dummies(sep=',')], axis='columns')

    return df_dummies


def dummies_word_count(df_dummies: pd.DataFrame, dynasty_name: str = '现代'):
    return pd.Series(
        [word
         for line_result in df_dummies[df_dummies[dynasty_name] == 1]['line_result'].values
         for word, tag in line_result
         if len(word) > 1
         ]
    ).value_counts()


def dynasty_lda(df_dummies: pd.DataFrame, dynasty_name: str = '现代', topic_num=None):
    words_ls = [
        [
            word for word, tag in line_result if len(word) > 1
        ] for line_result in df_dummies[df_dummies[dynasty_name] == 1]['line_result'].values
    ]
    dictionary, corpus = dict_corpus_comment(
        words_ls
    )
    if topic_num is None:
        topic_num_list = range(3, 33, 2)
        lda_coherence_list, coherence, lda, topic_num = lda_topic_sensitivity(corpus, dictionary, words_ls,
                                                                              topic_num_list=topic_num_list)
        # topic_num = topic_num_list[np.argmax(lda_coherence_list)]
        pickle_to_file(list(zip(topic_num_list, lda_coherence_list)), f'lda_coherence_{dynasty_name}.pkl')
    else:
        lda, coherence = lda_coherence(corpus, dictionary, topic_num, words_ls)
    lda.save(f'lda_{dynasty_name}_{topic_num}.model')

    return lda, corpus, dictionary


def dynasty_dtm(df_dummies: pd.DataFrame, topic_num: int = None):
    # 根据时间节点分割文档
    word_slice_num = []
    word_piece_total = []
    for dynasty_name in dynasty_ordered_list[::-1]:
        word_piece = [
            [
                word for word, tag in line_result if len(word) > 1
            ] for line_result in df_dummies[df_dummies[dynasty_name] == 1]['line_result'].values
        ]
        word_slice_num.append(len(word_piece))
        word_piece_total.extend(word_piece)

    dictionary, corpus = dict_corpus_comment(word_piece_total)

    # 计算最佳主题数量
    if topic_num is None:
        topic_num_list = range(2, 123, 5)
        lda_coherence_list, max_coherence, max_lda, max_num = lda_topic_sensitivity(corpus, dictionary,
                                                                                    word_piece_total,
                                                                                    topic_num_list=topic_num_list)
        topic_num = topic_num_list[np.argmax(lda_coherence_list)]
        pickle_to_file(list(zip(topic_num_list, lda_coherence_list)), f'coherence_{"全部朝代时间序列"}.pkl')

    # 训练模型
    dtm_model = DtmModel('dtm-win64.exe', corpus, word_slice_num, num_topics=topic_num,
                         id2word=dictionary, initialize_lda=True,
                         lda_sequence_min_iter=30, lda_sequence_max_iter=100,
                         lda_max_em_iter=50
                         )

    dtm_model.save(f'dtm_{"全部朝代时间序列"}_{topic_num}.model')

    # 得到各文本对应主题
    topic_index_list = np.argmax(dtm_model.gamma_, axis=1)
    for index, dynasty_name in enumerate(dynasty_ordered_list[::-1]):
        slice_num = word_slice_num[index]
        df_dummies.loc[df_dummies[dynasty_name] == 1, dynasty_name + 'topic_index'] = topic_index_list[0:slice_num]
        topic_index_list = topic_index_list[slice_num:]

    pickle_to_file(df_dummies, f'df_{"全部朝代时间序列"}_{topic_num}.pkl')


if __name__ == '__main__':
    df = pd.DataFrame(unpickle_from_file('df.pkl'))
    df_dummies = df_to_dummies(df)

    # for dynasty_name in dynasty_ordered_list:
    #     dynasty_lda(df_dummies, dynasty_name)

    # dynasty_lda(df_dummies, '秦代', 4)

    dynasty_dtm(df_dummies, )

    print()
