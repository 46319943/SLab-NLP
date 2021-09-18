from gensim.models.wrappers.dtmmodel import DtmModel
from .segmentation import *
from .base import *
from .topic_lda import LdaModelSLab
import math


class DtmlModelSLab():
    def __init__(self,
                 namespace: str,
                 docs: List[str],
                 time_slice: List[int]):
        '''
        初始化存储空间和分析的文本
        :param namespace:
        :param docs:
        :param time_slice: 每个时间切片对应的文本实体数量
        '''
        self.namespace = namespace
        Path(namespace).mkdir(exist_ok=True, parents=True)

        self.docs = docs
        self.time_slice = time_slice

        self.dictionary = None
        self.corpus = None

        self.topic_num = None
        self.topic_index_list = None
        self.dtm_model = None


    def model(self,
              topic_num_best: int = None,
              topic_num_list: List[int] = range(2, 22, 2)):

        pkuseg = PKUSegment()

        docs_segmented = list()
        word_segment_list = list()
        tag_segment_list = list()
        time_slice_segmented = list()

        time_doc_count_accumulate = 0
        for time_doc_count in self.time_slice:
            doc_list_part, word_segment_list_part, tag_segment_list_part = pkuseg.segment_docs(
                self.docs[time_doc_count_accumulate: time_doc_count_accumulate + time_doc_count],
                include_tag_list=['a', 'ad', 'j', 'l', 'n', 'ns', 'nt', 'nz', 'v', 'vd', 'vn'],
                min_length=2
            )
            docs_segmented.extend(doc_list_part)
            word_segment_list.extend(word_segment_list_part)
            tag_segment_list.extend(tag_segment_list_part)
            time_slice_segmented.append(len(word_segment_list_part))

            time_doc_count_accumulate += time_doc_count

        dictionary, corpus = word_segment_list_to_dictionary_corpus(word_segment_list)

        self.dictionary = dictionary
        self.corpus = corpus
        self.word_segment_list = word_segment_list
        self.tag_segment_list = tag_segment_list
        self.docs = docs_segmented

        # 有的文本在经过分词筛选之后，可能为空。该时间切片对应的文本数量就减少了
        self.time_slice = time_slice_segmented


        # 计算最佳主题数量
        if topic_num_best is None:

            lda_model = LdaModelSLab(self.namespace, docs_segmented)
            lda_model.word_segment_list = word_segment_list
            lda_model.corpus = corpus
            lda_model.dictionary = dictionary

            coherence_list, coherence_best, model_best, topic_num_best = lda_model.model_auto_select(topic_num_list)

        self.topic_num = topic_num_best

        # 训练模型
        self.dtm_model = DtmModel('dtm-win64.exe', corpus, time_slice_segmented, num_topics=topic_num_best,
                                  id2word=dictionary, initialize_lda=True,
                                  lda_sequence_min_iter=30, lda_sequence_max_iter=100,
                                  lda_max_em_iter=50
                                  )

        # 得到各文本对应主题
        self.topic_index_list = np.argmax(self.dtm_model.gamma_, axis=1)

        df = pd.DataFrame({'doc': docs_segmented, 'topic': self.topic_index_list})
        self.df = df
        return df

    def save(self):
        pickle_to_file(self, f'{self.namespace}/dtm_slab.pkl')

        # self.dtm_model.save(f'{self.namespace}/dtm_{self.topic_num}.model')
        # pickle_to_file(self.docs, f'{self.namespace}/docs.pkl')
        # pickle_to_file(self.df, f'{self.namespace}/dtm_df.pkl')

    @classmethod
    def load(cls, namespace: str):
        # docs = unpickle_from_file(f'{namespace}/docs.pkl')
        # instance = cls(namespace, docs)
        # instance.df = unpickle_from_file(f'{namespace}/dtm_df.pkl')

        instance = unpickle_from_file(f'{namespace}/dtm_slab.pkl')

        return instance

    def draw_topics(self, topn=10):
        '''
        绘制各个主题的词频图
        并统计每个主题的文档数量
        :param topn:
        :return:
        '''

        # 循环绘制每个主题
        for topic_index in range(self.topic_num):
            self.draw_topic(topic_index, topn)

        # 各主题数量
        df_topic = pd.DataFrame(np.argmax(self.dtm_model.gamma_, axis=1), columns=['topic'])
        # 聚合统计列
        df_topic.loc[:, 'count'] = 1
        df_g = df_topic.groupby('topic').size()

        df_g.barplot()
        plt.savefig(f'{self.namespace}/dtm_topic_num.png')

    def draw_topic(self, topic_index: int, topn=10):
        '''
        绘制单个主题的词频图
        :param topic_index:
        :param topn:
        :return:
        '''

        time_length = len(self.time_slice)

        x = range(time_length)

        # 统计所有时间的关键词
        word_set = set()
        for time_index in range(time_length):
            for prob, word in self.dtm_model.show_topic(topic_index, time_index, topn):
                word_set.add(word)
        word_stat = {word: [] for word in word_set}

        # 在各个时间下，根据关键词获取频率

        # 画图Y轴最大值
        max_prob = 0

        for time_index in range(time_length):
            word_dict = {word: prob for prob, word in self.dtm_model.show_topic(topic_index, time_index, topn)}
            for word in word_set:
                if word in word_dict:
                    word_stat[word].append(word_dict[word])
                    if word_dict[word] > max_prob:
                        max_prob = word_dict[word]
                else:
                    word_stat[word].append(0)

        # 统计当前主题文档数量
        current_topic_doc_num = pd.Series(
            np.argmax(self.dtm_model.gamma_, axis=1)
        ).value_counts().sort_index()[topic_index]
        total_doc_num = len(np.argmax(self.dtm_model.gamma_, axis=1))

        # 画图
        subplot_num = len(word_stat)
        subplot_col = 4
        subplot_row = math.ceil(float(subplot_num) / subplot_col)
        plt.figure(figsize=(4 * subplot_col, 4 * subplot_row))
        plt.suptitle(
            f'主题ID：{topic_index}，共{self.dtm_model.num_topics}个主题，当前主题文本数量：{current_topic_doc_num}/{total_doc_num}')

        for word_index, (word, prob_list) in enumerate(word_stat.items()):
            plt.subplot(subplot_row, subplot_col, word_index + 1)
            plt.plot(x, prob_list, label=word)
            plt.xticks([*range(0, x[-1], 2), x[-1]])
            plt.ylim(0, max_prob)
            plt.legend()

        plt.show()
        plt.savefig(f'{self.namespace}/dtm_topic{topic_index}.png')


    # TODO: 删除方法
    def print_topic_all_time_slice(self, topic_index, topn=10):
        time_index = 0
        while True:
            try:
                msg = self.dtm_model.print_topic(topic_index, time_index, topn)
                print(msg)
            except:
                return
            time_index += 1
