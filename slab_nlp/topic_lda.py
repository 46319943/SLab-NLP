from .base import *
from .segmentation import PKUSegment
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel

from typing import List


class LdaModelSLab():
    def __init__(self,
                 namespace: str,
                 docs: List[str]):
        '''
        初始化存储空间和分析的文本
        :param namespace:
        :param docs: 文本列表，每个文本是一个实体
        '''
        self.namespace = namespace
        Path(namespace).mkdir(exist_ok=True, parents=True)
        self.docs = docs

    def preprocess(self):
        '''
        分词、词性标注、字典和语料库生成
        结果存入成员变量
        :return:
        '''

        # 分词、词性标注
        self.segment = PKUSegment()
        doc_list, word_segment_list, tag_segment_list = self.segment.segment_docs(self.docs)

        self.docs = doc_list
        self.word_segment_list = word_segment_list
        self.tag_segment_list = tag_segment_list

        # 字典、语料库生成
        self.dictionary, self.corpus = word_segment_list_to_dictionary_corpus(word_segment_list)

    def model(self, num_topics, passes=50, iterations=400):
        '''
        根据成员变量中的语料库、词典，训练LDA模型
        返回LDA模型与topic coherence score
        :param corpus:
        :param dictionary:
        :param num_topics:
        :param docs:
        :param passes:
        :param iterations:
        :return:
        '''

        model = LdaMulticore(corpus=self.corpus, id2word=self.dictionary, num_topics=num_topics, eval_every=None,
                             passes=passes, iterations=iterations)
        cm = CoherenceModel(model=model, texts=self.word_segment_list, dictionary=self.dictionary, coherence='c_v')
        return model, cm.get_coherence()

    def model_auto_select(self, topic_num_list=range(3, 21, 2),
                          pass_num=50, iter_num=400):
        '''
        训练多个LDA模型，在不同主题数量中选择最优模型
        :param topic_num_list:
        :param pass_num:
        :param iter_num:
        :return:
        '''

        # 得分列表
        coherence_list = []
        coherence_best = 0
        topic_num_best = 0

        model_best = None

        for topic_num in topic_num_list:
            model, coherence = self.model(topic_num, pass_num, iter_num)
            if coherence > coherence_best:
                coherence_best = coherence
                model_best = model
                topic_num_best = topic_num
            coherence_list.append(coherence)
            print(f'topic num: {topic_num}, coherence: {coherence}')

        # 存储结果到成员变量
        self.coherence_list = coherence_list
        self.coherence_best = coherence_best
        self.topic_num_best = topic_num_best

        self.model_best = model_best

        # 绘制主题数量与得分图表
        plt.xticks(topic_num_list)
        plt.plot(topic_num_list, coherence_list)
        plt.savefig(f'{self.namespace}/coherence.png')

        return coherence_list, coherence_best, model_best, topic_num_best




    # TODO: 修改灵敏度代码
    def lda_pass_sensitivity(self, pass_num_list=[20, 30, 40, 50, 60], topic_num=14,
                             iter_num=400):
        lda_coherence_list = []
        for pass_num in pass_num_list:
            coherence = self.model(topic_num, pass_num, iter_num)
            lda_coherence_list.append(coherence)
        plt.plot(pass_num_list, lda_coherence_list)
        return lda_coherence_list

    def lda_iter_sensitivity(self, pass_num=50, topic_num=14,
                             iteration_num_list=[500, 600, 700, 800]):
        lda_coherence_list = []
        for iteration_num in iteration_num_list:
            lda, coherence = self.model(topic_num, pass_num, iteration_num)
            lda_coherence_list.append(coherence)
        plt.plot(iteration_num_list, lda_coherence_list)
        return lda_coherence_list
