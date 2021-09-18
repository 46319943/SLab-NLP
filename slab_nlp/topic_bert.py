from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import pkuseg
import networkx as nx
from .base import *
from .visualization import draw_word_freq_over_time
from typing import List
from tqdm import tqdm
from contextlib import redirect_stdout


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

    return linkage_matrix


class BertTopicSLab(BERTopic):
    def __init__(self,
                 namespace: str,
                 docs: List[str]):
        self.namespace = namespace
        Path(namespace).mkdir(exist_ok=True, parents=True)

        # 初始化各部分模型参数
        self.docs = docs
        self.segment_model = pkuseg.pkuseg(postag=True)

        self.sentence_model = SentenceTransformer("stsb-xlm-r-multilingual", device="cpu")
        self.umap_model = UMAP(n_neighbors=15, n_components=10, min_dist=0.0, metric='cosine')
        self.hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom',
                                     prediction_data=True)
        self.vectorizer_model = CountVectorizer(tokenizer=lambda text: [
            word for word, tag in self.segment_model.cut(text) if len(word) > 1
        ], token_pattern=None)

        # 调用父类构造函数
        super(BertTopicSLab, self).__init__(
            embedding_model=self.sentence_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
        )

        # sentence to vector and reduce dimension
        self.sentence_embeddings = self.sentence_model.encode(self.docs)
        self.umap_embeddings = UMAP(n_neighbors=15,
                                    n_components=5,
                                    min_dist=0.0,
                                    metric='cosine').fit(self.sentence_embeddings).transform(self.sentence_embeddings)

    def hierarchical_model(self,
                           compare_count: int = 10
                           ):
        '''
        根据UMAP降维后的距离进行层次聚类
        :param compare_count:
        :return:
        '''
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(self.umap_embeddings)

        # 绘制树形图
        plt.clf()
        self.hierarchical_linkage_matrix = plot_dendrogram(model, truncate_mode='level', p=3)
        plt.savefig(f"{self.namespace}/hierarchical.png", format="PNG")
        self.hierarchical_distance = model.distances_

        # compare_distance_list = self.hierarchical_distance[-compare_count:]
        # for distance1, distance2 in zip(compare_count[:-1], compare_count[1:]):
        #     self.hierarchical_compare(
        #         distance1, distance2)

        origin_labels = AgglomerativeClustering(n_clusters=compare_count + 1).fit(
            self.umap_embeddings).labels_
        for cluster_count in range(compare_count, 0, -1):
            # self.hierarchical_compare(cluster1=cluster_count, cluster2=cluster_count - 1)
            origin_labels = self.hierarchical_compare_single(cluster_count, origin_labels)
        return

    def extract_topics_by_c_tf_idf(self, documents: pd.DataFrame):
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        c_tf_idf, words = self._c_tf_idf(documents_per_topic, m=len(documents))
        topics = self._extract_words_per_topic(words, c_tf_idf, documents_per_topic['Topic'].values)
        return topics

    def hierarchical_compare(self,
                             distance1=None, distance2=None,
                             cluster1=None, cluster2=None,
                             sample_size=5, ):
        '''

        :param distance1:
        :param distance2:
        :param cluster1:
        :param cluster2:
        :param sample_size:
        :return:
        '''
        if distance1 and distance2:
            distance_min = min(distance1, distance2)
            distance_max = max(distance1, distance2)

            # smaller distance for more cluster
            model_large = AgglomerativeClustering(distance_threshold=distance_min, n_clusters=None)
            model_large.fit(self.umap_embeddings)
            # larger distancce for less cluster
            model_small = AgglomerativeClustering(distance_threshold=distance_max, n_clusters=None)
            model_small.fit(self.umap_embeddings)

        if cluster1 and cluster2:
            cluster_min = min(cluster1, cluster2)
            cluster_max = max(cluster1, cluster2)

            # smaller distance for more cluster
            model_large = AgglomerativeClustering(n_clusters=cluster_max)
            model_large.fit(self.umap_embeddings)
            # larger distancce for less cluster
            model_small = AgglomerativeClustering(n_clusters=cluster_min)
            model_small.fit(self.umap_embeddings)

        # 使用C-TF-IDF提取大类、小类的关键词
        topics_large = self.extract_topics_by_c_tf_idf(
            pd.DataFrame({'Document': self.docs, 'Topic': model_large.labels_})
        )
        topics_small = self.extract_topics_by_c_tf_idf(
            pd.DataFrame({'Document': self.docs, 'Topic': model_small.labels_})
        )

        # 将结果写入文件
        with open(f'{self.namespace}/hierarchical_cluter_info.txt', 'a') as f:
            with redirect_stdout(f):

                for cluster_index in range(model_small.n_clusters_):
                    mapping_from_index_list = np.unique(
                        model_large.labels_[model_small.labels_ == cluster_index]
                    )
                    if len(mapping_from_index_list) > 1:
                        for mapping_from_index in mapping_from_index_list:
                            mapping_from_count = np.count_nonzero(model_large.labels_ == mapping_from_index)
                            mapping_from_docs = np.array(self.docs)[model_large.labels_ == mapping_from_index]
                            mapping_from_docs_chioce = np.random.choice(mapping_from_docs, sample_size, False)

                            print(f'from cluster {mapping_from_index}({mapping_from_count}):\n')
                            print('\n'.join([doc.strip() for doc in mapping_from_docs_chioce]))
                            print(topics_large[mapping_from_index])
                            print()

                        mapping_to_count = np.count_nonzero(model_small.labels_ == cluster_index)
                        print(f'to cluster {cluster_index}({mapping_to_count})')
                        print(topics_small[cluster_index])
                        print()

        return model_large.labels_

        # print(
        #     f'{", ".join([str(mapping_from_index) + "(" + str(np.count_nonzero(model_large.labels_ == mapping_from_index)) + ")" for mapping_from_index in mapping_from_index_list])} -> {cluster_index}'
        # )

    def hierarchical_compare_single(self, cluster_num, origin_labels,
                                    sample_size=5):
        '''
        将聚类簇数量减少到指定数量，并且得到多到一的映射关系

        原始标签包含聚类号数量大于新聚类的聚类号数量
        将原始标签对应聚类成为大类
        将聚类的聚类号成为小类

        过程如下：
        遍历小类聚类编号
        其中，对于每个小类聚类编号，取其对应位置下的大类聚类编号，从而得到当前小类聚类编号对应的大类聚类编号
        即小类编号是从哪些大类编号上来的，于是得到大类到小类编号的多对一映射关系

        最后，将大类中的多个大类编号，根据每对多对一的关系，将多个映射成一个（多个中的第一个）

        :param cluster_num:
        :param origin_labels:
        :param sample_size: 每个聚类取样文档数量
        :return:
        '''

        # 原始聚类标签
        origin_labels = np.array(origin_labels).copy()

        # 根据指定数量纪念性聚类
        model = AgglomerativeClustering(n_clusters=cluster_num)
        model.fit(self.umap_embeddings)

        # 使用C-TF-IDF提取大类（原始聚类）、小类（指定数量聚类）的关键词
        topics_large = self.extract_topics_by_c_tf_idf(
            pd.DataFrame({'Document': self.docs, 'Topic': origin_labels})
        )
        topics_small = self.extract_topics_by_c_tf_idf(
            pd.DataFrame({'Document': self.docs, 'Topic': model.labels_})
        )

        # 记录映射关系，从而在保持各类标签顺序不变的情况下，生成映射后的标签
        mapping_pairs_info = list()

        # 将结果写入文件
        with open(f'{self.namespace}/hierarchical_cluter_info.txt', 'a') as f:
            with redirect_stdout(f):

                # 小类（指定数量聚类）的聚类号
                for cluster_index in range(model.n_clusters_):
                    # 当前小类聚类号对应的大类聚类号
                    mapping_from_index_list = np.unique(
                        # 选择当前聚类号在大类（原始聚类）中对应位置下的大类聚类号
                        origin_labels[
                            # 选择当前聚类号对应的索引位置
                            model.labels_ == cluster_index
                            ]
                    )
                    if len(mapping_from_index_list) > 1:

                        mapping_pairs_info.append(mapping_from_index_list)

                        for mapping_from_index in mapping_from_index_list:
                            mapping_from_count = np.count_nonzero(origin_labels == mapping_from_index)
                            mapping_from_docs = np.array(self.docs)[origin_labels == mapping_from_index]
                            mapping_from_docs_chioce = np.random.choice(mapping_from_docs, sample_size, False)

                            print(f'from cluster {mapping_from_index}({mapping_from_count}):\n')
                            print('\n'.join([doc.strip() for doc in mapping_from_docs_chioce]))
                            print(topics_large[mapping_from_index])
                            print()

                        mapping_to_count = np.count_nonzero(model.labels_ == cluster_index)
                        print(f'to cluster {cluster_index}-{mapping_from_index_list[0]}({mapping_to_count})')
                        print(topics_small[cluster_index])
                        print()

        # 将原始标签根据映射关系进行映射
        for mapping_pairs in mapping_pairs_info:
            mapping_pair_to = mapping_pairs[0]
            for mapping_pair_from in mapping_pairs[1:]:
                origin_labels[origin_labels == mapping_pair_from] = mapping_pair_to

        return origin_labels

    def model_time_series(self, timestamps: List[int] = None, time_slice: List[int] = None):
        if timestamps is None and time_slice is not None:
            timestamps = time_slice_to_timestamps(time_slice)

        if self.topic_index is None:
            self.model()

        topics_over_time = self.topics_over_time(self.docs, self.topic_index, timestamps)
        fig = self.visualize_topics_over_time(topics_over_time, top_n=10)
        fig.write_html(f'{self.namespace}/BERTopic_time_series.html')

        df = self.topics_keyword_fre_over_time(self.docs, self.topic_index, timestamps)
        draw_word_freq_over_time(df, save_dir=self.namespace)

    def model(self):

        # 保证输入文档维度与嵌入维度相同
        # docs = [doc for doc in docs if len(doc) > 10]

        self.topic_index, _ = self.fit_transform(self.docs, self.sentence_embeddings)
        self.visualize_topics().write_html(f"{self.namespace}/BERTopic_vis.html")

    def model_network(self, distance_threshold=0.5):
        '''
        每个节点代表文档
        根据距离阈值，将距离接近的文档连接到一起
        连接到一起的文档必须有对应主题序号，并且不能为同一主题
        即连接不同主题之间距离相近的文档
        :param distance_threshold:
        :return:
        '''

        if self.topic_index is None:
            self.model()

        df = pd.DataFrame(
            {'text': self.docs, 'topic': self.topic_index}
        )
        df = df.reset_index()

        # 创建图
        G = nx.Graph()
        # 根据ID添加节点
        G.add_nodes_from(df.index.tolist())

        # 根据umap降维结果计算距离矩阵
        distance_matrix = pairwise_distances(self.umap_embeddings, metric='minkowski', p=2)
        # 根据距离矩阵添加边
        for row in range(distance_matrix.shape[0]):
            for column in range(distance_matrix.shape[1]):
                # Upper matrix
                if row >= column:
                    continue

                distance = distance_matrix[row, column]

                # 过滤非聚类结果
                if self.topic_index[row] == -1 or self.topic_index[column] == -1:
                    continue

                # 过滤同类结果
                if self.topic_index[row] == self.topic_index[column]:
                    continue

                if distance < distance_threshold:
                    G.add_edge(row, column, weight=distance)
                    print(f'add edge {row} {column}')

        from pyvis.network import Network

        net = Network(notebook=True)
        net.from_nx(G)
        net.show(f'{self.namespace}/network_vis.html')

    def topics_keyword_fre_over_time(self,
                                     docs: List[str],
                                     topics: List[int],
                                     timestamps: Union[List[int]],
                                     topn: int = 10
                                     ) -> pd.DataFrame:

        documents = pd.DataFrame({"Document": docs, "Topic": topics, "Timestamps": timestamps})

        # Sort documents in chronological order
        documents = documents.sort_values("Timestamps")
        timestamps = documents.Timestamps.unique()

        # For each unique timestamp, create topic representations
        topics_over_time = []
        for index, timestamp in tqdm(enumerate(timestamps), disable=not self.verbose):
            # Calculate c-TF-IDF representation for a specific timestamp
            selection = documents.loc[documents.Timestamps == timestamp, :]
            documents_per_topic = selection.groupby(['Topic'], as_index=False).agg({'Document': ' '.join,
                                                                                    "Timestamps": "count"})
            c_tf_idf, words = self._c_tf_idf(documents_per_topic, m=len(selection), fit=False)

            # Extract the words per topic
            labels = sorted(list(documents_per_topic.Topic.unique()))
            words_per_topic = self._extract_words_per_topic(words, c_tf_idf, labels)
            topic_frequency = pd.Series(documents_per_topic.Timestamps.values,
                                        index=documents_per_topic.Topic).to_dict()

            # Fill dataframe with results
            topics_at_timestamp = [
                {
                    'topic': topic,
                    'timestamp': timestamp,
                    'word': word,
                    'frequency': freq,
                }
                for topic, values in words_per_topic.items()
                for word, freq in values[:topn]
            ]
            topics_over_time.extend(topics_at_timestamp)

        return pd.DataFrame(topics_over_time)

    def __getstate__(self):
        attributes = self.__dict__.copy()
        del attributes['segment_model']
        del attributes['sentence_model']
        for key in list(attributes.keys()):
            if key.startswith('_'):
                del attributes[key]
        return attributes
