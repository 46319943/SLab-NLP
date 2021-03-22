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
from typing import List


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
            # umap_model=umap_model,
            # hdbscan_model=hdbscan_model,
            vectorizer_model=self.vectorizer_model,
        )

        # sentence to vector and reduce dimension
        self.sentence_embeddings = self.sentence_model.encode(self.docs)
        self.umap_embeddings = UMAP(n_neighbors=15,
                                    n_components=5,
                                    min_dist=0.0,
                                    metric='cosine').fit(self.sentence_embeddings).transform(self.sentence_embeddings)

    def hierarchical_model(self):
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(self.umap_embeddings)

        self.hierarchical_linkage_matrix = plot_dendrogram(model, truncate_mode='level', p=3)
        plt.savefig(f"{self.namespace}/hierarchical.png", format="PNG")
        self.hierarchical_distance = model.distances_

        return

    def extract_topics_by_c_tf_idf(self, documents: pd.DataFrame):
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        c_tf_idf, words = self._c_tf_idf(documents_per_topic, m=len(documents))
        topics = self._extract_words_per_topic(words, c_tf_idf)
        return topics

    def hierarchical_compare(self, distance1, distance2, sample_size=5):
        distance_min = min(distance1, distance2)
        distance_max = max(distance1, distance2)

        # smaller distance for more cluster
        model_large = AgglomerativeClustering(distance_threshold=distance_min, n_clusters=None)
        model_large.fit(self.umap_embeddings)
        # larger distancce for less cluster
        model_small = AgglomerativeClustering(distance_threshold=distance_max, n_clusters=None)
        model_small.fit(self.umap_embeddings)

        # 使用C-TF-IDF提取大类、小类的关键词
        topics_large = self.extract_topics_by_c_tf_idf(
            pd.DataFrame({'Document': self.docs, 'Topic': model_large.labels_})
        )
        topics_small = self.extract_topics_by_c_tf_idf(
            pd.DataFrame({'Document': self.docs, 'Topic': model_small.labels_})
        )

        for cluster_index in range(model_small.n_clusters_):
            mapping_from_index_list = np.unique(
                model_large.labels_[model_small.labels_ == cluster_index]
            )
            if len(mapping_from_index_list) > 1:
                for mapping_from_index in mapping_from_index_list:
                    mapping_from_count = np.count_nonzero(model_large.labels_ == mapping_from_index)
                    mapping_from_docs = np.array(self.docs)[model_large.labels_ == mapping_from_index]
                    mapping_from_docs_chioce = np.random.choice(mapping_from_docs, sample_size)

                    print(f'from cluster {mapping_from_index}({mapping_from_count}):\n')
                    print(''.join(mapping_from_docs_chioce))
                    print(topics_large[mapping_from_index])
                    print()

                mapping_to_count = np.count_nonzero(model_small.labels_ == cluster_index)
                print(f'to cluster {cluster_index}({mapping_to_count})')
                print(topics_small[cluster_index])

                # print(
                #     f'{", ".join([str(mapping_from_index) + "(" + str(np.count_nonzero(model_large.labels_ == mapping_from_index)) + ")" for mapping_from_index in mapping_from_index_list])} -> {cluster_index}'
                # )

    def model_time_series(self, timestamps: List[int]):
        if self.topic_index is None:
            self.model()

        topics_over_time = self.topics_over_time(self.docs, self.topic_index, timestamps)
        fig = self.visualize_topics_over_time(topics_over_time, top_n=10)
        fig.write_html(f'{self.namespace}/BERTopic_time_series.html')

    def model(self):

        # 保证输入文档维度与嵌入维度相同
        # docs = [doc for doc in docs if len(doc) > 10]

        self.topic_index, _ = self.fit_transform(self.docs, self.sentence_embeddings)
        self.visualize_topics().write_html(f"{self.namespace}/BERTopic_vis.html")

    def model_network(self, distance_threshold=0.5):
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

    def __getstate__(self):
        attributes = self.__dict__.copy()
        del attributes['segment_model']
        del attributes['sentence_model']
        return attributes
