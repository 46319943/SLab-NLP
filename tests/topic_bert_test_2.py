from slab_nlp.topic_bert import BertTopicSLab

if __name__ == '__main__':
    docs = open(r'start.txt', encoding='UTF-8').readlines()
    docs = [doc.strip() for doc in docs if len(doc.strip()) > 5]
    model = BertTopicSLab('起点', docs)
    # model.hierarchical_model()
    model.model()
    # model.hierarchical_compare(20, 30)
