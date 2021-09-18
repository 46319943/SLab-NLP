from slab_nlp.topic_bert import BertTopicSLab

if __name__ == '__main__':
    docs = open(r'C:\Document\中共嘉兴\text.txt', encoding='UTF-8').readlines()
    model = BertTopicSLab('中共BERT', docs)
    # model.hierarchical_model()
    model.model()
    model.hierarchical_model()
    # model.hierarchical_compare(20, 30)
