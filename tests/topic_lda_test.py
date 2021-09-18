from slab_nlp.topic_lda import LdaModelSLab

if __name__ == '__main__':
    docs = open(r'C:\Document\中共嘉兴\text.txt', encoding='UTF-8').readlines()
    model = LdaModelSLab('中共', docs)
    model.preprocess()
    model.model_auto_select()
    print()
