from slab_nlp.topic_bert import BertTopicSLab

if __name__ == '__main__':
    docs = open('one.txt', encoding='UTF-8').readlines()
    docs = [
        line
        for doc in docs
        for line in doc.split('。')
        if len(line) > 5
    ]
    model = BertTopicSLab('中央一号', docs)
    model.model_()

    print()