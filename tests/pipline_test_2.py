from slab_nlp.base import *
from slab_nlp.topic_dtm import DtmlModelSLab
from slab_nlp.topic_bert import BertTopicSLab
from slab_nlp.visualization import draw_sentiment

if __name__ == '__main__':

    dtm_slab = DtmlModelSLab.load('起点')

    if dtm_slab is None:

        docs_time_slice = []
        time_slice = []

        for time_slice_index in range(1, 12):
            docs = open(rf'start_{time_slice_index}.txt', encoding='UTF-8').readlines()
            docs = [doc.strip() for doc in docs if len(doc.strip()) > 5]

            docs_time_slice.extend(docs)
            time_slice.append(len(docs))

        dtm_slab = DtmlModelSLab('起点', docs_time_slice, time_slice)
        dtm_slab.model()

    docs = dtm_slab.docs
    time_slice = dtm_slab.time_slice

    model = BertTopicSLab('起点', docs)
    model.model()

    df = dtm_slab.df
    df['bert_topic'] = model.topic_index
    dtm_slab.save()

    model.hierarchical_model()
    model.model_time_series(time_slice=time_slice)

    draw_sentiment(df, save_dir='起点')

    print()
