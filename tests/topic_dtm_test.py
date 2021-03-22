from slab_nlp.topic_dtm import DtmlModelSLab
import re


def year_process(docs):
    docs_postprocess = list()

    doc_counter = 0
    doc_time_slice = list()

    for doc in docs:
        match_result = re.match(r'(\d{4})年', doc)
        if match_result is not None and len(doc) <= 6:
            year_str = match_result.group(1)
            year_num = int(year_str)
            print(year_num)

            doc_time_slice.append(doc_counter)
            doc_counter = 0

        else:
            docs_postprocess.append(doc)
            doc_counter += 1

    return docs_postprocess, doc_time_slice


if __name__ == '__main__':
    import numpy as np

    docs = open(r'C:\Document\中共嘉兴\text.txt', encoding='UTF-8').readlines()
    docs, time_slice = year_process(docs)
    model = DtmlModelSLab('中共DTM', docs)
    time_slice = np.array(time_slice)
    time_slice = time_slice[0::2] + time_slice[1::2]
    model.model(docs, time_slice, 11)


