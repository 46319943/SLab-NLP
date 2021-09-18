import pkuseg
import paddlehub as hub
from typing import List, Tuple
import abc


class SegmentResult():
    def __init__(self):
        self.result = None


class BaseSegment():
    # singleton
    segment_model = None

    def __init__(self):
        if BaseSegment.segment_model is None:
            BaseSegment.segment_model = self.get_segment_model()
        self.segment_model = BaseSegment.segment_model

    @abc.abstractmethod
    def get_segment_model(self):
        return

    @abc.abstractmethod
    def segment_text(self,
                     text: str,
                     stop_word_list: List[str] = [],
                     include_tag_list: List[str] = [],
                     min_length: int = 0) -> Tuple[List[str], List[str]]:
        return

    def segment_docs(self,
                     docs: List[str],
                     stop_word_list: List[str] = [],
                     include_tag_list: List[str] = [],
                     min_length: int = 0
                     ):
        doc_list = list()
        word_segment_list = list()
        tag_segment_list = list()

        for doc in docs:
            word_segment, tag_segment = self.segment_text(doc, stop_word_list=stop_word_list,
                                                          include_tag_list=include_tag_list, min_length=min_length)
            if len(word_segment) != 0:
                doc_list.append(doc)
                word_segment_list.append(word_segment)
                tag_segment_list.append(tag_segment)

        return doc_list, word_segment_list, tag_segment_list


class PKUSegment(BaseSegment):
    tag_list = ['n', 't', 's', 'f', 'm', 'q', 'b', 'r', 'v', 'a', 'z', 'd', 'p', 'c', 'u', 'y', 'e', 'o', 'i', 'l', 'j',
                'h', 'k', 'g', 'x', 'w', 'nr', 'ns', 'nt', 'nx', 'nz', 'vd', 'vn', 'vx', 'ad', 'an']

    def get_segment_model(self):
        return pkuseg.pkuseg(postag=True)

    def segment_text(self,
                     text: str,
                     stop_word_list: List[str] = [],
                     include_tag_list: List[str] = [],
                     min_length: int = 0
                     ):
        # word_segment, tag_segment = self.segment_model.cut(text)
        # word_tag_segment = list(zip(word_segment, tag_segment))
        word_tag_segment = self.segment_model.cut(text.strip())

        # TODO: 重构抽离

        if len(include_tag_list) == 0:
            include_tag_list = PKUSegment.tag_list
        word_tag_segment_filter = list(zip(
            *[[word, tag] for word, tag in word_tag_segment
              if word not in stop_word_list and tag in include_tag_list and len(word) >= min_length]
        ))
        if len(word_tag_segment_filter) == 0:
            return [], []
        word_segment, tag_segment = word_tag_segment_filter
        return word_segment, tag_segment


class PaddleLACSegment(BaseSegment):
    tag_list = []

    def get_segment_model(self):
        return hub.Module(name="lac")

    def segment_text(self,
                     text: str,
                     stop_word_list: List[str] = [],
                     include_tag_list: List[str] = [],
                     min_length: int = 0
                     ):
        word_segment, tag_segment = self.segment_model.cut(text=[text.strip()], return_tag=True)[0]
        word_tag_segment = list(zip(word_segment, tag_segment))
        if len(include_tag_list) == 0:
            include_tag_list = PaddleLACSegment.tag_list
        word_segment, tag_segment = list(zip(
            *[[word, tag] for word, tag in word_tag_segment
              if word not in stop_word_list and tag in include_tag_list and len(word) >= min_length]
        ))
        return word_segment, tag_segment
