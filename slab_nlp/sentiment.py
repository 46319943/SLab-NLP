import paddlehub as hub
from typing import List


class PaddleSkepSentiment():
    def __init__(self):
        self.moulde = hub.Module(name="ernie_skep_sentiment_analysis")

    def sentiment_text(self, text: str):
        sentiment_result = self.moulde.predict_sentiment([text], use_gpu=False)[0]
        sentiment_score = sentiment_result['positive_probs'] - \
                          sentiment_result['negative_probs']
        return sentiment_score

    def sentiment_docs(self, docs: List[str]):
        sentiment_result_list = self.moulde.predict_sentiment(docs, use_gpu=False)
        sentiment_score_list = [
            sentiment_result['positive_probs'] - sentiment_result['negative_probs']
            for sentiment_result in sentiment_result_list
        ]
        return sentiment_score_list
