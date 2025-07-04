from transformers import pipeline

class TextAnalyzer:
    def __init__(self):
        print("Loading NLP pipelines...")
        self.keyword_extractor = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.intent_model = pipeline("text-classification", model="microsoft/deberta-v3-small", return_all_scores=True)

    def analyze_text(self, text: str) -> dict:
        candidate_labels = ["electronics", "fashion", "furniture", "outdoor", "accessories", "home decor"]
        keyword_result = self.keyword_extractor(text, candidate_labels)

        intent_scores = self.intent_model(text)

        return {
            "keywords": keyword_result['labels'],
            "intent_scores": intent_scores
        }
