# from transformers import pipeline

# class TextAnalyzer:
#     def __init__(self):
#         print("Loading NLP pipelines...")
#         self.keyword_extractor = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
#         self.intent_model = pipeline("text-classification", model="microsoft/deberta-v3-small", return_all_scores=True)

#     def analyze_text(self, text: str) -> dict:
#         candidate_labels = ["electronics", "fashion", "furniture", "outdoor", "accessories", "home decor"]
#         keyword_result = self.keyword_extractor(text, candidate_labels)

#         intent_scores = self.intent_model(text)

#         return {
#             "keywords": keyword_result['labels'],
#             "intent_scores": intent_scores
#         }


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class TextAnalyzer:
    def __init__(self):
        print("Loading Phi-2 for prompt-based keyword/intent extraction...")
        model_id = "microsoft/phi-2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

    def analyze_text(self, description: str) -> dict:
        prompt = f"Extract keywords and intent from: {description}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Optional: try to split keywords/intent if response is structured
        return {
            "raw_output": response
        }