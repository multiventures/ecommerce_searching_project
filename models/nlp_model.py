from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class TextAnalyzer:
    def __init__(self):
        print("Loading Phi-2 model for NLP tasks...")
        model_id = "microsoft/phi-2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

    def analyze_text(self, description: str) -> dict:
        """
        Extracts keywords from a product description using phi-2.
        """
        prompt = f"Extract keywords from the following text:\n{description}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id
        )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"keywords": result}

    def generate_intent(self, text: str) -> dict:
        """
        Generates the intent of the description in a complete sentence.
        """
        prompt = f"What is the intent of this description? Answer in one sentence:\n{text}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id
        )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"intent": result}
