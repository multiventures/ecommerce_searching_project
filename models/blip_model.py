from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

class ImageAnalyzer:
    def __init__(self):
        print("Loading BLIP-2 model...")
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)

        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16 if device_type == "cuda" else torch.float32
        ).to(self.device)

    def describe_image(self, image: Image.Image) -> str:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        description = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return description
