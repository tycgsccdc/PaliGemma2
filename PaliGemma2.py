from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
from transformers.image_utils import load_image
import torch

model_id = "google/paligemma2-28b-mix-448"
#model_id = "google/paligemma2-3b-mix-224"

url="https://s.teachifycdn.com/image/width=1920,quality=80/attachment/public_image/18058dee-59f8-4bb6-a29b-13b8ea1c6db1/7f4f6e37-535c-48b5-a8dd-5aa84986e27a.png" #托斯夫人
#url="https://s.teachifycdn.com/image/width=1920,quality=80/attachment/public_image/43215c84-b41c-4a7d-b4d4-09006c002a1a/16d53dd0-c7ae-4f7f-86a1-cb8a1783ce14.png" #＃太宰治
#url="https://s.teachifycdn.com/image/width=1920,quality=80/attachment/public_image/c94d7eec-6ef3-413a-b764-5b323ab424ad/3f05fbda-eb20-4450-9d6a-1c5048303933.png" #泰戈尔
#url="https://s.teachifycdn.com/image/width=1920,quality=80/attachment/public_image/18058dee-59f8-4bb6-a29b-13b8ea1c6db1/7f4f6e37-535c-48b5-a8dd-5aa84986e27a.png" #托斯夫人
#url="https://s.teachifycdn.com/image/width=1920,quality=80/attachment/public_image/68ab7600-b3a6-4186-b4cd-5bba122d4db5/28a9c102-7cec-429c-af53-841122405437.png" #菊池寬


image = load_image(url)

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="mps").eval()
processor = PaliGemmaProcessor.from_pretrained(model_id)

prompt = "<image>" + "ocr\n"
model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(model.device)
input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=2048, do_sample=False)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)
