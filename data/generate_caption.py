import pandas as pd
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import os
from tqdm import tqdm
import open_clip


input_path = ""  # path for HR

name_list = []
with open(f"data/ImageNet/Obj512_all/all.txt", 'r') as f:
    for line in f.readlines():
        name_list.append(line.rstrip('\n'))
image_path_num = len(name_list)

result_name = f"data/ImageNet/Obj512_all/blip2_imagenet_captions_all.json"

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
torch_dtype=torch.float16
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', device=torch.device("cuda"))
tokenizer = open_clip.get_tokenizer('ViT-H-14')

df = pd.DataFrame(columns=['filename', 'caption1', 'caption2', 'caption3', 'clip_score1', 'clip_score2', 'clip_score3'], dtype=float)

for name in tqdm(name_list):
    # load image
    image = Image.open(os.path.join(input_path, name)).convert('RGB')

    # generate caption
    inputs = processor(image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    generated_text = generated_text.lower().replace('.', ',').rstrip(',')
    caption1 = generated_text

    prompt = "a photo of"
    inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    generated_text = generated_text.lower().replace('.', ',').rstrip(',')
    caption2 = prompt + ' ' + generated_text

    prompt = "Question: Please describe the contents in the photo in details. Answer:"
    inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    generated_text = generated_text.lower().replace('.', ',').rstrip(',')
    caption3 = generated_text

    image = preprocess(image).unsqueeze(0)
    text = tokenizer([caption1, caption2, caption3])
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = clip_model.encode_image(image.half().cuda())
        text_features = clip_model.encode_text(text.cuda())
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = image_features @ text_features.T

    print([name, caption1, caption2, caption3, float(text_probs[0, 0]), float(text_probs[0, 1]), float(text_probs[0, 2])])
    df.loc[len(df.index)] = [name, caption1, caption2, caption3, float(text_probs[0, 0]), float(text_probs[0, 1]), float(text_probs[0, 2])]

df.to_json(result_name)
