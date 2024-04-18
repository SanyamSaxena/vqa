from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import json
from tqdm import tqdm

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_path):
    images = []
#   for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values=pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    return preds

images_file = '/home/iitm/khyati_sanyam/vqa/data/lrdataset/LR_split_train_images.json'
with open(images_file) as json_data:
    imagesJSON = json.load(json_data)
images = [img['id'] for img in imagesJSON['images']]
path = '/home/iitm/khyati_sanyam/vqa/data/lrdataset/data/'
captions = {}
# paths = []
for img in tqdm(images):
    # paths.append(path+str(img)+'.tif')
    caption = predict_step(path+str(img)+'.tif')
    captions[str(img)] = caption
# captions = predict_step(paths)
with open('captions.json','w') as f:
    json.dump(captions,f)
