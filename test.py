import sys
from transformers import BertConfig, RobertaConfig, RobertaModel, BertModel
from maskrcnn_benchmark.engine.predictor_glip import *
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import requests
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm
import json
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

def load_local_image(file_path):
    """
    Given a local file path of an image, loads the image and
    returns a NumPy array representing the image in BGR format
    """
    # Open the local image using PIL
    pil_image = Image.open(file_path).convert("RGB")

    # Convert the PIL image to a NumPy array in BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]

    return image

config_file = "Swin_T_O365_GoldG.yaml"
weight_file = "tiny_model_o365_goldg_cc_sbu.pth"

cfg.local_rank = 0
cfg.num_gpus = 4
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

glip_demo = GLIPDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
    show_mask_heatmaps=False
)

caption_pascal_person = "person . bird . cat . cow . dog . horse . sheep . aeroplane . bicycle . boat . bus . car . motorbike . train . bottle . chair . dining table . potted plant . sofa . tv/monitor ."

json_file_path_input = "../../../DATASET/PASCAL.json"
with open(json_file_path_input, "r") as json_file_input:
    input_data = json.load(json_file_input)

output_results = []

for entry in tqdm(input_data, desc="Running Object Detection", unit="image"):
    img_path = '../../../DATASET/' + entry["img"]
    local_image = load_local_image(img_path)
    result, top_predictions = glip_demo.run_on_web_image(original_image=local_image,mask_image = local_image, original_caption = caption_pascal_person, attention_caption = caption_pascal, thresh = 0.51,fixed_layer=index,fusion_image=local_image,fusion_caption=caption_pascal_person)
    labels = top_predictions.get_field("labels").tolist()
    result_list = [0] * 20
    for i in labels:
        result_list[i - 1] = 1
    output_results.append({"annotation": result_list, "img": entry["img"]})

json_file_path_output = f"output.json"
with open(json_file_path_output, "w") as json_file_output:
    json.dump(output_results, json_file_output)
