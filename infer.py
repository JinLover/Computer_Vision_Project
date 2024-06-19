#! /usr/bin/env python3

from __future__ import division

import argparse
from tqdm import tqdm
import numpy as np
import os
from PIL import Image

import torch
import zipfile

from pytorchyolo.models import load_model
from pytorchyolo.utils.utils import non_max_suppression, rescale_boxes
from pytorchyolo.detect import _create_data_loader
import json
from pathlib import Path

import argparse

"""
File to generate prediction result for submission to CodaLab
model_num : number of the checkpoint file to test. default as 500.
(default train length is 500 epochs)
save : if set, generate pred.json and predctions.zip file.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--model_num', dest='model_num', default=500, type=int)
parser.add_argument('--save', dest='save', action='store_true')

args = parser.parse_args()

MODEL_NUM = args.model_num

model = load_model("config/yolov3-custom.cfg", f'checkpoints/yolov3_ckpt_{MODEL_NUM}.pth').cuda()
model.eval()
images = sorted([str(x) for x in Path("Dataset/test").rglob("*.png")])
dataloader = _create_data_loader(images, 1, model.hyperparams["height"], 8)

pred = []
# conf_thres = 0.5
conf_thres = 0.01

nms_thres = 0.4

elapsed_time = []

for (img_paths, input_imgs) in tqdm(dataloader, desc="Detecting"):
    name = img_paths[0].split("/")[-1].split(".")[0]
    # Get detections
    with torch.no_grad():
        # check the inference time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        detections = model(input_imgs.cuda())
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        
        h, w = (345, 640)
        detections = rescale_boxes(detections[0], model.hyperparams["height"], (h, w))
        for x1, y1, x2, y2, conf, cls_pred in detections:
            x1, y1, x2, y2, conf, cls_pred = map(lambda x:x.item(), [x1, y1, x2, y2, conf, cls_pred])
            x1, y1, x2, y2 = map(lambda x: 0 if x < 0 else x, [x1, y1, x2, y2])
            y1, y2 = map(lambda y: h if y > h else y, [y1, y2])
            x1, x2 = map(lambda x: w if x > w else x, [x1, x2])
            pred.append({
                        "image_id": int(name),
                        "category_id": int(cls_pred)+1,
                        "bbox": list(map(lambda x:float(f"{x:.3f}"), [x1, y1, x2-x1, y2-y1])),
                        "score": float(f"{conf:.5f}"),
                    })
        end.record()
        torch.cuda.synchronize()
        elapsed_time.append(start.elapsed_time(end))
print(f"Average inference time: {np.mean(elapsed_time):.3f} ms")


if args.save is True: 
    # Save the list of predictions as a JSON file
    with open("pred.json", "w") as f:
        json.dump(pred, f)

    # Create a ZIP file and add the JSON file to it
    zip_file = zipfile.ZipFile("predictions.zip", "w")
    zip_file.write("pred.json")
    zip_file.close()