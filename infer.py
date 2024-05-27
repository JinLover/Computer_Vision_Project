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

model = load_model("config/yolov3-custom.cfg", "checkpoints/yolov3_ckpt_300.pth").cuda()
model.eval()
images = sorted([str(x) for x in Path("Dataset/test").rglob("*.png")])
dataloader = _create_data_loader(images, 1, model.hyperparams["height"], 8)

pred = []
# conf_thres = 0.5
conf_thres = 0.01
nms_thres = 0.4

for (img_paths, input_imgs) in tqdm(dataloader, desc="Detecting"):
    name = img_paths[0].split("/")[-1].split(".")[0]
    # Get detections
    with torch.no_grad():
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

# Save the list of predictions as a JSON file
with open("pred.json", "w") as f:
    json.dump(pred, f)

# Create a ZIP file and add the JSON file to it
zip_file = zipfile.ZipFile("predictions.zip", "w")
zip_file.write("pred.json")
zip_file.close()