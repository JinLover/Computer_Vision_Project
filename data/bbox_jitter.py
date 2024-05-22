import copy
import json
import yaml
import cv2
import numpy as np
from tqdm import tqdm
import os
import re

# 맨 처음 파일 받았을 때
# train set 456개(unlabeled 포함), annotations 1748개
default_labeled_images = 230
default_annotations = 1748

def remove_temp_files():
    current_dir = os.getcwd()
    print(current_dir)
    train_dir = os.path.join(current_dir + '/train')
    for filename in os.listdir(train_dir):
        f = os.path.join(train_dir, filename)
        
        if re.search("tmp*", string=str(f)) is not None:
            #print(f)
            if os.path.exists(f):
                os.remove(f)

def make_bbox_annotation(id, image_id, category_id, bbox : list):
    ret = dict()
    ret["id"] = int(id)
    ret["image_id"] = image_id
    ret["category_id"] = int(category_id)
    assert len(bbox) == 4
    ret["bbox"] = bbox
    
    return ret

# augmentation is only for training
def make_img_info_annotation(id, width, height):
    ret = dict()
    ret["id"] = id
    ret["width"] = width
    ret['height'] = height
    ret['file_name'] = f'train/{id}.png'
    return ret

def get_number_of_labeled_images(train_json):
    len(train_json['images'])
    chk = dict()
    for i in range(len(train_json['annotations'])):
        ann_i = train_json['annotations'][i]
        ith_img_id = ann_i['image_id']
        chk[ith_img_id] = "chk"

    label_cnt = 0
    for i in range(len(train_json['images'])):
        if chk.get(i) is None:
            pass
        else:
            label_cnt += 1
            
    return label_cnt

def augmentation_loop(train_json, random_flip, max_jitter_x, max_jitter_y, exception_list = []):
    """
    iterate through all annotations,
    overwrite augmented images
    """
    current_dir = os.getcwd()
    #print(current_dir)
    train_dir = os.path.join(current_dir + '/train')
    img_dir_list = os.listdir(train_dir)
    image_len = len(img_dir_list)
    img_info = train_json['images']
    annotations = train_json['annotations']
    annotations_len = len(annotations)

    aug_chk = dict()
    tmp_img_id_list = []
    tmp_img_ann_list = []
    tmp_bbox_ann_list = []
    excluding_samples = dict()
    for i in range(len(exception_list)):
        excluding_samples[i] = i

    for i in tqdm(range(default_annotations)):
        image_id = annotations[i]['image_id']
        if excluding_samples.get(image_id) is not None:
            # skip the whole augmentation process
            continue
        
        img_h = img_info[image_id]['height']
        img_w = img_info[image_id]['width']
        aug_path = f'train/tmp_{image_id}.png'
        img_path = img_info[image_id]['file_name']
        
        # try to read augmented image,
        # if fails, then read original image
        # and write in aug_path
        if os.path.exists(aug_path):
            img = cv2.imread(aug_path)
        else:
            img = cv2.imread(img_path)
            cv2.imwrite(aug_path, img)
            img = cv2.imread(aug_path)
        #print(img.shape)
        
        [x,y,w,h] = annotations[i]['bbox']
        if w == 0 or h==0:
            continue
        if w is None or h is None:
            continue
        if w*h < 200:
            # exclude small bboxes
            continue
        category_id = annotations[i]['category_id']
        # crop image
        original_img = cv2.imread(img_path)
        cropped_bbox_region = original_img[y:y+h, x:x+w , :]
        # generate jitter positions
        jitter_x = np.round(np.random.uniform(-max_jitter_x, max_jitter_x, 1)).astype(int).item()
        jitter_y = np.round(np.random.uniform(-max_jitter_y, max_jitter_y, 1)).astype(int).item()
        
        # fill cropped area with Gaussian noise
        try:
            img[y:y+h, x:x+w , :] = np.random.normal(0,30, size = (h,w,3))
        except Exception as e:
            print(f'pass annotation {i}')
            continue
            
        # new_x = (x + jitter_x) // 1
        # new_y = (y + jitter_y) // 1
        
        # #print(new_x, new_y)
        
        # paste_target = img[max(new_y,0):min(new_y+h,img_h) , max(new_x,0):min(new_x+w, img_w), :]
        # new_h, new_w, _ = paste_target.shape
        
        # # pasted region should be inside the
        # # original image boundary
        # y1 = min(max(new_y,0),img_h)
        # y2 = min(new_y+new_h,img_h)
        # x1 = min(max(new_x,0),img_w)
        # x2 = min(new_x+new_w,img_w)
        # new_h = y2 - y1
        # new_w = x2 - x1
        
        y_start = max(0, y+jitter_y)
        y_end = min(y+jitter_y+h, img_h)
        new_y = y_start
        new_h = y_end - y_start
        
        x_start = max(0, x+jitter_x)
        x_end = min(x+jitter_x+w, img_w)
        new_x = x_start
        new_w = x_end - x_start
        
        final_crop = cropped_bbox_region[:new_h, :new_w, :]
        if random_flip is True:
            do_flip = True #if np.random.uniform(0.0, 1.0, 1).item() < 0.5 else False
            if do_flip is True:
                final_crop = cv2.flip(cropped_bbox_region[:new_h, :new_w, :], 1)
        
        #paste cropped region
        try:
            img[y_start:y_end, x_start:x_end, :] = final_crop
        except Exception as e:
            #print(new_h, new_w, y1, y2, x1, x2)
            print(f'{new_h}, {new_w}, {y_start}, {y_end}, {x_start}, {x_end}')
            continue

        cv2.imwrite(aug_path, img)
        
        tmp_img_id = f'tmp_{image_id}'
        tmp_img_info_ann = make_img_info_annotation(tmp_img_id, img_w, img_h)
        new_ann_id = annotations_len + i
        tmp_bbox_ann = make_bbox_annotation(new_ann_id, tmp_img_id, category_id, [new_x, new_y, new_w, new_h])

        if aug_chk.get(tmp_img_id) is None:
            aug_chk[tmp_img_id] = "chk"
            tmp_img_id_list.append(tmp_img_id)
        tmp_img_ann_list.append(tmp_img_info_ann)
        tmp_bbox_ann_list.append(tmp_bbox_ann)

    # re-map tmp_img_id into real image id
    # change all temporary image ids
    
    img_id_mapping = dict()
    print(len(tmp_img_id_list))
    for i in range(len(tmp_img_id_list)):
        img_id_mapping[tmp_img_id_list[i]] = image_len + i

    for img_ann in tmp_img_ann_list:
        tmp_id = img_ann['id']
        img_ann['id'] = img_id_mapping[tmp_id]

    for bbox_ann in tmp_bbox_ann_list:
        tmp_id = bbox_ann['image_id']
        bbox_ann['image_id'] = img_id_mapping[tmp_id]

    # change train image name from
    # tmp_i.png to j.png
    for tmp_id in tmp_img_id_list:
        aug_path = f'train/{tmp_id}.png'
        img = cv2.imread(aug_path)
        cv2.imwrite(f'train/{img_id_mapping[tmp_id]}.png', img)

    # add augmented annotations to original json data
    # and save them
    img_info.extend(tmp_img_ann_list)
    annotations.extend(tmp_bbox_ann_list)
    with open('train.json', 'w') as f:
        output_json = dict()
        output_json['images'] = img_info
        output_json['annotations'] = annotations
        json.dump(output_json, f)

def generate_bbox_augmented_dataset(json_file_name, max_jitter_x = 15.0, max_jitter_y = 15.0, random_flip=True,
                                    augment_iters = 1, exception_list = []):
    """
    Args:
        json_file_name (string): name of the train.json file to be augmented
        output_file_name (string, optional): name of the augmented json file. If none, original json_file_name is used.
        augment_iters (int, optional): number of augmentation iterations. Defaults to 1.
        exception_list (list, optional): images excluded from augmentation. Defaults to [].
    """
    
    for i in tqdm(range(augment_iters)):
        with open(json_file_name, 'r') as f:
            train_json = json.load(f)
        augmentation_loop(train_json, random_flip, max_jitter_x, max_jitter_y, exception_list)
        remove_temp_files()

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file_name', dest='json_file_name', required=True, type=str)
    parser.add_argument('--max_jitter_x', dest='max_jitter_x', required=True, type=float)
    parser.add_argument('--max_jitter_y', dest='max_jitter_y', required=True, type= float)
    parser.add_argument('--random_flip', dest='random_flip', action='store_true')
    parser.add_argument('--augment_iters', dest='augment_iters', required=True, type=int)
    parser.add_argument('--exception_list', dest='exception_list', nargs='+', required=True)
    
    args = parser.parse_args()
    
    generate_bbox_augmented_dataset(
        args.json_file_name, args.max_jitter_x, args.max_jitter_y,
        args.random_flip, args.augment_iters, args.exception_list
    )