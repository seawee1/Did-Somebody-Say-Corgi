import argparse
import os
from os.path import join as opjoin
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import pathlib
import pickle
import pandas as pd
import shutil
import json
from urllib.request import urlopen

def parse_prediction_file(prediction_path):
    with open(prediction_path, 'r') as f:
        content = f.read()
    lines = content.split('\n')
    predictions = []
    for line in lines:
        split = line.split(' ')
        if len(line) == 0: break
        pred_dict = {
            'cls': int(split[0]),
            'x': float(split[1]),
            'y': float(split[2]),
            'w': float(split[3]),
            'h': float(split[4])
        }
        predictions.append(pred_dict)
    return predictions

# Calculate overlap between bounding boxes
def get_box_overlap(bb1, bb2):
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0, 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    overlap1 = intersection_area/bb1_area
    overlap2 = intersection_area/bb2_area
    return overlap1, overlap2

def calculate_crop_box(w, h, prediction, target_res, min_res=None):
    # Image is too small
    min_side_image = min(w, h)
    if min_res is not None and min_side_image < min_res:
        return None

    # Convert relative to pixel value bounding box
    x_bbox_center = round(prediction['x'] * w)
    y_bbox_center = round(prediction['y'] * h)
    w_bbox = round(prediction['w'] * w)
    h_bbox = round(prediction['h'] * h)

    # Initial crop tries to not induce crop upscaling in order to reach target_res
    if min_res is None:
        crop_box_size = max(target_res, max(w_bbox, h_bbox))
    else:
        crop_box_size = max(min(target_res,min_res), max(w_bbox, h_bbox))

    if crop_box_size > min_side_image:
        crop_box_size = min_side_image

    is_partial_crop = True if crop_box_size < w_bbox or crop_box_size < h_bbox else False
    crop_box_cover = ((min(w_bbox, crop_box_size) * min(h_bbox, crop_box_size))) / crop_box_size**2

    '''
    # Maximum coverage possible
    crop_box_max_cover = ((min(w_bbox, min_res) * min(h_bbox, min_res))) / min_res ** 2
    if crop_box_cover < min_cover:
        # Coverage can't be reached
        if crop_box_max_cover < min_cover:
            return None

        # Iteratively reduce crop size and check coverage
        num_steps = 10
        step_size = (crop_box_size - min_res) / num_steps
        for step in range(num_steps):
            crop_box_size = round(crop_box_size - step_size)
            crop_box_cover = ((min(w_bbox, crop_box_size) * min(h_bbox, crop_box_size))) / crop_box_size**2

            # We don't want such training images, because the object is too thin to be cropped using this method
            if crop_box_size < w_bbox or crop_box_size < h_bbox:
                return None

            if crop_box_cover >= min_cover:
                break
    '''

    # Try to center crop around bbox center
    crop_x = max(0, x_bbox_center - round(crop_box_size / 2))
    crop_y = max(0, y_bbox_center - round(crop_box_size / 2))

    # Adjust in case the bbox is partially outside the image
    crop_x_over = w - (crop_x + crop_box_size)
    crop_y_over = h - (crop_y + crop_box_size)
    crop_x += crop_x_over if crop_x_over < 0 else 0
    crop_y += crop_y_over if crop_y_over < 0 else 0

    # Some assertions
    assert crop_x >= 0 and crop_x + crop_box_size <= w
    assert crop_y >= 0 and crop_y + crop_box_size <= h

    '''
    x_bbox = x_bbox_center - round(w_bbox / 2)
    y_bbox = y_bbox_center - round(h_bbox / 2)
    assert get_box_overlap(
        [crop_x, crop_y, crop_x + crop_box_size, crop_y + crop_box_size],
        [x_bbox, y_bbox, x_bbox + w_bbox, y_bbox + h_bbox]
    )[0] >= min_cover
    '''

    center_shift = (
        (crop_x + round(crop_box_size/2) - x_bbox_center)/w,
        (crop_y + round(crop_box_size/2) - y_bbox_center)/h
    )

    crop_box_dict = {
        'x': (crop_x + round(crop_box_size/2))/w,
        'y': (crop_y + round(crop_box_size/2))/h,
        'w': crop_box_size/w,
        'h': crop_box_size/h,
        'crop_size': crop_box_size,
        'crop_cover': crop_box_cover,
        'center_shift_x': center_shift[0],
        'center_shift_y': center_shift[1],
        'is_partial': is_partial_crop,
        'im_w': w,
        'im_h': h
    }

    return crop_box_dict

def create_proxy_file():
    json_url = "https://raw.githubusercontent.com/scidam/proxy-list/master/proxy.json"

    with urlopen(json_url) as url:
        json_proxies = json.loads(url.read().decode('utf-8'))

    print("The total number of proxies: ", len(json_proxies['proxies']))
    for proxy in json_proxies['proxies']:
        host = proxy['ip']
        port = proxy['port']
        with open('proxies.txt', 'a') as f:
            f.write(host + ':' + port + '\n')