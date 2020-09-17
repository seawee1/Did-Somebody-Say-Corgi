import argparse
import os
from os.path import join as opjoin
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import pathlib
import pickle
from imagededup.methods import PHash
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

'''
def clean(image_dir, pred_dir):
    image_names = read_image_dir(image_dir)

    removed = 0
    for image_name in image_names:
        basename, ext = get_base_ext(image_name)

        if len(ext) < 2 or ext not in ['jpg', 'jpeg', 'png']:
            os.remove(opjoin(image_dir, image_name))
            removed += 1
            continue

        # Try to open
        try:
            im = Image.open(opjoin(image_dir, image_name))

            # Convert to RGB, save as JPEG
            im = im.convert('RGB')
            im.save(opjoin(image_dir, image_name), "JPEG")
        except:
            # Remove image
            os.remove(opjoin(image_dir, image_name))

            # Remove prediction file
            base, ext = get_base_ext(image_name)
            if os.isfile(os.path.join(pred_dir, base + '.txt')):
                os.remove(os.path.join(pred_dir, base + '.txt'))
            removed += 1

    print(f'Removed {removed} images during cleaning process')
'''
'''
def filter(image_dir, pred_dir, supervision_dir):
    image_names = read_image_dir(image_dir)

    # Create supervision directory
    pathlib.Path(supervision_dir).mkdir(parents=True, exist_ok=True)

    supervisions = 0
    for image_name in tqdm(image_names):
        basename, ext = get_base_ext(image_name)

        # Load prediction
        predictions = parse_prediction_file(os.path.join(pred_dir, basename + '.txt'))

        # If unequal to 1, supervision is needed
        if len(predictions) != 1:
            os.rename(opjoin(image_dir, image_name), opjoin(supervision_dir, image_name))
            supervisions += 1

    print(f'For {supervisions} images supervision is needed... Check supervision subfolder!')
'''

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

#def calculate_crop_box(w, h, prediction, target_res, min_res, min_cover):
def calculate_crop_box(w, h, prediction, target_res, min_res, min_cover):
    # Image is too small
    min_side_image = min(w, h)
    if min_side_image < min_res:
        return None, None, None, None, None, None

    # Convert relative to pixel value bounding box
    x_bbox_center = round(prediction['x'] * w)
    y_bbox_center = round(prediction['y'] * h)
    w_bbox = round(prediction['w'] * w)
    h_bbox = round(prediction['h'] * h)

    # Initial crop tries to not induce crop upscaling in order to reach target_res
    crop_box_size = max(target_res, max(w_bbox, h_bbox))

    if crop_box_size > min_side_image:
        crop_box_size = min_side_image

    # Check if is partial crop
    # TODO... What to do then?
    is_partial_crop = True if crop_box_size < w_bbox or crop_box_size < h_bbox else False

    crop_box_cover = ((min(w_bbox, crop_box_size) * min(h_bbox, crop_box_size))) / crop_box_size**2

    # Maximum coverage possible
    crop_box_max_cover = ((min(w_bbox, min_res) * min(h_bbox, min_res))) / min_res ** 2
    if crop_box_cover < min_cover:
        # Coverage can't be reached
        if crop_box_max_cover < min_cover:
            return None, None, None, None, None, None

        # Iteratively reduce crop size and check coverage
        num_steps = 10
        step_size = (crop_box_size - min_res) / num_steps
        for step in range(num_steps):
            crop_box_size = round(crop_box_size - step_size)
            crop_box_cover = ((min(w_bbox, crop_box_size) * min(h_bbox, crop_box_size))) / crop_box_size**2

            # We don't want such training images, because the object is too thin to be cropped using this method
            if crop_box_size < w_bbox or crop_box_size < h_bbox:
                return None, None, None, None, None, None

            if crop_box_cover >= min_cover:
                break

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

    x_bbox = x_bbox_center - round(w_bbox / 2)
    y_bbox = y_bbox_center - round(h_bbox / 2)
    assert get_box_overlap(
        [crop_x, crop_y, crop_x + crop_box_size, crop_y + crop_box_size],
        [x_bbox, y_bbox, x_bbox + w_bbox, y_bbox + h_bbox]
    )[0] >= min_cover

    crop_box_dict = {
        'x': (crop_x + round(crop_box_size/2))/w,
        'y': (crop_y + round(crop_box_size/2))/h,
        'w': crop_box_size/w,
        'h': crop_box_size/h
    }

    center_shift = (
        (crop_x + round(crop_box_size/2) - x_bbox_center)/w,
        (crop_y + round(crop_box_size/2) - y_bbox_center)/h
    )

    return crop_box_dict, center_shift, crop_box_size, crop_box_cover, crop_box_max_cover, is_partial_crop

'''
import random
from PIL import Image, ImageDraw
import pandas as pd
from os.path import join as opjoin

DATA_DIR = 'D:\\dataset'
IMAGE_DIR = 'D:\\dataset\images'
LABEL_DIR = 'D:\\dataset\labels'
from pathlib import Path
data = [Path(x).stem for x in os.listdir(LABEL_DIR) if Path(x).suffix == '.txt']

num_samples = 0
ims = []
sample_idx = []
prediction_lst = []
while num_samples < 20:
    idx = random.randint(0, len(data))

    predictions = parse_prediction_file(opjoin(LABEL_DIR, data[idx] + '.txt'))
    if len(predictions) != 1: continue

    im = Image.open(opjoin(IMAGE_DIR, data[idx] + '.jpeg'))

    ims.append(im)
    sample_idx.append(idx)
    prediction_lst.append(predictions[0])
    num_samples += 1

labels = []
for i, idx in enumerate(sample_idx):
    w, h = ims[i].size
    crop_box, center_shift, crop_box_size, crop_box_cover, is_partial_crop = calculate_crop_box(w, h,
                                                                                                             prediction_lst[
                                                                                                                 i],
                                                                                                             1024, 0,
                                                                                                             0.0)
    labels.append(f'{crop_box_size}, {center_shift}, {crop_box_cover}, {is_partial_crop}')
    sample_idx.append(idx)

'''

def crop(image_dir, pred_dir, crop_dir, supervision_dir, target_res, min_cover, min_res, max_shift):
    image_names = read_image_dir(image_dir)
    pathlib.Path(supervision_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(crop_dir).mkdir(parents=True, exist_ok=True)

    skipped = 0
    for image_name in tqdm(image_names):
        basename, ext = get_base_ext(image_name)

        # Open image
        im = Image.open(opjoin(image_dir, image_name))
        skip = False

        if os.path.isfile(opjoin(pred_dir, basename + '.txt')):
            # Load prediction
            predictions = parse_prediction_file(os.path.join(pred_dir, basename + '.txt'))

            if len(predictions) != 1:
                skip = True
                skipped += 1
            else:
                prediction = predictions[0]

                # Image dimensions
                w, h = im.size
                min_side_img = min(w, h)

                # BBox
                x_bbox = round(prediction['x'] * w)
                y_bbox = round(prediction['y'] * h)
                w_bbox = round(prediction['w'] * w)
                h_bbox = round(prediction['h'] * h)

                # Calculate longer side of bbox
                crop_size = max(w_bbox, h_bbox)

                if crop_size > min(w, h):
                    crop_size = min(w,h)

                # Skip, because we only do 2x and 4x super resolution
                if crop_size < 256:
                    skipped += 1
                    continue

                # Try to center crop around bbox center
                crop_x = max(0, x_bbox - crop_size / 2)
                crop_y = max(0, y_bbox - crop_size / 2)

                # Adjust in case the bbox is partially outside the image
                crop_x_over = w - (crop_x + crop_size)
                crop_y_over = h - (crop_y + crop_size)
                if crop_x_over < 0:
                    crop_x += crop_x_over
                if crop_y_over < 0:
                    crop_y += crop_y_over

                # Crop and save based on crop size
                im_crop = im.crop((crop_x, crop_y, crop_x + crop_size, crop_y + crop_size))
                im_crop = im_crop.convert('RGB')

                im_crop.save(os.path.join(crop_dir, image_name), "JPEG")

        else:
            skip = True

    print(f'Skipped {skipped} images, because crop was too small or other reasons.')

def remove_duplicates(image_dir):
    phasher = PHash()

    # Generate encodings for all images in an image directory
    encodings = phasher.encode_images(image_dir=image_dir)

    # Find duplicates using the generated encodings
    duplicates = phasher.find_duplicates(encoding_map=encodings)

    # Remove duplicates
    duplicate_keys = [key for key in duplicates.keys() if len(duplicates[key]) != 0]
    removed = 0
    for key in duplicates.keys():
        if key in duplicate_keys:
            for dup in duplicates[key]:
                try:
                    base, ext = get_base_ext(dup)
                    os.remove(os.path.join(image_dir, dup))
                    duplicate_keys.remove(dup)
                except:
                    pass
                removed += 1

    print(f'Removed {removed} duplicates')

'''
def create_dummy_index():
    image_names = read_image_dir()
    ids = [get_base_ext(x)[0] for x in image_names]
    flagged = [False] * len(ids)

    df = pd.DataFrame(list(zip(ids, flagged)),columns=['id', 'flagged'])

    df.to_csv(open(os.path.join(image_dir, 'index.csv'), 'w', encoding='utf-8'), index=False, line_terminator='\n')

def index_to_labels():
    df = pd.read_csv(os.path.join(image_dir, 'index.csv'))
    ldir = os.path.join(image_dir, 'labels')
    idir = os.path.join(image_dir, 'img')
    pathlib.Path(ldir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(idir).mkdir(parents=True, exist_ok=True)

    for item in df.iterrows():
        item = item[1]
        filename = item['id']
        if item['flagged'] == True:
            continue

        if os.path.isfile(os.path.join(image_dir, 'images', filename + '.jpg')):
            filename += '.jpg'
        elif os.path.isfile(os.path.join(image_dir, 'images', filename + '.jpeg')):
            filename += '.jpeg'
        elif os.path.isfile(os.path.join(image_dir, 'images', filename + '.png')):
            filename += '.png'
        else:
            print(filename)

        bbox = np.array([[item['bboxX'], item['bboxY'], item['bboxX'] + item['bboxSize'], item['bboxY'] + item['bboxSize']]])
        labels = np.array([18])
        scores = np.array([1.0])
        label_dict = {'boxes': bbox, 'labels': labels, 'scores': scores}

        basename, ext = get_base_ext(filename)

        pickle.dump(label_dict, open(os.path.join(ldir, basename + '.pkl'), 'wb'))
        shutil.copyfile(os.path.join(image_dir, 'images', filename), os.path.join(idir, filename))
'''

'''
def rename():
    image_names = read_image_dir(image_dir)
    image_dir_tmp = os.path.join(image_dir, 'tmp')
    pred_dir_tmp = os.path.join(image_dir, 'pred_dir_tmp')

    pathlib.Path(image_dir_tmp).mkdir(parents=True, exist_ok=True)
    #pathlib.Path(pred_dir_tmp).mkdir(parents=True, exist_ok=True)

    if os.path.isdir(pred_dir):
        os.rename(pred_dir, pred_dir_tmp)

    pathlib.Path(pred_dir).mkdir(parents=True, exist_ok=True)
    for image_name in tqdm(image_names):
        os.rename(os.path.join(image_dir, image_name), os.path.join(image_dir_tmp, image_name))

    counter = 0
    for image_name in tqdm(image_names):
        base, ext = get_base_ext(image_name)
        os.rename(os.path.join(image_dir_tmp, image_name), os.path.join(image_dir, f'{counter}.{ext}'))
        os.rename(os.path.join(pred_dir_tmp, base + '.pkl'), os.path.join(pred_dir, f'{counter}.pkl'))
        counter += 1

    os.remove(image_dir_tmp)
    os.remove(pred_dir_tmp)
'''

'''
def resize(image_dir):
    image_names = read_image_dir(image_dir)
    for image_name in tqdm(image_names):
        image = Image.open(os.path.join(image_dir, image_name))

        w, h = image.size
        if w == args.resize:
            continue

        image = image.resize((args.resize, args.resize))
        image.save(os.path.join(image_dir, image_name))
'''

def clean_index():
    pass

def expand_index():
    pass


def read_image_dir(image_dir):
    image_names = [x for x in os.listdir(image_dir) if not os.path.isdir(opjoin(image_dir, x))]
    return image_names

def get_base_ext(file_name):
    # In case filename has dots in it
    basename = ''
    split = file_name.split('.')
    for i in range(len(split) - 1):
        basename += split[i] + '.'
    basename = basename[:-1]
    ext = split[-1]
    return basename, ext

def clean_index(index_path):
    pass

def expand_index(index_path):
    pass

def dataset(args):
    image_dir = args.image_dir
    pred_dir = os.path.join(image_dir, 'labels')
    supervision_dir = os.path.join(image_dir, 'supervision')
    crop_dir = os.path.join(image_dir, 'crop')

    if args.clean:
        clean(image_dir, pred_dir)
    elif args.remove_duplicates:
        remove_duplicates(image_dir)
    #elif args.filter:
        #filter(image_dir, pred_dir, supervision_dir)
    elif args.crop is not None:
        crop(image_dir, pred_dir, crop_dir, supervision_dir, *args.crop)
    #elif args.rename:
    #    rename()
    #elif args.resize > 0:
    #    resize(image_dir)
    #elif args.create_dummy_index:
    #    create_dummy_index()
    #elif args.index_to_labels:
    #    index_to_labels()


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


def instatouch(args):
    if args.clean_index is not None:
        clean_index(args.clean_index)
    if args.expand_index is not None:
        expand_index(args.expand_index)
    if args.create_proxy_file:
        create_proxy_file()


'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StyleGAN2 dataset tool.')
    subparser = parser.add_subparsers(help='Choose between [dataset, instatouch]')

    dataset_parser = subparser.add_parser('dataset')
    dataset_parser.set_defaults(func=dataset)
    dataset_parser.add_argument('image_dir')
    dataset_parser.add_argument('--clean', action='store_true', help='Removes damaged image files and convertes healty ones to RGB JPEG.')
    dataset_parser.add_argument('--remove-duplicates', action='store_true', help='Removes duplicate images using Perceptual Hashing method.')
    #dataset_parser.add_argument('--filter', nargs=3, metavar=('min_cover', 'min_res', 'max_shift'), help='Full example: num_pred != 1 and cover >= 0.5 and max_side >= 700')
    dataset_parser.add_argument('--crop', nargs=4, metavar=('target_res', 'min_cover', 'min_res', 'max_shift'), help='')
    #dataset_parser.add_argument('--crop', action='store_true', help='Crops images based on predicted bounding box to quadratic image. ')
    #dataset_parser.add_argument('--rename', action='store_true', help='Renames images and prediction files to [1,2,3,4,...,num_images].')
    #dataset_parser.add_argument('--resize', type=int, default=-1, help='Resizes images in folder to [arg] pixel size.')
    #parser.add_argument('--create-dummy-index', action='store_true', help='Create dummy .csv for annot_tool.py.')
    #parser.add_argument('--index-to-labels', action='store_true', help='Creates prediction .pkl from index.csv.')

    insta_parser = subparser.add_parser('instatouch')
    insta_parser.set_defaults(func=dataset)
    insta_parser.add_argument('--clean-index', type=str, default=None, help='')
    insta_parser.add_argument('--expand-index', type=str, default=None, help='')
    insta_parser.add_argument('--create-proxy-file', action='store_true', help='')
    #insta_parser.add_argument('--combine-indices')

    args = parser.parse_args()
    print(args.crop)

    args.func(args)
'''

