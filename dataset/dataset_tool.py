import argparse
import os
from os.path import join as opjoin
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import pathlib
import pickle
#from imagededup.methods import PHash
import pandas as pd
import shutil

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

def clean():
    image_names = read_image_dir()

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


def filter():
    image_names = read_image_dir()

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

def crop():
    image_names = read_image_dir()

    # Create crop directory
    pathlib.Path(crop_dir_256).mkdir(parents=True, exist_ok=True)
    pathlib.Path(crop_dir_512).mkdir(parents=True, exist_ok=True)
    pathlib.Path(crop_dir_1024).mkdir(parents=True, exist_ok=True)

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
                bbox_center_x = x_bbox + w_bbox / 2
                bbox_center_y = y_bbox + h_bbox / 2
                crop_x = max(0, bbox_center_x - crop_size / 2)
                crop_y = max(0, bbox_center_y - crop_size / 2)

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

                if crop_size >= 256 and crop_size < 512:
                    crop_dir = crop_dir_256
                elif crop_size >= 512 and crop_size < 1024:
                    crop_dir = crop_dir_512
                else:
                    crop_dir = crop_dir_1024

                im_crop.save(os.path.join(crop_dir, image_name), "JPEG")

        else:
            skip = True

    print(f'Skipped {skipped} images, because crop was too small or other reasons.')

def remove_duplicates():
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
                    if os.isfile(os.path.join(pred_dir, base+'.pkl')):
                        os.remove(os.path.join(prd_dir, base+'.pkl'))

                    duplicate_keys.remove(dup)
                except:
                    pass
                removed += 1
    print(f'Removed {removed} duplicates')

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

def rename():
    image_names = read_image_dir()
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

def resize():
    image_names = read_image_dir()
    for image_name in tqdm(image_names):
        image = Image.open(os.path.join(image_dir, image_name))

        w, h = image.size
        if w == args.resize:
            continue

        image = image.resize((args.resize, args.resize))
        image.save(os.path.join(image_dir, image_name))


def read_image_dir():
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StyleGAN2 dataset tool.')
    parser.add_argument('image_dir')
    parser.add_argument('--clean', action='store_true', help='Removes damaged image files and convertes healty ones to RGB JPEG.')
    parser.add_argument('--remove-duplicates', action='store_true', help='Removes duplicate images using Perceptual Hashing method.')
    parser.add_argument('--filter', action='store_true', help='Moves all images with more than one bounding box to image_dir/supervision.')
    parser.add_argument('--crop', action='store_true', help='Crops images based on predicted bounding box to quadratic image. ')
    parser.add_argument('--rename', action='store_true', help='Renames images and prediction files to [1,2,3,4,...,num_images].')
    parser.add_argument('--resize', type=int, default=-1, help='Resizes images in folder to [arg] pixel size.')

    parser.add_argument('--create-dummy-index', action='store_true', help='Create dummy .csv for annot_tool.py.')
    parser.add_argument('--index-to-labels', action='store_true', help='Creates prediction .pkl from index.csv.')

    args = parser.parse_args()

    image_dir = args.image_dir
    pred_dir = os.path.join(image_dir, 'labels')
    supervision_dir = os.path.join(image_dir, 'supervision')
    crop_dir_256 = os.path.join(image_dir, 'crop256')
    crop_dir_512 = os.path.join(image_dir, 'crop512')
    crop_dir_1024 = os.path.join(image_dir, 'crop1024')

    if args.clean:
        clean()
    elif args.remove_duplicates:
        remove_duplicates()
    elif args.filter:
        filter()
    elif args.crop:
        crop()
    elif args.rename:
        rename()
    elif args.resize > 0:
        resize()
    elif args.create_dummy_index:
        create_dummy_index()
    elif args.index_to_labels:
        index_to_labels()

