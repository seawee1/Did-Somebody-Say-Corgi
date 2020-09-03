import argparse
import torchvision
import os
from os.path import join as opjoin
import numpy as np
from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor
from tqdm import tqdm
import pathlib
import pickle
from imagededup.methods import PHash
import pandas as pd
import shutil


def predict_labels():
    image_names = read_image_dir()

    # Load model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Create label directory
    pathlib.Path(pred_dir).mkdir(parents=True, exist_ok=True)

    for image_name in tqdm(image_names):
        basename, ext = get_base_ext(image_name)

        if ext not in ['jpg', 'jpeg', 'png']:
            continue

        # Predict
        image = Image.open(os.path.join(image_dir, image_name))
        image_tensor = ToTensor()(image).unsqueeze(0)
        predictions = model(image_tensor)[0]

        # Cast to numpy
        for key, value in predictions.items():
            predictions[key] = value.detach().numpy()

        # Dump dict into label folder
        pickle.dump(predictions, open(opjoin(pred_dir, image_name.split('.')[0] + '.pkl'), "wb"))

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

            # Check resolution
            w, h = im.size
            if min(w, h) < args.clean:
                removed += 1
                im.close()
                os.remove(opjoin(image_dir, image_name))
                continue

            # Convert to RGB, save as JPEG
            im = im.convert('RGB')
            im.save(opjoin(image_dir, image_name), "JPEG")
        except:
            os.remove(opjoin(image_dir, image_name))

            base, ext = get_base_ext(image_name)
            if os.isfile(os.path.join(pred_dir, base + '.pkl')):
                os.remove(os.path.join(prd_dir, base + '.pkl'))
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
        prediction = pickle.load(open(os.path.join(pred_dir, basename + '.pkl'), 'rb'))

        # Find predictions > probability threshold
        idxs = np.argwhere((prediction['labels'] == 18) & (prediction['scores'] > args.filter)).flatten()

        # If unequal to 1, supervision is needed
        if len(idxs) != 1:
            os.rename(opjoin(image_dir, image_name), opjoin(supervision_dir, image_name))
            supervisions += 1

    print(f'For {supervisions} images supervision is needed... Check supervision subfolder')

def crop():
    image_names = read_image_dir()

    # Create crop directory
    pathlib.Path(crop_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(supervision_dir).mkdir(parents=True, exist_ok=True)

    center_crops = 0
    for image_name in tqdm(image_names):
        basename, ext = get_base_ext(image_name)

        # Open image
        im = Image.open(opjoin(image_dir, image_name))
        center_crop = False

        if os.path.isfile(opjoin(pred_dir, basename + '.pkl')):
            # Load prediction
            prediction = pickle.load(open(opjoin(pred_dir, basename + '.pkl'), 'rb'))

            # Find predictions > probability threshold
            p_thresh = 0.9
            if args.filter != -1.0:
                p_thresh = args.filter
            idxs = np.argwhere((prediction['labels'] == 18) & (prediction['scores'] > p_thresh)).flatten()

            # If unequal to 1, supervision is needed. Here we just do a center crop
            if len(idxs) != 1:
                center_crop = True
                #os.rename(opjoin(image_dir, image_name), opjoin(supervision_dir, image_name))
            else:
                # Image dimensions
                w, h = im.size
                min_side_img = min(w, h)

                # Get bbox
                idx = idxs[0]
                bbox = prediction['boxes'][idx]

                # Scale to at least 1024 pixels
                if min_side_img < 1024:
                    scale_f = 1024 / min_side_img
                    im = im.resize((int(w * scale_f), int(h * scale_f)))
                    w, h = im.size
                    min_side_img = min(w, h)
                    bbox *= scale_f

                # Calculate longer side of bbox
                max_side_bbox = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]))

                # Ensure that crop is at least 1024, fits into image and,
                # if possible, has size max_side_bbox + max_size_img/10
                crop_size = min(min_side_img, max(1024, max_side_bbox + min_side_img / 10))

                # Try to center crop around bbox center
                bbox_center_x = bbox[0] + (bbox[2] - bbox[0]) / 2
                bbox_center_y = bbox[1] + (bbox[3] - bbox[1]) / 2
                crop_x = max(0, bbox_center_x - crop_size / 2)
                crop_y = max(0, bbox_center_y - crop_size / 2)

                # Adjust in case the bbox is partially outside the image
                crop_x_over = w - (crop_x + crop_size)
                crop_y_over = h - (crop_y + crop_size)
                if crop_x_over < 0:
                    crop_x += crop_x_over
                if crop_y_over < 0:
                    crop_y += crop_y_over

        else:
            center_crop = True

        if center_crop:
            # Image dimensions
            w, h = im.size
            min_side_img = min(w, h)

            # Scale to at least 1024 pixels
            if min_side_img < 1024:
                scale_f = 1024/min_side_img
                im = im.resize((int(w * scale_f), int(h * scale_f)))
                w, h = im.size
                min_side_img = min(w, h)

            crop_size = max(1024, min_side_img)
            crop_x = w/2 - crop_size/2
            crop_y = h/2 - crop_size/2

            center_crops += 1

        # Crop, resize (only if specified via argument) and save
        im_crop = im.crop((crop_x, crop_y, crop_x + crop_size, crop_y + crop_size))
        if args.crop is not None:
            im_crop = im_crop.resize((args.crop, args.crop))
        im_crop = im_crop.convert('RGB')
        im_crop.save(os.path.join(crop_dir, image_name), "JPEG")

    print(f'{center_crops} images had no label file and were center cropped')

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
    parser.add_argument('--clean', nargs='?', type=int, const=700, default=-1, help='Removes damaged image files and convertes healty ones to RGB JPEG. Also removes images with shorter length < [arg] (default=700).')
    parser.add_argument('--remove-duplicates', action='store_true', help='Removes duplicate images using Perceptual Hashing method.')
    parser.add_argument('--predict', action='store_true', help='Predicts corgi bounding boxes using pre-trained COCO Faster R-CNN and saves them into image_dir/labels.')
    parser.add_argument('--filter', nargs='?', type=float, const=0.9, default=-1.0, help='Moves all images with more than one dog bounding box to image_dir/supervision.')
    parser.add_argument('--crop', nargs='?', type=int, const=None, default=-1.0, help='Crops images based on predicted bounding box to quadratic image. You can specify target size via [arg] (default=do not resize).')
    parser.add_argument('--rename', action='store_true', help='Renames images and prediction files to [1,2,3,4,...,num_images].')
    parser.add_argument('--create-dummy-index', action='store_true', help='Create dummy .csv for annot_tool.py.')
    parser.add_argument('--index-to-labels', action='store_true', help='Creates prediction .pkl from index.csv.')

    args = parser.parse_args()
    image_dir = args.image_dir
    pred_dir = os.path.join(image_dir, 'labels')
    supervision_dir = os.path.join(image_dir, 'supervision')
    crop_dir = os.path.join(image_dir, 'crop')
    if args.crop is not None:
        crop_dir = os.path.join(image_dir, f'crop_{args.crop}')

    if args.create_dummy_index:
        create_dummy_index()
    elif args.index_to_labels:
        index_to_labels()

    if args.rename:
        rename()

    if args.clean != -1:
        clean()
    if args.predict:
        predict_labels()

    if args.crop != -1.0:
        crop()
    elif args.filter != -1.0:
        filter()

    if args.remove_duplicates:
        remove_duplicates()
