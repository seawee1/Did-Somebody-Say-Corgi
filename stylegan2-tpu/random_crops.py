import argparse, os
import numpy as np
from PIL import Image
from random import randrange
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True, help="Input directory of images to randomly crop.")
parser.add_argument('-o', '--output', type=str, required=True, help="Output destination to store randomly-cropped images.")
parser.add_argument('-n', '--max_images', type=int, required=True, help="Max images to randomly crop")
args = parser.parse_args()

def random_crop(image, target, samples, file): 
    x = image.width
    y = image.height
    for i in range(samples):
        x1 = randrange(0, x - target)
        y1 = randrange(0, y - target)
        image.crop((x1, y1, x1 + target, y1 + target)).save(target_directory + "/" + str(i) + file)


directory = args.input
target_directory = args.output
max_images = args.max_images
added = 0

with tqdm(total=max_images) as pbar:
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpeg") and added != max_images:
                if "._" in filepath:
                    continue
                im = Image.open(filepath).convert('RGB')
                random_crop(im, 512, 5, file)
                added += 1
                pbar.update(1)
            else:
                break
