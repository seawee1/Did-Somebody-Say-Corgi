import argparse, os
import numpy as np
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True, help="Input directory of images to resize.")
parser.add_argument('-o', '--output', type=str, required=True, help="Output destination to store resized images.")
args = parser.parse_args()

# Artistic crop: In-frame, centered mammogram.              
def flipImage(image):
    def isLeftEdgeBlank(image):
        return image[:, -1].sum(axis=0) > image[:, 0].sum(axis=0) 
    return image if isLeftEdgeBlank(np.asarray(image)).all() else image.transpose(Image.FLIP_LEFT_RIGHT)

directory = args.input
target_directory = args.output

# This is a waste of memory and compute, 
# but it's nice to have an idea on the progress of the script.
filescount = 0
for dirPath, subdirList, fileList in os.walk(directory):
    filescount += len(list(filter(lambda x: x.endswith(".jpeg"), fileList)))

with tqdm(total=filescount) as pbar:
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpeg"):
                if "._" in filepath:
                    continue
                im = Image.open(filepath).convert('RGB')
                img = flipImage(im)
                img.save(target_directory + '/' + file)
                pbar.update(1)


