import shutil
import os
import glob
import cv2
from PIL import Image
import numpy as np

# print(111)
data_path='/home/ubuntu/Documents/harzva/ActiveLearning/prototypical-networks/data/omniglot/data/'
# print(glob.glob(data_path+'*/*/*.png'))
for j in glob.glob(data_path+'*/*/*.png'):
    # print(j)
    catCopyIm = Image.open(j)
    j=j.replace('omniglot','omniglot_rotpng')
    os.makedirs(os.path.dirname(j),exist_ok=True)
    # print(f'{j[:-4]}_rot000.png')
    # print(j)

    catCopyIm.rotate(0).save(j)


