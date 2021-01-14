import shutil
import os
import glob
# import cv2
from PIL import Image
import numpy as np
# import pysnooper
from tqdm import tqdm 
_CIFAR_CATEGORY_SPLITS_DIR='/home/test/Documents/hzh/ActiveLearning/data/omniglot/splits/vinyals'

def read_categories(filename):
    with open(filename) as f:
        categories = f.readlines()
    categories = [x.strip() for x in categories]
    return categories
# data_path='/home/test/Documents/hzh/ActiveLearning/prototypical-networks/data/omniglot/data/'
data_path='/home/test/Documents/hzh/ActiveLearning/data/omniglot/data/'
train_category_names =[glob.glob(data_path+os.path.dirname(i)+'/*.png') for i in read_categories(os.path.join(_CIFAR_CATEGORY_SPLITS_DIR, "train.txt"))]
val_category_names =[glob.glob(data_path+os.path.dirname(i)+'/*.png')  for i in read_categories(os.path.join(_CIFAR_CATEGORY_SPLITS_DIR, "val.txt"))]
test_category_names =[glob.glob(data_path+os.path.dirname(i)+'/*.png')  for i in read_categories(os.path.join(_CIFAR_CATEGORY_SPLITS_DIR, "test.txt"))]
def get_xdir(filename,x=1):
    if x>1:
        filename=os.path.dirname(filename)
        return get_xdir(filename,x-1)
    else:
        filename=os.path.basename(filename)
        return os.path.basename(filename)
filename=get_xdir(filename='/home/test/Documents/hzh/ActiveLearning/data/omniglot/test/Keble_character08/1253_03.png',x=3)




# @pysnooper.snoop()
def generate_omniglot(category_names=test_category_names,key='test',rot=True,datasetname='omniglot'):
    pbar = tqdm(total=len(category_names), ncols=50,desc=f"generate {key} {datasetname}")
    for i in category_names:
        for j in i:
            '''打开图像'''
            catCopyIm = Image.open(j)#/home/test/Documents/hzh/ActiveLearning/data/omniglot/data/Kannada/character34/1238_15.png
            # name2=os.path.basename(os.path.split(j)[0])+'/'+os.path.split(j)[1]#character47/1318_13.png
            name2=get_xdir(j,3)+'_'+get_xdir(j,2)+'/'+get_xdir(j,1)#character47/1318_13.png
            j=f'/home/test/Documents/hzh/ActiveLearning/data/{datasetname}/{key}/{name2}'
            os.makedirs(os.path.dirname(j),exist_ok=True)
            # print(j,'---------->',f'{j[:-4]}_rot000.png')

            catCopyIm.rotate(0).save(f'{j[:-4]}_rot000.png')
            if rot:
                # '''逆时针旋转90度的新Image图像'''
                catCopyIm.rotate(90).save(f'{j[:-4]}_rot090.png')

                # '''逆时针旋转180度的新Image图像'''
                catCopyIm.rotate(180).save(f'{j[:-4]}_rot180.png')

                # '''逆时针旋转270度的新Image图像'''
                catCopyIm.rotate(270).save(f'{j[:-4]}_rot270.png')
        pbar.update(1)
    pbar.close()
generate_omniglot(category_names=test_category_names,key='test',rot=False)   #1692  i=20 j为地址字符串的长度93+-
generate_omniglot(category_names=val_category_names,key='val',rot=False)   #688
generate_omniglot(category_names=train_category_names,key='train')   #4112

generate_omniglot(category_names=test_category_names,key='test',rot=False,datasetname='omniglot_single' )
generate_omniglot(category_names=val_category_names,key='val',rot=False,datasetname='omniglot_single' )
generate_omniglot(category_names=train_category_names,key='train',rot=False,datasetname='omniglot_single' )



     
