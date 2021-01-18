import cv2
from  glob  import glob
image_file='/home/ubuntu/Documents/hzh/ActiveLearning/scatter_vic'
image_list=glob(image_file+'/*/*.png')#
for i in image_list:
    temp=cv2.imread(i)
    x, y = temp.shape[0:2]
    img_test1 = cv2.resize(temp, (int(y / 2), int(x / 2)))
    cv2.imwrite(i[:-4]+'_500.png',img_test1)