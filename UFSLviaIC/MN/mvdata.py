import shutil
import os
import glob
# _CIFAR_CATEGORY_SPLITS_DIR ="/home/ubuntu/Dataset/Partition1/hzh/data/cifar-100-python/cifar-fs_splits"
_CIFAR_CATEGORY_SPLITS_DIR='/home/ubuntu/Dataset/Partition1/hzh/data/omniglot/splits/vinyals'

# alphabet, character, rot = d['class'].split('/')
# image_dir = os.path.join(OMNIGLOT_DATA_DIR, 'data', alphabet, character)

# class_images = sorted(glob.glob(os.path.join(image_dir, '*.png')))
def read_categories(filename):
    with open(filename) as f:
        categories = f.readlines()
    categories = [x.strip() for x in categories]
    # print(categories)
    return categories
train_category_names =read_categories(
    os.path.join(_CIFAR_CATEGORY_SPLITS_DIR, "train.txt")
)
val_category_names =read_categories(
    os.path.join(_CIFAR_CATEGORY_SPLITS_DIR, "val.txt")
)
test_category_names = read_categories(
    os.path.join(_CIFAR_CATEGORY_SPLITS_DIR, "test.txt")
)
data_path='/home/ubuntu/Dataset/Partition1/hzh/data/omniglot/data/'
for i in train_category_names:
    #Angelic/character01/rot000=i
    i=glob.glob(data_path+os.path.dirname(i)+'/*.png')
    for j in i
    print(f"{i}",f"/home/ubuntu/Dataset/Partition1/hzh/data/omniglot/train/{i[:-6]}")


    # shutil.copytree(f"{i}",f"/home/ubuntu/Dataset/Partition1/hzh/data/omniglot/train/{os.path.dirname(i[:-6])}") 

#     # shutil.copytree(f"/home/ubuntu/Dataset/Partition1/hzh/data/cifar100/data/{i}",f"/home/ubuntu/Dataset/Partition1/hzh/data/CIFARFS/train/{i}") 
#     #oldfile只能是文件夹，newfile可以是文件，也可以是目标目录
# for i in train_category_names:
#     shutil.copytree(f"/home/ubuntu/Dataset/Partition1/hzh/data/omniglot/data/{i}",f"/home/ubuntu/Dataset/Partition1/hzh/data/omniglot/val/{i}") 
#     #oldfile只能是文件夹，newfile可以是文件，也可以是目标目录
# for i in train_category_names:
#     shutil.copytree(f"/home/ubuntu/Dataset/Partition1/hzh/data/omniglot/data/{i}",f"/home/ubuntu/Dataset/Partition1/hzh/data/omniglot/test/{i}") 
    #oldfile只能是文件夹，newfile可以是文件，也可以是目标目录
# filename=os.listdir("/home/ubuntu/Dataset/Partition1/hzh/data/FC100") 
# for i in filename:
#     print(i)
#     with open(f"/home/ubuntu/Dataset/Partition1/hzh/data/FC100/meta/{i}.txt","w") as f:
#         classame=os.listdir(f"/home/ubuntu/Dataset/Partition1/hzh/data/FC100/{i}") 
#         for j in classame:
#             print(j)
#             f.write(f'{j}\n')



# with open(filename) as f:
#     categories = f.readlines()
#     categories = [x.strip() for x in categories]
    # ABSPATH=os.path.abspath(sys.argv[0])
    # if test!='test':
    #     for i in glob.glob(os.path.dirname(writer_dir)+'/test*'):
    #         os.rmdir(i)
    #     for dor_path in file_list:
    #         dor_path=os.path.join(dir_or_file_path,dor_path)
    ''' ['mn_FC100_two_ic_ufsl_2net_res_sgd_acc_duli_tensorboard.py']
    mn_FC100_two_ic_ufsl_2net_res_sgd_acc_duli_tensorboard.py'''
    
        # print(sys.argv)
        # print(sys.argv[0])
        # print(ABSPATH)
        # print(os.path.dirname(ABSPATH))
    '''
    /home/ubuntu/Documents/hzh/ActiveLearning/UFSLviaIC/MN/mn_FC100_two_ic_ufsl_2net_res_sgd_acc_duli_tensorboard.py
    /home/ubuntu/Documents/hzh/ActiveLearning/UFSLviaIC/MN'''