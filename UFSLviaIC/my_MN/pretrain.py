import os
import sys
import math
import torch
import random
import platform
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from alisuretool.Tools import Tools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from mn_fsl_test_tool import FSLTestTool
from fscifar_ic_res import ICResNet
from torchvision.models import resnet18, resnet34, resnet50, vgg16_bn
from mn_tool import MatchingNet, Normalize, RunnerTool,ResNet12Small,CNNEncoder,Classifier,ResNet12
import argparse
from tensorboardX import SummaryWriter
from logger import get_root_logger,collect_env
import time
import shutil
from datetime import datetime
import glob
import pysnooper
##############################################################################################################

def image_rotate(catCopyIm):
    list_=[0,90,180,270]
    degree=random.sample(list_, 1)[0]
    return catCopyIm.rotate(degree)
#@pysnooper.snoop("/home/ubuntu/Documents/hzh/ActivateLearning/UFSLviaIC/my_MN/debug.log", prefix="--*--")
#@pysnooper.snoop()
class CIFARDataset(object):
    print('*'*60,'CIFARDataset')

    def __init__(self, data_list, num_way, num_shot, image_size=32):
        print('*'*60,'__init__')
        self.data_list, self.num_way, self.num_shot = data_list, num_way, num_shot

        self.data_dict = {}
        for index, label, image_filename in self.data_list:
            if label not in self.data_dict:
                self.data_dict[label] = []
            self.data_dict[label].append((index, label, image_filename))
            pass

        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        # mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        normalize = transforms.Normalize(mean, std)
        change = transforms.Resize(image_size) if image_size > 32 else lambda x: x

        self.transform = transforms.Compose([
            change, transforms.RandomCrop(image_size, padding=4),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),

            # change, transforms.RandomResizedCrop(size=image_size),
            # transforms.ColorJitter(0.4, 0.4, 0.4, 0.4), transforms.RandomGrayscale(p=0.2),

            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        self.transform_test = transforms.Compose([change, transforms.ToTensor(), normalize])
        pass

    def __len__(self):
        print('*'*60,'__len__')
        return len(self.data_list)


    def __getitem__(self, item):
        print('*'*60,'__getitem__')
        # 当前样本
        now_label_image_tuple = self.data_list[item]
        now_index, now_label, now_image_filename = now_label_image_tuple
        now_label_k_shot_image_tuple = random.sample(self.data_dict[now_label], self.num_shot)

        # 其他样本
        other_label = list(self.data_dict.keys())
        other_label.remove(now_label)
        other_label = random.sample(other_label, self.num_way - 1)
        other_label_k_shot_image_tuple_list = []
        for _label in other_label:
            other_label_k_shot_image_tuple = random.sample(self.data_dict[_label], self.num_shot)
            other_label_k_shot_image_tuple_list.extend(other_label_k_shot_image_tuple)
            pass

        # c_way_k_shot
        c_way_k_shot_tuple_list = now_label_k_shot_image_tuple + other_label_k_shot_image_tuple_list
        random.shuffle(c_way_k_shot_tuple_list)

        task_list = c_way_k_shot_tuple_list + [now_label_image_tuple]
        task_data = torch.cat([torch.unsqueeze(self.read_image(one[2], self.transform), dim=0) for one in task_list])
        task_label = torch.Tensor([int(one_tuple[1] == now_label) for one_tuple in c_way_k_shot_tuple_list])
        task_index = torch.Tensor([one[0] for one in task_list]).long()
        return task_data, task_label, task_index
    @staticmethod
    def get_data_all(data_root):
        print('*'*60,'get_data_all')
        train_folder = os.path.join(data_root, "train")

        count_image, count_class, data_train_list = 0, 0, []
        for label in os.listdir(train_folder):
            now_class_path = os.path.join(train_folder, label)
            if os.path.isdir(now_class_path):
                for name in os.listdir(now_class_path):
                    data_train_list.append((count_image, count_class, os.path.join(now_class_path, name)))
                    count_image += 1
                    pass
                count_class += 1
            pass

        return data_train_list

    @staticmethod
    def read_image(image_path, transform=None):
        print('*'*60,'read_image')     
        image = Image.open(image_path).convert('RGB')
        if transform is not None:
            image = transform(image)
        return image
    print('*'*60,'CIFARDataset ender')
    pass


##############################################################################################################

#@pysnooper.snoop()
# @pysnooper.snoop("/home/ubuntu/Documents/hzh/ActivateLearning/UFSLviaIC/my_MN/debug.log", prefix="--*--")
class Runner(object):
    print('*'*60,'Runner')

    def __init__(self):
        # all data
        self.data_train = CIFARDataset.get_data_all(Config.data_root)
        # self.task_train = CIFARDataset(self.data_train, Config.num_way, Config.num_shot, image_size=Config.image_size)

        # model
        self.matching_net = RunnerTool.to_cuda(Config.matching_net)
        self.norm = Normalize(2)
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                            std= [x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_test = transforms.Compose(
                [transforms.ToTensor(), normalize]
        )

        # self.test_tool = FSLTestTool(self.matching_test, data_root=Config.data_root,
        #                              num_way=Config.num_way, num_shot=Config.num_shot,
        #                              episode_size=Config.episode_size, test_episode=Config.test_episode,
        #                              transform=self.task_train.transform_test,Config=Config)
        self.test_tool = FSLTestTool(self.matching_test, data_root=Config.data_root,
                                     num_way=Config.num_way, num_shot=Config.num_shot,
                                     episode_size=Config.episode_size, test_episode=Config.test_episode,
                                     transform=transform_test,Config=Config)
        pass

    def load_model(self):
        if os.path.exists(Config.mn_dir):
            self.matching_net.load_state_dict(torch.load(Config.mn_dir))
            Tools.print("load proto net success from {}".format(Config.mn_dir), txt_path=Config.log_file)
        pass

    # def load_model(self):
    #     pkl_list=glob.glob(f'{os.path.dirname(Config.mn_dir)}/*pkl')
    #     print(pkl_list)
    #     print(Config.mn_dir)
    #     assert len(pkl_list)==1 and os.path.exists(pkl_list[0]),'len(pkl_list) must is 1'
    #     self.matching_net.load_state_dict(torch.load(pkl_list[0]))
    #     Config.logger.info("load proto net success from {}".format(pkl_list[0]))
    #     pass

    def matching_test(self, samples, batches):
        batch_num, _, _, _ = batches.shape

        sample_z = self.matching_net(samples)[0]  # 5x64*5*5
        batch_z = self.matching_net(batches)[0]  # 75x64*5*5
        z_support = sample_z.view(Config.num_way * Config.num_shot, -1)
        z_query = batch_z.view(batch_num, -1)
        _, z_dim = z_query.shape

        z_support_expand = z_support.unsqueeze(0).expand(batch_num, Config.num_way * Config.num_shot, z_dim)
        z_query_expand = z_query.unsqueeze(1).expand(batch_num, Config.num_way * Config.num_shot, z_dim)

        # 相似性
        z_support_expand = self.norm(z_support_expand)
        similarities = torch.sum(z_support_expand * z_query_expand, -1)
        similarities = torch.softmax(similarities, dim=1)
        similarities = similarities.view(batch_num, Config.num_way, Config.num_shot)
        predicts = torch.mean(similarities, dim=-1)
        return predicts
    print('*'*60,'ender Runner')

    pass


##############################################################################################################
# @pysnooper.snoop("/home/ubuntu/Documents/hzh/ActivateLearning/UFSLviaIC/my_MN/debug.log", prefix="--*--")
#@pysnooper.snoop()
def parse_args():
    print('*'*60,' parse_args')
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--drop_rate', '-dr',type=float, default=0.2, help=' drop_rate')
    parser.add_argument('--gpu', '-g',type=str, default='01', help=' gpu')
    parser.add_argument('--batch_size', '-bs',type=int, default=64, help=' batchsize')
    parser.add_argument('--train_epoch', '-te',type=int, default=400, help='train_epoch')
    parser.add_argument('--debug','-d',  action='store_true', default=True,help=' debug')
    parser.add_argument('--fsl_backbone', '-fb',default='icres', help='fsl_backbone is c4')
    parser.add_argument('--num_way', '-w',type=int, default=5, help=' num_way=5')
    parser.add_argument('--num_shot', '-s',type=int, default=1, help=' num_shot=1')
    parser.add_argument('--val', '-v',type=str, default='UFSLviaIC/my_MN/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli_FC100/eval-res12/1_2100_64_5_1_500_200_512_1_1.0_1.0_head_png_res12_ic_FC100.pkl',help=' only val wegit _dir')
    parser.add_argument('--lr',type=int, default=3,help=' lr function id')
    parser.add_argument('--convert',type=str, default='RGB',help=' Image.open(x).convert(RGB)') 
    parser.add_argument('--dataset','-ds',type=str,default='FC100',help='dataset _name')     
    args = parser.parse_args()
    print('*'*60,' parse_args')
    return args
# @pysnooper.snoop("/home/ubuntu/Documents/hzh/ActivateLearning/UFSLviaIC/my_MN/debug.log", prefix="--*--")
#@pysnooper.snoop()
class Config(object):
    print('*'*60,'config')
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu#net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
    batch_size = args.batch_size
    train_epoch = args.train_epoch
    num_way =args.num_way
    num_shot = args.num_shot
    convert=args.convert
    input_dim=len(convert)
    num_workers = 8
    learning_rate = 0.01
    num_way_test = 5
    # num_shot_test = 0
    val_freq=1 if args.debug else 10
    episode_size = 15#MiniImageNetTask
    # episode_size = 100#my
    test_episode = 10#600代
    first_epoch, t_epoch = 200, 100
    if args.lr==1:
        adjust_learning_rate = RunnerTool.adjust_learning_rate1
    elif args.lr==2:
        adjust_learning_rate = RunnerTool.adjust_learning_rate2
    elif args.lr==3:
        adjust_learning_rate = RunnerTool.adjust_learning_rate3
    hid_dim = 64
    z_dim = 64
    is_png = True
    # is_png = False
    ###############################################################################################
    drop_rate=args.drop_rate

    if args.dataset=='omniglot' or args.dataset=='omniglot_single':
        print(f'your dataset is {args.dataset}')
        image_size=28
    
        transform = transforms.Compose([
        # lambda x: np.asarray(x),
        # lambda x: Image.open(x).convert('L'),
        lambda x: image_rotate(x),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        lambda x: x/255])#lambda x: np.asarray(x),
        transform_test = transforms.Compose(
                [transforms.ToTensor(),lambda x: x/255])
    elif args.dataset=='CIFARFS' or args.dataset=='FC100':
    # else:
        print(f'your dataset is {args.dataset}')
        image_size=32
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                            std= [x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform = transforms.Compose([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomGrayscale(p=0.2),   
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize,
            ]
        )
        transform_test = transforms.Compose(
                [transforms.ToTensor(), normalize]
        )
    dataset=args.dataset
    if  args.fsl_backbone=='res12small':
        matching_net = ResNet12Small(avg_pool=True, drop_rate=drop_rate,inplanes=input_dim)
        commit=f'{num_way}w{num_shot}s_DR{drop_rate}_{args.fsl_backbone}_lr{args.lr}_{dataset}_{convert}'
    elif args.fsl_backbone=='c4':
        matching_net = MatchingNet(hid_dim=hid_dim, z_dim=z_dim,input_dim=input_dim)
        commit=f'{num_way}w{num_shot}s_{args.fsl_backbone}_lr{args.lr}_{dataset}_{convert}'
    elif args.fsl_backbone=='mnc4':
        matching_net=Classifier(layer_size = hid_dim, num_channels=input_dim,nClasses= 0, image_size = image_size)
        commit=f'{num_way}w{num_shot}s_{args.fsl_backbone}_lr{args.lr}_{dataset}_{convert}'
    elif args.fsl_backbone=='rnc4':
        matching_net=CNNEncoder(input_dim=input_dim)#fsl conv4->res12
        commit=f'{num_way}w{num_shot}s_{args.fsl_backbone}_lr{args.lr}_{dataset}_{convert}'
    elif args.fsl_backbone=='res12N':
        matching_net = ResNet12(avg_pool=True, drop_rate=drop_rate,inplanes=input_dim)
        commit=f'{num_way}w{num_shot}s_{args.fsl_backbone}_lr{args.lr}_{dataset}_{convert}'
    elif args.fsl_backbone=='icres':
        matching_net= ICResNet(low_dim=512, modify_head=True, resnet=resnet34)
        commit=f'{num_way}w{num_shot}s_{args.fsl_backbone}_lr{args.lr}_{dataset}_{convert}'

    else:
        raise Exception



    DataParallel=True if len(args.gpu)>=2 else False

    ###############################################################################################
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')# or 
    model_name =f'EP{train_epoch}_BS{batch_size}_ft{first_epoch}_{t_epoch}_mn_{commit}'


    data_root = f'/home/ubuntu/Documents/hzh/ActiveLearning/data/{dataset}'
    if not os.path.exists(data_root):
        data_root = f'/home/ubuntu/Documents/hzh/ActivateLearning/data/{dataset}'
    
    _root_path = f"./models_mn/fsl_sgd_modify_{dataset}"
    # _root_path = "../models_rn/two_ic_ufsl_2net_res_sgd_acc_duli"
################################################################################################down is same
    if not args.debug:
        debug=""
        for i in glob.glob(_root_path+'/debug*'):
            shutil.rmtree(i)
            print(f'delete {i}')
    else:
        print(f'debug is {args.debug},and  you are debugging ')
        debug="debug"



    date_dir=f'{_root_path}/{debug+current_time}_{model_name}'
    mn_dir=Tools.new_dir(f"{date_dir}/{model_name}.pkl") if not args.val else args.val
    date_dir=f'{_root_path}/{debug+current_time}_{model_name}' if not args.val \
                        else Tools.new_dir(os.path.dirname(mn_dir)+f'/{debug}eval/')##
    writer = SummaryWriter(date_dir+'/runs')

    shutil.copy(os.path.abspath(sys.argv[0]),date_dir)

    log_file = os.path.join(date_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file,name=f"FSL-{commit}")
    for name, val in collect_env().items():
        logger.info(f'{name}: {val}')
    logger.info(model_name)
    logger.info(log_file)
    logger.info(f'DataParallel is {DataParallel}')
    logger.info(f"platform.platform{platform.platform()}")
    logger.info(f"config:   ")
    logger.info(f"args.gpu :{args.gpu} ,is_png:   {is_png},num_way_test:   {num_way_test}, test_episode:   {test_episode}") 
    logger.info(f"first_epoch:   {first_epoch},t_epoch:   {t_epoch}, val_freq:   {val_freq},episode_size:   {episode_size}")
    logger.info(f'hid_dim:   {hid_dim},z_dim:   {z_dim} , is_png:   {is_png},input_dim: {input_dim}')
    ABSPATH=os.path.abspath(sys.argv[0])

    pass


if __name__ == '__main__':
    runner = Runner()
    runner.load_model()
    runner.matching_net.eval()
    runner.test_tool.val(epoch=Config.train_epoch, is_print=True)
    runner.test_tool.test(test_avg_num=1, epoch=Config.train_epoch, is_print=True)
    pass
