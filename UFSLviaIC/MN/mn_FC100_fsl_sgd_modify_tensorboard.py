import os
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
from mn_miniimagenet_fsl_test_tool import TestTool
from mn_miniimagenet_tool import MatchingNet, Normalize, RunnerTool,ResNet12Small
from tensorboardX import SummaryWriter
from logger import get_root_logger,collect_env
import time
import shutil
from datetime import datetime
import glob
import sys
import argparse
##############################################################################################################
'''
1.tensorboard
2.res12
3.ominiglot
'''

class FC100Dataset(object):

    def __init__(self, data_list, num_way, num_shot):#故此函数被声明为私有方法，不可类外调用。
        self.data_list, self.num_way, self.num_shot = data_list, num_way, num_shot

        self.data_dict = {}
        for index, label, image_filename in self.data_list:
            if label not in self.data_dict:
                self.data_dict[label] = []
            self.data_dict[label].append((index, label, image_filename))
            #(0, 0, '/mnt/4T/Data/data/miniImagenet/miniImageNet_png/train/n03838899/38081.png')
            pass


        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std= [x / 255.0 for x in [63.0, 62.1, 66.7]])
        self.transform = transforms.Compose([
                transforms.Resize([84, 84]),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomGrayscale(p=0.2),   
                transforms.RandomCrop(84, padding=4),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.transform_test = transforms.Compose(
                [transforms.Resize((84,84)),transforms.ToTensor(), normalize]
        )
        pass

    def __len__(self):
        return len(self.data_list)
    
    @staticmethod
    # @pysnooper.snoop("/home/ubuntu/Documents/hzh/ActiveLearning-master/UFSLviaIC/MN/log/debug.log", prefix="--*--")
    def get_data_all(data_root):
        train_folder = os.path.join(data_root, "train")
        count_image, count_class, data_train_list = 0, 0, []
        for label in os.listdir(train_folder):
            now_class_path = os.path.join(train_folder, label)#''/home/ubuntu/Documents/hzh/ActiveLearning-master/data/cifar100/data/train'
            if os.path.isdir(now_class_path):
                for name in os.listdir(now_class_path):
                    data_train_list.append((count_image, count_class, os.path.join(now_class_path, name)))
                    count_image += 1
                    pass
                count_class += 1
            pass

        return data_train_list
    # @pysnooper.snoop()
    def __getitem__(self, item):
        # 当前样本
        now_label_image_tuple = self.data_list[item]#(9891, 16, '/mnt/4T/Data/data/mi.../10836.png')
        now_index, now_label, now_image_filename = now_label_image_tuple
        now_label_k_shot_image_tuple = random.sample(self.data_dict[now_label], self.num_shot)#[(9868, 16, '/mnt/4T/Data/data/mi.../10854.png')]

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
        #data_batch_size, data_image_num, data_num_channel, data_width, data_weight 
        task_data = torch.cat([torch.unsqueeze(self.read_image(one[2], self.transform), dim=0) for one in task_list])# torch.Size([64, 6, 1, 3, 32, 32])
        task_label = torch.Tensor([int(one_tuple[1] == now_label) for one_tuple in c_way_k_shot_tuple_list])
        task_index = torch.Tensor([one[0] for one in task_list]).long()

        return task_data, task_label, task_index

    @staticmethod
    def read_image(image_path, transform=None):
        image = Image.open(image_path).convert(Config.convert)
        loader = transforms.Compose([transforms.ToTensor()])  
        if transform is not None:
            image = transform(image)
        else:
            image = loader(image).unsqueeze(0)[0]# torch.Size([1, 3, 32, 32])
        return image

    pass


##############################################################################################################


class Runner(object):

    def __init__(self):
        self.best_accuracy = 0.0
        self.adjust_learning_rate = Config.adjust_learning_rate

        # all data
        self.data_train = FC100Dataset.get_data_all(Config.data_root)
        self.task_train = FC100Dataset(self.data_train, Config.num_way, Config.num_shot)
        self.task_train_loader = DataLoader(self.task_train, Config.batch_size, True, num_workers=Config.num_workers)

        # model
        self.matching_net = RunnerTool.to_cuda(Config.matching_net)
        RunnerTool.to_cuda(self.matching_net.apply(RunnerTool.weights_init))
        self.norm = Normalize(2)

        # loss
        self.loss = RunnerTool.to_cuda(nn.MSELoss())

        # optim
        self.matching_net_optim = torch.optim.SGD(
            self.matching_net.parameters(), lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)

        if Config.DataParallel:
            self.matching_net  = torch.nn.DataParallel(self.matching_net , device_ids=range(torch.cuda.device_count()))
            Config.logger.info(f'torch.cuda.device_count()  {range(torch.cuda.device_count())}')


        self.test_tool = TestTool(self.matching_test, data_root=Config.data_root,
                                  num_way=Config.num_way_test,  num_shot=Config.num_shot,
                                  episode_size=Config.episode_size, test_episode=Config.test_episode,
                                  transform=self.task_train.transform_test,Config=Config)
        pass

    def load_model(self):
        pkl_list=glob.glob(f'{os.path.dirname(Config.mn_dir)}/*pkl')
        assert len(pkl_list)==1 and os.path.exists(pkl_list[0]),'len(pkl_list) must is 1'
        self.matching_net.load_state_dict(torch.load(pkl_list[0]))
        Config.logger.info("load proto net success from {}".format(pkl_list[0]))
        pass

    def matching(self, task_data):
        data_batch_size, data_image_num, data_num_channel, data_width, data_weight = task_data.shape
        data_x = task_data.view(-1, data_num_channel, data_width, data_weight)
        net_out = self.matching_net(data_x)
        z = net_out.view(data_batch_size, data_image_num, -1)

        # 特征
        z_support, z_query = z.split(Config.num_shot * Config.num_way, dim=1)
        z_batch_size, z_num, z_dim = z_support.shape
        z_support = z_support.view(z_batch_size, Config.num_way * Config.num_shot, z_dim)
        z_query_expand = z_query.expand(z_batch_size, Config.num_way * Config.num_shot, z_dim)

        # 相似性
        z_support = self.norm(z_support)
        similarities = torch.sum(z_support * z_query_expand, -1)
        similarities = torch.softmax(similarities, dim=1)
        similarities = similarities.view(z_batch_size, Config.num_way, Config.num_shot)
        predicts = torch.mean(similarities, dim=-1)
        return predicts

    def matching_test(self, samples, batches):
        batch_num, _, _, _ = batches.shape

        sample_z = self.matching_net(samples)  # 5x64*5*5
        batch_z = self.matching_net(batches)  # 75x64*5*5
        z_support = sample_z.view(Config.num_way_test * Config.num_shot, -1)
        z_query = batch_z.view(batch_num, -1)
        _, z_dim = z_query.shape

        z_support_expand = z_support.unsqueeze(0).expand(batch_num, Config.num_way_test * Config.num_shot, z_dim)
        z_query_expand = z_query.unsqueeze(1).expand(batch_num, Config.num_way_test * Config.num_shot, z_dim)

        # 相似性
        z_support_expand = self.norm(z_support_expand)
        similarities = torch.sum(z_support_expand * z_query_expand, -1)
        similarities = torch.softmax(similarities, dim=1)
        similarities = similarities.view(batch_num, Config.num_way_test, Config.num_shot)
        predicts = torch.mean(similarities, dim=-1)
        return predicts

    def train(self):
        Config.logger.info("Training...")

        for epoch in range(1, 1 + Config.train_epoch):
            self.matching_net.train()

            mn_lr= self.adjust_learning_rate(self.matching_net_optim, epoch,
                                             Config.first_epoch, Config.t_epoch, Config.learning_rate)

   
            Config.writer.add_scalar('mn_lr', mn_lr, epoch)#name y x

            all_loss = 0.0
            for task_data, task_labels, task_index in tqdm(self.task_train_loader):
                task_data, task_labels = RunnerTool.to_cuda(task_data), RunnerTool.to_cuda(task_labels)

                # 1 calculate features
                predicts = self.matching(task_data)

                # 2 loss
                loss = self.loss(predicts, task_labels)
                all_loss += loss.item()

                # 3 backward
                self.matching_net.zero_grad()
                loss.backward()
                self.matching_net_optim.step()
                ###########################################################################
                pass

            ###########################################################################
            # print
            Config.logger.info('Epoch:[{}] mn_lr={} loss:{:.5f}'.format(epoch, mn_lr,all_loss / len(self.task_train_loader)))
            Config.writer.add_scalar('loss', all_loss / len(self.task_train_loader), epoch)#name y x
            ###########################################################################

            ###########################################################################
            # Val
            if epoch % Config.val_freq == 0:
                Config.logger.info("Test {} {} .......".format(epoch, Config.model_name))
                self.matching_net.eval()

                val_accuracy = self.test_tool.val(episode=epoch, is_print=True)
                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    for i in glob.glob(os.path.dirname(Config.mn_dir)+'/*pkl'):
                        Config.logger.info("delete {}".format(i))
                        os.remove(i)
                    torch.save(self.matching_net.state_dict(),os.path.dirname(Config.mn_dir)+'/'+'fsl-'+str(epoch)+os.path.basename(Config.mn_dir))
                    Config.logger.info("Save networks for epoch: {}".format(epoch))
                    Config.logger.info("Save {}".format(os.path.dirname(Config.mn_dir)+'/'+'fsl-'+str(epoch)+os.path.basename(Config.mn_dir)))
    
                    pass
                pass
            ###########################################################################os.path.split(path) 
            pass

        pass

    pass

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--drop_rate', '-dr',type=float, default=0.2, help=' drop_rate')
    parser.add_argument('--gpu', '-g',type=str, default='0', help=' gpu')
    parser.add_argument('--batch_size', '-bs',type=int, default=64, help=' batchsize')
    parser.add_argument('--train_epoch', '-te',type=int, default=400, help='train_epoch')
    parser.add_argument('--debug','-d',  action='store_true', default=False,help=' debug')
    parser.add_argument('--fsl_backbone', '-fb',default='c4', help='fsl_backbone is c4')
    parser.add_argument('--num_way', '-w',type=int, default=5, help=' num_way=5')
    parser.add_argument('--num_shot', '-s',type=int, default=1, help=' num_shot=1')
    parser.add_argument('--val', '-v',type=str, default='',help=' only val wegit _dir')
    parser.add_argument('--lr',type=int, default=2,help=' lr function id')
    parser.add_argument('--convert',type=str, default='RGB',help=' Image.open(x).convert(RGB)')    
    args = parser.parse_args()
    return args
class Config(object):
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
    
    val_freq=1 if args.debug else 10
    episode_size = 15#MiniImageNetTask
    test_episode = 600 #600代
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
    dataset='FC100'
    image_size=84
    if  args.fsl_backbone=='res12':
        matching_net = ResNet12Small(avg_pool=True, drop_rate=drop_rate,inplanes=len(args.convert))
        commit=f'{num_way}w{num_shot}s_DR{drop_rate}_{args.fsl_backbone}_lr{args.lr}_{dataset}_{convert}'
    elif args.fsl_backbone=='c4':
        matching_net = MatchingNet(hid_dim=hid_dim,input_dim=input_dim, z_dim=z_dim)
        commit=f'{num_way}w{num_shot}s_{args.fsl_backbone}_lr{args.lr}_{dataset}_{convert}'
    elif args.fsl_backbone=='mnc4':
        matching_net=Classifier(layer_size = hid_dim, num_channels=input_dim,nClasses= 0, image_size = image_size)
        commit=f'{num_way}w{num_shot}s_{args.fsl_backbone}_lr{args.lr}_{dataset}_{convert}'
    elif args.fsl_backbone=='rnc4':
        matching_net=CNNEncoder()#fsl conv4->res12
        commit=f'{num_way}w{num_shot}s_{args.fsl_backbone}_lr{args.lr}_{dataset}_{convert}'
    DataParallel=True if len(args.gpu)>=2 else False
    ###############################################################################################
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')# or 
    model_name =f'EP{train_epoch}_BS{batch_size}_ft{first_epoch}_{t_epoch}_mn_{commit}'


    data_root = '/home/ubuntu/Dataset/Partition1/hzh/data/FC100'
    if not os.path.exists(data_root):
        data_root = '/home/test/Documents/hzh/ActiveLearning/data/FC100'
    
    _root_path = "../models_mn/fsl_sgd_modify"
    # _root_path = "../models_rn/two_ic_ufsl_2net_res_sgd_acc_duli"
################################################################################################down is same
    if not args.debug:
        debug=""
        for i in glob.glob(_root_path+'/debug*'):
            shutil.rmtree(i)
            print(f'delete {i}')
    else:
            debug="debug"
            print(f'you are debugging ')
    date_dir=f'{_root_path}/{debug+current_time}_{model_name}'
    mn_dir=Tools.new_dir(f"{date_dir}/{model_name}.pkl") if not args.val else args.val
    writer = SummaryWriter(date_dir+'/runs')

    shutil.copy(os.path.abspath(sys.argv[0]),date_dir)

    log_file = os.path.join(date_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file,name=f"FSL-{commit}")
    for name, val in collect_env().items():
        logger.info(f'{name}: {val}')
    logger.info(model_name)
    logger.info(f'DataParallel is {DataParallel}')
    logger.info(f"platform.platform{platform.platform()}")
    logger.info(f"config:   ")
    logger.info(f"args.gpu :{args.gpu} ,is_png:   {is_png},num_way_test:   {num_way_test}, test_episode:   {test_episode}") 
    logger.info(f"first_epoch:   {first_epoch},t_epoch:   {t_epoch}, val_freq:   {val_freq},episode_size:   {episode_size}")
    logger.info(f'hid_dim:   {hid_dim},z_dim:   {z_dim} , is_png:   {is_png},input_dim: {input_dim}')
    ABSPATH=os.path.abspath(sys.argv[0])

    pass

if __name__ == '__main__':
    # init the logger before other steps
    args = parse_args()
    runner = Runner()
    if not args.val:
        runner.train()
    runner.load_model()
    runner.matching_net.eval()#zhengze drop  
    runner.test_tool.val(episode=Config.train_epoch, is_print=True)
    runner.test_tool.test(test_avg_num=5, episode=Config.train_epoch, is_print=True)
    pass

"""
4conv
2020-12-16 03:54:40 Test 400 2_400_64_5_1_200_100_png .......
2020-12-16 03:56:38 Train 400 Accuracy: 0.8914444444444446
2020-12-16 03:56:38 Val   400 Accuracy: 0.30077777777777776
2020-12-16 03:56:38 Test1 400 Accuracy: 0.36833333333333335
2020-12-16 03:56:38 Test2 400 Accuracy: 0.36895555555555554
2020-12-16 03:56:38 load proto net success from ../models_mn/fsl/2_400_64_5_1_200_100_png_mn_5way_1shot_FC100.pkl
2020-12-16 03:58:35 Train 400 Accuracy: 0.666
2020-12-16 03:58:35 Val   400 Accuracy: 0.3071111111111111
2020-12-16 03:58:35 Test1 400 Accuracy: 0.356
2020-12-16 03:58:35 Test2 400 Accuracy: 0.3651111111111111
2020-12-16 04:03:17 episode=400, Test accuracy=0.36291111111111113
2020-12-16 04:03:17 episode=400, Test accuracy=0.36415555555555557
2020-12-16 04:03:17 episode=400, Test accuracy=0.3675111111111111
2020-12-16 04:03:17 episode=400, Test accuracy=0.36611111111111116
2020-12-16 04:03:17 episode=400, Test accuracy=0.3665111111111111
2020-12-16 04:03:17 episode=400, Mean Test accuracy=0.36544

res12


"""
