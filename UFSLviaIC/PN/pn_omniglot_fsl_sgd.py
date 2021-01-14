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
from pn_miniimagenet_fsl_test_tool import TestTool
from pn_miniimagenet_tool import ProtoNet, RunnerTool
from tensorboardX import SummaryWriter
from logger import get_root_logger,collect_env
import time
import shutil
from datetime import datetime
import glob
import sys
import argparse

##############################################################################################################
def image_rotate(catCopyIm):
    list_=[0,90,180,270]
    degree=random.sample(list_, 1)[0]
    return catCopyIm.rotate(degree)

class ominiglotDataset(object):

    def __init__(self, data_list, num_way, num_shot):
        self.data_list, self.num_way, self.num_shot = data_list, num_way, num_shot

        self.data_dict = {}
        for index, label, image_filename in self.data_list:
            if label not in self.data_dict:
                self.data_dict[label] = []
            self.data_dict[label].append((index, label, image_filename))
            pass

        self.transform = transforms.Compose([
                # lambda x: np.asarray(x),
                # lambda x: Image.open(x).convert('RGB'),
                lambda x: image_rotate(x),
                transforms.Resize((28,28)),
                transforms.ToTensor(),
                lambda x: x/255
            ]
        )#lambda x: np.asarray(x),
        self.transform_test = transforms.Compose(
                [transforms.Resize((28,28)), transforms.ToTensor(),lambda x: x/255]
        )
        pass

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def get_data_all(data_root):
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

    def __getitem__(self, item):
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
    def read_image(image_path, transform=None):
        image = Image.open(image_path).convert('RGB')
        if transform is not None:
            image = transform(image)
        return image

    pass


##############################################################################################################


class Runner(object):

    def __init__(self):
        self.best_accuracy = 0.0
        self.adjust_learning_rate = Config.adjust_learning_rate

        # all data
        self.data_train = ominiglotDataset.get_data_all(Config.data_root)
        self.task_train = ominiglotDataset(self.data_train, Config.num_way, Config.num_shot)
        self.task_train_loader = DataLoader(self.task_train, Config.batch_size, shuffle=True, num_workers=Config.num_workers)

        # model
        self.proto_net = RunnerTool.to_cuda(Config.proto_net)
        RunnerTool.to_cuda(self.proto_net.apply(RunnerTool.weights_init))

        # optim
        self.proto_net_optim = torch.optim.SGD(
            self.proto_net.parameters(), lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)

        self.test_tool = TestTool(self.proto_test, data_root=Config.data_root,
                                  num_way=Config.num_way,  num_shot=Config.num_shot,
                                  episode_size=Config.episode_size, test_episode=Config.test_episode,
                                  transform=self.task_train.transform_test)
        pass

    def load_model(self):
        if os.path.exists(Config.pn_dir):
            self.proto_net.load_state_dict(torch.load(Config.pn_dir))
            Config.logger.info("load proto net success from {}".format(Config.pn_dir))
        pass

    def proto(self, task_data):
        data_batch_size, data_image_num, data_num_channel, data_width, data_weight = task_data.shape
        data_x = task_data.view(-1, data_num_channel, data_width, data_weight)
        net_out = self.proto_net(data_x)
        z = net_out.view(data_batch_size, data_image_num, -1)

        z_support, z_query = z.split(Config.num_shot * Config.num_way, dim=1)
        z_batch_size, z_num, z_dim = z_support.shape
        z_support = z_support.view(z_batch_size, Config.num_way, Config.num_shot, z_dim)

        z_support_proto = z_support.mean(2)
        z_query_expand = z_query.expand(z_batch_size, Config.num_way, z_dim)

        dists = torch.pow(z_query_expand - z_support_proto, 2).sum(2)
        log_p_y = F.log_softmax(-dists, dim=1)
        return log_p_y

    def proto_test(self, samples, batches, num_way, num_shot):
        batch_num, _, _, _ = batches.shape

        sample_z = self.proto_net(samples)  # 5x64*5*5
        batch_z = self.proto_net(batches)  # 75x64*5*5
        sample_z = sample_z.view(num_way, num_shot, -1)
        batch_z = batch_z.view(batch_num, -1)
        _, z_dim = batch_z.shape

        z_proto = sample_z.mean(1)
        z_proto_expand = z_proto.unsqueeze(0).expand(batch_num, num_way, z_dim)
        z_query_expand = batch_z.unsqueeze(1).expand(batch_num, num_way, z_dim)

        dists = torch.pow(z_query_expand - z_proto_expand, 2).sum(2)
        log_p_y = F.log_softmax(-dists, dim=1)
        return log_p_y

    def train(self):
        Config.logger.info("Training...")

        for epoch in range(1, 1 + Config.train_epoch):
            self.proto_net.train()


            all_loss = 0.0
            pn_lr = self.adjust_learning_rate(self.proto_net_optim, epoch,
                                              Config.first_epoch, Config.t_epoch, Config.learning_rate)
            Config.logger.info('Epoch: [{}] pn_lr={}'.format(epoch, pn_lr))

            for task_data, task_labels, task_index in tqdm(self.task_train_loader):
                task_data, task_labels = RunnerTool.to_cuda(task_data), RunnerTool.to_cuda(task_labels)

                # 1 calculate features
                log_p_y = self.proto(task_data)

                # 2 loss
                loss = -(log_p_y * task_labels).sum() / task_labels.sum()
                all_loss += loss.item()

                # 3 backward
                self.proto_net.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.proto_net.parameters(), 0.5)
                self.proto_net_optim.step()
                ###########################################################################
                pass

            ###########################################################################
            # print
            Config.logger.info("{:6} loss:{:.3f}".format(epoch, all_loss / len(self.task_train_loader)))
            ###########################################################################

            ###########################################################################
            # Val
            if epoch % Config.val_freq == 0:
                Config.logger.info("Test {} {} .......".format(epoch, Config.model_name))
                self.self.proto_net.eval()

                val_accuracy = self.test_tool.val(episode=epoch, is_print=True,Config=Config)
                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    for i in glob.glob(os.path.dirname(Config.mn_dir)+'/*pkl'):
                        Config.logger.info("delete {}".format(i))
                        os.remove(i)
                    torch.save(self.self.proto_net.state_dict(),os.path.dirname(Config.mn_dir)+'/'+'fsl-'+str(epoch)+os.path.basename(Config.mn_dir))
                    Config.logger.info("Save networks for epoch: {}".format(epoch))
                    Config.logger.info("Save {}".format(os.path.dirname(Config.mn_dir)+'/'+'fsl-'+str(epoch)+os.path.basename(Config.mn_dir)))
    
                    pass
                pass
            ###########################################################################os.path.split(path) 
            pass

        pass

    pass


##############################################################################################################


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--drop_rate', '-dr',type=float, default=0.2, help=' drop_rate')
    parser.add_argument('--gpu', '-g',type=str, default='0', help=' gpu')
    parser.add_argument('--batch_size', '-bs',type=int, default=64, help=' batchsize')
    parser.add_argument('--train_epoch', '-te',type=int, default=500, help='train_epoch')
    parser.add_argument('--debug','-d',  action='store_true', help=' debug')
    parser.add_argument('--res12', '-r12',action='store_true', help='fsl_backbone is res12')
    parser.add_argument('--num_way', '-w',type=int, default=5, help=' num_way=5')
    parser.add_argument('--num_shot', '-s',type=int, default=1, help=' num_shot=1')
    parser.add_argument('--val', '-v',type=str, default='',help=' only val wegit _dir')
    parser.add_argument('--lr',type=int, default=2,help=' lr function id')  
      
    args = parser.parse_args()
    return args

class Config(object):
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    batch_size = args.batch_size
    train_epoch = args.train_epoch
    num_way =args.num_way
    num_shot = args.num_shot
    num_workers = 8
    learning_rate = 0.01
    num_way_test = 5
    # num_shot_test = 0
    val_freq=1 if args.debug else 10
    episode_size = 15#MiniImageNetTask
    has_norm = False
    # episode_size = 100#my
    test_episode = 600#600代
    first_epoch, t_epoch = 200,150
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
    dataset='omniglot_single'
    proto_net = ProtoNet(hid_dim=hid_dim, z_dim=z_dim, has_norm=has_norm)#fsl conv4->res12
    fsl_backbone='c4'
    commit=f'{num_way}w{num_shot}s_{fsl_backbone}_lr{args.lr}_{dataset}_weights_init_normal'
    DataParallel=True if len(args.gpu)>=2 else False

    ###############################################################################################
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')# or 
    model_name =f'EP{train_epoch}_BS{batch_size}_ft{first_epoch}_{t_epoch}_mn_{commit}'


    data_root = f'/home/ubuntu/Documents/hzh/ActiveLearning/data/{dataset}'
    if not os.path.exists(data_root):
        data_root = f'/home/test/Documents/hzh/ActiveLearning/data/{dataset}'
    
    _root_path = "../models_pn/fsl_sgd_modify"
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
    pn_dir=Tools.new_dir(f"{date_dir}/{model_name}.pkl") if not args.val else args.val
    writer = SummaryWriter(date_dir+'/runs')

    shutil.copy(os.path.abspath(sys.argv[0]),date_dir)

    log_file = os.path.join(date_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file,name=f"PN-{commit}")#orUfslPN
    logger.info(model_name)
    logger.info(f'DataParallel is {DataParallel}')
    logger.info(f"platform.platform{platform.platform()}")
    logger.info(f"config:   ")
    logger.info(f"args.gpu :{args.gpu} ,is_png:   {is_png},num_way_test:   {num_way_test}, test_episode:   {test_episode}") 
    logger.info(f"first_epoch:   {first_epoch},t_epoch:   {t_epoch}, val_freq:   {val_freq},episode_size:   {episode_size}")
    logger.info(f'hid_dim:   {hid_dim},z_dim:   {z_dim} , is_png:   {is_png},input_dim: {input_dim}')
    ABSPATH=os.path.abspath(sys.argv[0])

    pass
##############################################################################################################

if __name__ == '__main__':
    # init the logger before other steps
    args = parse_args()
    runner = Runner()
    if not args.val:
        runner.train()
    runner.load_model()
    runner.proto_net.eval()#zhengze drop  
    runner.test_tool.val(episode=Config.train_epoch, is_print=True,Config=Config)
    runner.test_tool.test(test_avg_num=5, episode=Config.train_epoch, is_print=True,Config=Config)
    pass