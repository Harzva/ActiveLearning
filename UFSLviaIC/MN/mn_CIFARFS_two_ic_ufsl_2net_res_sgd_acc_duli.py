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
from torchvision.models import resnet18, resnet34
from mn_miniimagenet_fsl_test_tool import TestTool
from mn_miniimagenet_ic_test_tool import ICTestTool
from mn_miniimagenet_tool import MatchingNet, Normalize, RunnerTool,ResNet12Small,CNNEncoder,Classifier,CNNEncoder,Classifier
from tensorboardX import SummaryWriter
import sys
from logger import get_root_logger,collect_env
import time
import socket
from datetime import datetime
import shutil
import glob
import argparse


##############################################################################################################


class CIFARFSDataset(object):

    def __init__(self, data_list, num_way, num_shot):
        self.data_list, self.num_way, self.num_shot = data_list, num_way, num_shot
        self.data_id = np.asarray(range(len(self.data_list)))

        self.classes = None
        self.features = None

        self.data_dict = {}
        for index, label, image_filename in self.data_list:
            if label not in self.data_dict:
                self.data_dict[label] = []
            self.data_dict[label].append((index, label, image_filename))
            pass

        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std= [x / 255.0 for x in [63.0, 62.1, 66.7]])

        self.transform_train_ic = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4), transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])

        self.transform_train_fsl = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.transform_test = transforms.Compose(
                [transforms.Resize((32,32)), transforms.ToTensor(), normalize]
        )
        pass

    def __len__(self):
        return len(self.data_list)

    def set_samples_class(self, classes):
        self.classes = classes
        pass

    def set_samples_feature(self, features):
        self.features = features
        pass

    def __getitem__(self, item):
        # 当前样本
        now_label_image_tuple = self.data_list[item]
        now_index, _, now_image_filename = now_label_image_tuple
        _now_label = self.classes[item]

        now_label_k_shot_index = self._get_samples_by_clustering_label(_now_label, True, num=self.num_shot)
        # now_label_k_shot_index = self._get_samples_by_clustering_label(_now_label, True, num=self.num_shot, now_index=now_index)

        is_ok_list = [self.data_list[one][1] == now_label_image_tuple[1] for one in now_label_k_shot_index]

        # 其他样本
        other_label_k_shot_index_list = self._get_samples_by_clustering_label(_now_label, False,
                                                                              num=self.num_shot * (self.num_way - 1))

        # c_way_k_shot
        c_way_k_shot_index_list = now_label_k_shot_index + other_label_k_shot_index_list
        random.shuffle(c_way_k_shot_index_list)

        if len(c_way_k_shot_index_list) != self.num_shot * self.num_way:
            return self.__getitem__(random.sample(list(range(0, len(self.data_list))), 1)[0])

        task_list = [self.data_list[index] for index in c_way_k_shot_index_list] + [now_label_image_tuple]

        task_data = []
        for one in task_list:
            transform = self.transform_train_ic if one[2] == now_image_filename else self.transform_train_fsl
            task_data.append(torch.unsqueeze(self.read_image(one[2], transform), dim=0))
            pass
        task_data = torch.cat(task_data)

        task_label = torch.Tensor([int(index in now_label_k_shot_index) for index in c_way_k_shot_index_list])
        task_index = torch.Tensor([one[0] for one in task_list]).long()
        return task_data, task_label, task_index, is_ok_list

    def _get_samples_by_clustering_label(self, label, is_same_label=False, num=1, now_index=None, k=1):
        if is_same_label:
            if now_index:
                now_feature = self.features[now_index]

                if k == 1:
                    search_index = self.data_id[self.classes == label]
                else:
                    top_k_class = np.argpartition(now_feature, -k)[-k:]
                    search_index = np.hstack([self.data_id[self.classes == one] for one in top_k_class])
                    pass

                search_index_list = list(search_index)
                if now_index in search_index_list:
                    search_index_list.remove(now_index)
                other_features = self.features[search_index_list]

                # sim_result = np.matmul(other_features, now_feature)
                now_features = np.tile(now_feature[None, ...], reps=[other_features.shape[0], 1])
                sim_result = np.sum(now_features * other_features, axis=-1)

                sort_result = np.argsort(sim_result)[::-1]
                return list(search_index[sort_result][0: num])
            return random.sample(list(np.squeeze(np.argwhere(self.classes == label), axis=1)), num)
        else:
            return random.sample(list(np.squeeze(np.argwhere(self.classes != label))), num)
        pass

    @staticmethod
    def read_image(image_path, transform=None):
        image = Image.open(image_path).convert(Config.convert)
        if transform is not None:
            image = transform(image)
        return image

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

    pass


##############################################################################################################


class ICResNet(nn.Module):

    def __init__(self, resnet, low_dim=512, modify_head=False,input_dim=3):
        super().__init__()
        self.resnet = resnet(num_classes=low_dim)
        self.l2norm = Normalize(2)
        if modify_head:
            self.resnet.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)
            pass
        pass
    def forward(self, x):
        out_logits = self.resnet(x)
        out_l2norm = self.l2norm(out_logits)
        return out_logits, out_l2norm

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class ProduceClass(object):

    def __init__(self, n_sample, out_dim, ratio=1.0):
        super().__init__()
        self.out_dim = out_dim
        self.n_sample = n_sample
        self.class_per_num = self.n_sample // self.out_dim * ratio
        self.count = 0
        self.count_2 = 0
        self.class_num = np.zeros(shape=(self.out_dim,), dtype=np.int)
        self.classes = np.zeros(shape=(self.n_sample,), dtype=np.int)
        self.features = np.random.random(size=(self.n_sample, self.out_dim))
        pass

    def init(self):
        class_per_num = self.n_sample // self.out_dim
        self.class_num += class_per_num
        for i in range(self.out_dim):
            self.classes[i * class_per_num: (i + 1) * class_per_num] = i
            pass
        np.random.shuffle(self.classes)
        pass

    def reset(self):
        self.count = 0
        self.count_2 = 0
        self.class_num *= 0
        pass

    def cal_label(self, out, indexes):
        out_data = out.data.cpu()
        top_k = out_data.topk(self.out_dim, dim=1)[1].cpu()
        indexes_cpu = indexes.cpu()

        self.features[indexes_cpu] = out_data

        batch_size = top_k.size(0)
        class_labels = np.zeros(shape=(batch_size,), dtype=np.int)

        for i in range(batch_size):
            for j_index, j in enumerate(top_k[i]):
                if self.class_per_num > self.class_num[j]:
                    class_labels[i] = j
                    self.class_num[j] += 1
                    self.count += 1 if self.classes[indexes_cpu[i]] != j else 0
                    self.classes[indexes_cpu[i]] = j
                    self.count_2 += 1 if j_index != 0 else 0
                    break
                pass
            pass
        pass

    def get_label(self, indexes):
        _device = indexes.device
        return torch.tensor(self.classes[indexes.cpu()]).long().to(_device)

    pass


##############################################################################################################


class Runner(object):

    def __init__(self):
        self.best_accuracy = 0.0
        self.adjust_learning_rate = Config.adjust_learning_rate

        # all data
        self.data_train = CIFARFSDataset.get_data_all(Config.data_root)
        self.task_train = CIFARFSDataset(self.data_train, Config.num_way, Config.num_shot)
        self.task_train_loader = DataLoader(self.task_train, Config.batch_size, True, num_workers=Config.num_workers)

        # IC
        self.produce_class = ProduceClass(len(self.data_train), Config.ic_out_dim, Config.ic_ratio)
        self.produce_class.init()
        self.task_train.set_samples_class(self.produce_class.classes)
        self.task_train.set_samples_feature(self.produce_class.features)

        # model
        self.matching_net = RunnerTool.to_cuda(Config.matching_net)
        self.ic_model = RunnerTool.to_cuda(ICResNet(low_dim=Config.ic_out_dim,input_dim=Config.input_dim,
                                                    resnet=Config.resnet, modify_head=Config.modify_head))
        if Config.DataParallel:
            self.matching_net  = torch.nn.DataParallel(self.matching_net , device_ids=range(torch.cuda.device_count()))
            Config.logger.info(f'torch.cuda.device_count()  {range(torch.cuda.device_count())}')
            self.ic_model = torch.nn.DataParallel(self.ic_model, device_ids=range(torch.cuda.device_count()))
        self.norm = Normalize(2)

        RunnerTool.to_cuda(self.matching_net.apply(RunnerTool.weights_init))
        RunnerTool.to_cuda(self.ic_model.apply(RunnerTool.weights_init))

        # optim
        self.matching_net_optim = torch.optim.SGD(
            self.matching_net.parameters(), lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)
        self.ic_model_optim = torch.optim.SGD(
            self.ic_model.parameters(), lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)

        # loss
        self.ic_loss = RunnerTool.to_cuda(nn.CrossEntropyLoss())
        self.fsl_loss = RunnerTool.to_cuda(nn.MSELoss())

        # Eval
        self.test_tool_fsl = TestTool(self.matching_test, data_root=Config.data_root,
                                      num_way=Config.num_way, num_shot=Config.num_shot,
                                      episode_size=Config.episode_size, test_episode=Config.test_episode,
                                      transform=self.task_train.transform_test,Config=Config)
        self.test_tool_ic = ICTestTool(feature_encoder=None, ic_model=self.ic_model,
                                       data_root=Config.data_root, batch_size=Config.batch_size,
                                       num_workers=Config.num_workers, ic_out_dim=Config.ic_out_dim,
                                       transform=self.task_train.transform_test,Config=Config)
        pass

    def load_model(self):
        mn_pkl_list=glob.glob(f'{os.path.dirname(Config.mn_dir)}/*mn*pkl')
        print("*"*60,mn_pkl_list)
        if len(mn_pkl_list)!=1:
            Config.logger.info('your model format is new fsl-*pkl')
            mn_pkl_list=glob.glob(f'{os.path.dirname(Config.mn_dir)}/fsl-*pkl')

        assert len(mn_pkl_list)==1 and os.path.exists(mn_pkl_list[0]),'len(pkl_list) must is 1 and pkl_list[0] must exist'
        Config.logger.info("load matching net success from {}".format(mn_pkl_list[0]))
        self.matching_net.load_state_dict(torch.load(mn_pkl_list[0]))


        ic_pkl_list=glob.glob(f'{os.path.dirname(Config.ic_dir)}/*ic*pkl')
        if len(ic_pkl_list)!=1:
            Config.logger.info('your model format is new ic-*pkl')
            ic_pkl_list=glob.glob(f'{os.path.dirname(Config.mn_dir)}/ic-*pkl')
        assert len(ic_pkl_list)==1 and os.path.exists(ic_pkl_list[0]),'len(pkl_list) must is 1 and pkl_list[0] must exist'
        Config.logger.info("load ic_model success from {}".format(ic_pkl_list[0]))
        self.ic_model.load_state_dict(torch.load(ic_pkl_list[0]))


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

    def train(self):
        Config.logger.info("Training...")

        # Init Update
        try:
            self.matching_net.eval()
            self.ic_model.eval()
            Config.logger.info("Init label {} .......")
            self.produce_class.reset()
            for task_data, task_labels, task_index, task_ok in tqdm(self.task_train_loader):
                ic_labels = RunnerTool.to_cuda(task_index[:, -1])
                task_data, task_labels = RunnerTool.to_cuda(task_data), RunnerTool.to_cuda(task_labels)
                ic_out_logits, ic_out_l2norm = self.ic_model(task_data[:, -1])
                self.produce_class.cal_label(ic_out_l2norm, ic_labels)
                pass
            Config.logger.info("Train [0]: {}/{}".format(self.produce_class.count, self.produce_class.count_2))
        finally:
            pass

        for epoch in range(1, 1 + Config.train_epoch):
            self.matching_net.train()
            self.ic_model.train()
            mn_lr= self.adjust_learning_rate(self.matching_net_optim, epoch,
                                             Config.first_epoch, Config.t_epoch, Config.learning_rate)
            ic_lr = self.adjust_learning_rate(self.ic_model_optim, epoch,
                                              Config.first_epoch, Config.t_epoch, Config.learning_rate)
            Config.logger.info('Epoch:[{}] mn_lr={} ic_lr={}'.format(epoch, mn_lr, ic_lr))
            Config.writer.add_scalars('lr', {'mn_lr': mn_lr,'ic_lr': ic_lr}, epoch)

            self.produce_class.reset()
            # Config.logger.info(self.task_train.classes)
            is_ok_total, is_ok_acc = 0, 0
            all_loss, all_loss_fsl, all_loss_ic = 0.0, 0.0, 0.0
            for task_data, task_labels, task_index, task_ok in tqdm(self.task_train_loader):
                ic_labels = RunnerTool.to_cuda(task_index[:, -1])
                task_data, task_labels = RunnerTool.to_cuda(task_data), RunnerTool.to_cuda(task_labels)

                ###########################################################################
                # 1 calculate features
                relations = self.matching(task_data)
                ic_out_logits, ic_out_l2norm = self.ic_model(task_data[:, -1])

                # 2
                ic_targets = self.produce_class.get_label(ic_labels)
                self.produce_class.cal_label(ic_out_l2norm, ic_labels)

                # 3 loss
                loss_fsl = self.fsl_loss(relations, task_labels)
                loss_ic = self.ic_loss(ic_out_logits, ic_targets)
                loss = loss_fsl * Config.loss_fsl_ratio + loss_ic * Config.loss_ic_ratio
                all_loss += loss.item()
                all_loss_fsl += loss_fsl.item()
                all_loss_ic += loss_ic.item()

                # 4 backward
                self.ic_model.zero_grad()
                loss_ic.backward()
                self.ic_model_optim.step()

                self.matching_net.zero_grad()
                loss_fsl.backward()
                self.matching_net_optim.step()

                # is ok
                is_ok_acc += torch.sum(torch.cat(task_ok))
                is_ok_total += torch.prod(torch.tensor(torch.cat(task_ok).shape))
                ###########################################################################
                pass

            ###########################################################################
            Config.logger.info("Epoch:[{}] loss:{:.3f} fsl:{:.3f} ic:{:.3f} ok:{:.3f}({}/{})".format(epoch, all_loss / len(self.task_train_loader),
                all_loss_fsl / len(self.task_train_loader), all_loss_ic / len(self.task_train_loader),
                int(is_ok_acc) / int(is_ok_total), is_ok_acc, is_ok_total, ))
            Config.writer.add_scalars('loss', {'loss_all': all_loss / len(self.task_train_loader),
                            'loss_fsl':  all_loss_fsl / len(self.task_train_loader),
                            'loss_ic':  all_loss_ic / len(self.task_train_loader),}, epoch)
            Config.writer.add_scalar('loss_fsl', all_loss_fsl / len(self.task_train_loader), epoch)
            Config.writer.add_scalar('ok_acc', int(is_ok_acc) / int(is_ok_total), epoch)


            Config.logger.info("Train: [{}] {}/{}".format(epoch, self.produce_class.count, self.produce_class.count_2))
            Config.writer.add_scalars('produce_class', {'produce_class.count': self.produce_class.count,'produce_class.count_2': self.produce_class.count_2}, epoch)

            # Val
            if epoch % Config.val_freq == 0:
                self.matching_net.eval()
                self.ic_model.eval()

                self.test_tool_ic.val(epoch=epoch, is_print=True)
                val_accuracy = self.test_tool_fsl.val(episode=epoch, is_print=True)


                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy  
                    for i in glob.glob(os.path.dirname(Config.mn_dir)+'/*pkl'):
                        Config.logger.info("delete {}".format(i))
                        os.remove(i)
                    torch.save(self.matching_net.state_dict(), os.path.dirname(Config.mn_dir)+'/'+'fsl-'+str(epoch)+os.path.basename(Config.mn_dir))
                    torch.save(self.ic_model.state_dict(), os.path.dirname(Config.ic_dir)+'/'+'ic-'+str(epoch)+os.path.basename(Config.ic_dir))
                    Config.logger.info("Save networks for epoch: {}".format(epoch))
                    Config.logger.info("Save {}".format(os.path.dirname(Config.mn_dir)+'/'+'fsl-'+str(epoch)+os.path.basename(Config.mn_dir)))
                    Config.logger.info("Save {}".format(os.path.dirname(Config.ic_dir)+'/'+'ic-'+str(epoch)+os.path.basename(Config.ic_dir)))
                    pass
                pass
            pass
            ###########################################################################
        pass

    pass




def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--drop_rate', '-dr',type=float, default=0.2, help=' drop_rate')
    parser.add_argument('--gpu', '-g',type=str, default='0', help=' gpu')
    parser.add_argument('--batch_size', '-bs',type=int, default=64, help=' batchsize')
    parser.add_argument('--train_epoch', '-te',type=int, default=2100, help='train_epoch')
    parser.add_argument('--debug','-d',  action='store_true', help=' debug')
    parser.add_argument('--fsl_backbone', '-fb',default='c4', help='fsl_backbone is c4')
    parser.add_argument('--num_way', '-w',type=int, default=5, help=' num_way=5')
    parser.add_argument('--num_shot', '-s',type=int, default=1, help=' num_shot=1')
    parser.add_argument('--val', '-v',type=str, default='',help=' only val wegit _dir')
    parser.add_argument('--lr',type=int, default=2,help=' lr function id')
    parser.add_argument('--convert',type=str, default='RGB',help=' Image.open(x).convert(RGB)')      
      
    args = parser.parse_args()
    return args

class Config(object):
    ##################### ic
    ic_out_dim = 512
    ic_ratio = 1
    loss_fsl_ratio = 1.0
    loss_ic_ratio = 1.0
    # resnet = resnet18
    resnet = resnet34#ic11111
    modify_head = True
    # modify_head = False
    ##################### ic
    args = parse_args()
    convert=args.convert
    input_dim=len(convert)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    batch_size = args.batch_size
    train_epoch = args.train_epoch
    num_workers = 8
    learning_rate = 0.01
    num_way = 5
    num_shot = 1
    
    val_freq=1 if args.debug else 10
    episode_size = 15#MiniImageNetTask
    test_episode = 600 #600代
    first_epoch, t_epoch = 500,200
    adjust_learning_rate = RunnerTool.adjust_learning_rate1
    hid_dim = 64
    z_dim = 64
    is_png = True
    # is_png = False
    ###############################################################################################
    drop_rate=args.drop_rate
    dataset='CIFARFS'
    image_size=32
    if  args.fsl_backbone=='res12':
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
    DataParallel=True if len(args.gpu)>=2 else False
    ###############################################################################################
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')# or 
    head="_head" if modify_head else ""
    model_name =f'EP{train_epoch}_BS{batch_size}_ft{first_epoch}_{t_epoch}_mn_{commit}'

    data_root = '/home/ubuntu/Dataset/Partition1/hzh/data/CIFARFS'
    if not os.path.exists(data_root):
        data_root = '/home/test/Documents/hzh/ActiveLearning/data/CIFARFS'

    _root_path = "../models_mn/two_ic_ufsl_2net_res_sgd_acc_duli"
    # _root_path = "../models_rn/two_ic_ufsl_2net_res_sgd_acc_duli"
################################################################################################down is same
    if not args.debug:
        debug=""
        for i in glob.glob(_root_path+'/debug*'):
            shutil.rmtree(i)
            Tools.print(f'delete {i}')
    else:
            debug="debug"
            Tools.print(f'INFO - you are debugging ')
    date_dir=f'{_root_path}/{debug+current_time}_{model_name}'
    mn_dir=Tools.new_dir(f"{date_dir}/{model_name}.pkl") if not args.val else args.val
    ic_dir = Tools.new_dir(f"{date_dir}/{model_name}_{head}.pkl")
    writer = SummaryWriter(date_dir+'/runs')

    shutil.copy(os.path.abspath(sys.argv[0]),date_dir)

    log_file = os.path.join(date_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file,name=f"UFSL-{commit}")
    for name, val in collect_env().items():
        logger.info(f'{name}: {val}')
    logger.info(model_name)
    logger.info(f'DataParallel is {DataParallel}')
    logger.info(f"platform.platform{platform.platform()}")
    logger.info(f"config:   ")
    logger.info(f"args.gpu :{args.gpu}  is_png:   {is_png}, test_episode:   {test_episode}") 
    logger.info(f"first_epoch:   {first_epoch},t_epoch:   {t_epoch}, val_freq:   {val_freq},episode_size:   {episode_size}")
    logger.info(f'hid_dim:   {hid_dim},z_dim:   {z_dim} , is_png:   {is_png}')
    ABSPATH=os.path.abspath(sys.argv[0])

    pass


if __name__ == '__main__':
    # init the logger before other steps
    args = parse_args()
    runner = Runner()
    if not args.val:
        runner.train()
    runner.load_model()
    runner.matching_net.eval()
    runner.ic_model.eval()
    runner.test_tool_ic.val(epoch=Config.train_epoch, is_print=True)
    runner.test_tool_fsl.val(episode=Config.train_epoch, is_print=True)
    runner.test_tool_fsl.test(test_avg_num=5, episode=Config.train_epoch, is_print=True)


##############################################################################################################


"""
ALISURE
is_png = False, resnet = resnet18, modify_head = False
0_2100_64_5_1_500_200_512_1_1.0_1.0_mn.pkl
2020-12-10 05:17:18   2101 loss:1.092 fsl:0.069 ic:1.023 ok:0.258(9911/38400)
2020-12-10 05:17:18 Train: [2100] 8957/1725
2020-12-10 05:19:20 load matching net success from ../models_mn/two_ic_ufsl_2net_res_sgd_acc_duli/0_2100_64_5_1_500_200_512_1_1.0_1.0_mn.pkl
2020-12-10 05:19:20 load ic model success from ../models_mn/two_ic_ufsl_2net_res_sgd_acc_duli/0_2100_64_5_1_500_200_512_1_1.0_1.0_ic.pkl
2020-12-10 05:19:20 Test 2100 .......
2020-12-10 05:19:34 Epoch: 2100 Train 0.4955/0.7817 0.0000
2020-12-10 05:19:34 Epoch: 2100 Val   0.5825/0.9148 0.0000
2020-12-10 05:19:34 Epoch: 2100 Test  0.5555/0.9013 0.0000
2020-12-10 05:21:17 Train 2100 Accuracy: 0.4896666666666667
2020-12-10 05:21:17 Val   2100 Accuracy: 0.45155555555555554
2020-12-10 05:25:17 episode=2100, Mean Test accuracy=0.45596


Harzva-----fsl--4conv 
2020-12-18 21:15:16 Test 1690 .......
2020-12-18 21:16:10 Epoch: 1690 Train 0.4051/0.7067 0.0000
2020-12-18 21:16:10 Epoch: 1690 Val   0.4941/0.8628 0.0000
2020-12-18 21:16:10 Epoch: 1690 Test  0.5096/0.8661 0.0000
2020-12-18 21:17:14 Train 1690 Accuracy: 0.5996666666666668
2020-12-18 21:17:14 Val   1690 Accuracy: 0.47355555555555556
2020-12-18 21:17:14 Test1 1690 Accuracy: 0.5275555555555556
2020-12-18 21:17:14 Test2 1690 Accuracy: 0.5244

  
2020-12-19 04:04:35   2101 loss:1.219 fsl:0.063 ic:1.156 ok:0.354(13591/38400)
2020-12-19 04:04:35 Train: [2100] 10219/2096
2020-12-19 04:06:32 load matching net success from ../models_mn/two_ic_ufsl_2net_res_sgd_acc_duli/4_2100_64_5_1_500_200_512_1_1.0_1.0_head_png_mn_5way_1shot_CIFARFS.pkl
2020-12-19 04:06:32 load ic model success from ../models_mn/two_ic_ufsl_2net_res_sgd_acc_duli/4_2100_64_5_1_500_200_512_1_1.0_1.0_head_png_ic_CIFARFS.pkl
2020-12-19 04:06:32 Test 2100 .......
2020-12-19 04:07:25 Epoch: 2100 Train 0.3975/0.7035 0.0000
2020-12-19 04:07:25 Epoch: 2100 Val   0.4931/0.8650 0.0000
2020-12-19 04:07:25 Epoch: 2100 Test  0.4997/0.8682 0.0000
2020-12-19 04:08:27 Train 2100 Accuracy: 0.5851111111111111
2020-12-19 04:08:27 Val   2100 Accuracy: 0.48911111111111116
2020-12-19 04:08:27 Test1 2100 Accuracy: 0.5263333333333333
2020-12-19 04:08:27 Test2 2100 Accuracy: 0.5345333333333333
2020-12-19 04:10:54 episode=2100, Test accuracy=0.5232
2020-12-19 04:10:54 episode=2100, Test accuracy=0.5251555555555556
2020-12-19 04:10:54 episode=2100, Test accuracy=0.5291777777777777
2020-12-19 04:10:54 episode=2100, Test accuracy=0.5346
2020-12-19 04:10:54 episode=2100, Test accuracy=0.5344888888888889
2020-12-19 04:10:54 episode=2100, Mean Test accuracy=0.5293244444444445

Harzva-----Res12
2020-12-24 07:06:54,161 - UFSL - INFO - Test 2100 .......
2020-12-24 07:07:49,771 - UFSL - INFO - Epoch: 2100 KNN_train_acc 0.4312/0.7211 0.0000
2020-12-24 07:07:49,771 - UFSL - INFO - Epoch: 2100 KNN_val_acc   0.4964/0.8696 0.0000
2020-12-24 07:07:49,771 - UFSL - INFO - Epoch: 2100 KNN_test_acc  0.5110/0.8652 0.0000
2020-12-24 07:10:55,680 - UFSL - INFO - fsl_Train 2100 Accuracy: 0.6548888888888889
2020-12-24 07:10:55,680 - UFSL - INFO - fsl_Val   2100 Accuracy: 0.5012222222222222
2020-12-24 07:10:55,681 - UFSL - INFO - fsl_Test1 2100 Accuracy: 0.5547777777777778
2020-12-24 07:10:55,681 - UFSL - INFO - fsl_Test2 2100 Accuracy: 0.5584222222222223

"""