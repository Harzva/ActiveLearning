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
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from mn_miniimagenet_fsl_test_tool import TestTool
from mn_miniimagenet_tool import MatchingNet, RunnerTool
import pysnooper
import pickle
##############################################################################################################


class Config(object):
    gpu_id = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # train_epoch = 300
    train_epoch = 180
    learning_rate = 0.001
    num_workers = 8

    val_freq = 10

    num_way = 5
    num_shot = 1
    batch_size = 64

    episode_size = 15
    test_episode = 600

    hid_dim = 64
    z_dim = 64

    # loss_is_mse = False
    loss_is_mse = True

    is_png = True
    # is_png = False

    matching_net = MatchingNet(hid_dim=hid_dim, z_dim=z_dim)

    model_name = "{}_{}_{}_{}_{}{}".format(train_epoch, batch_size, hid_dim, z_dim,
                                           "mse" if loss_is_mse else "ce", "_png" if is_png else "")
    Tools.print(model_name)
    print("platform.platform()",platform.platform())



    if "Linux" in platform.platform():
        # data_root ='/mnt/4T/Data/data/miniImagenet'
        data_root = '/home/ubuntu/Documents/hzh/ActiveLearning-master/data/cifar100'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
    else:
        data_root = "F:\\data\\miniImagenet"
    print("data_root",data_root)
    data_root = os.path.join(data_root, "data") if is_png else data_root
    # data_root = os.path.join(data_root, "miniImageNet_png") if is_png else data_root
    Tools.print(data_root)

    mn_dir = Tools.new_dir("../models_mn/fsl/{}_mn_{}way_{}shot.pkl".format(model_name, num_way, num_shot))
    pass

_CIFAR_DATASET_DIR = "/home/ubuntu/Documents/hzh/ActiveLearning-master/data"
_CIFAR_CATEGORY_SPLITS_DIR ="/home/ubuntu/Documents/hzh/ActiveLearning-master/data/cifar-100-python/cifar-fs_splits"
# "/home/ubuntu/Documents/hzh/ActiveLearning-master/data/cifar100/splits/bertinetto"
_CIFAR_MEAN_PIXEL = [x / 255.0 for x in [125.3, 123.0, 113.9]]
_CIFAR_STD_PIXEL = [x / 255.0 for x in [63.0, 62.1, 66.7]]

class CifarfsDataset(object):

    def __init__(self, data_list, num_way, num_shot):#故此函数被声明为私有方法，不可类外调用。
        self.data_list, self.num_way, self.num_shot = data_list, num_way, num_shot

        self.data_dict = {}
        for index, label, image_filename in self.data_list:
            if label not in self.data_dict:
                self.data_dict[label] = []
            self.data_dict[label].append((index, label, image_filename))
            #(0, 0, '/mnt/4T/Data/data/miniImagenet/miniImageNet_png/train/n03838899/38081.png')
            # print("data_dict",self.data_dict[label])
            pass

        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std= [x / 255.0 for x in [63.0, 62.1, 66.7]])
        self.transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.transform_test = transforms.Compose(
                [lambda x: np.asarray(x), transforms.ToTensor(), normalize]
        )
        pass

    def __len__(self):
        return len(self.data_list)
    
    @staticmethod
    # @pysnooper.snoop("/home/ubuntu/Documents/hzh/ActiveLearning-master/UFSLviaIC/MN/log/debug.log", prefix="--*--")
    def get_data_all(data_root):
        def read_categories(filename):
            with open(filename) as f:
                categories = f.readlines()
            categories = [x.strip() for x in categories]
            return categories

        def get_label_ids(category_names):
            label_ids = [all_category_names.index(cname) for cname in category_names]
            return label_ids

        train_folder = os.path.join(data_root)#'/home/ubuntu/Documents/hzh/ActiveLearning-master/data/cifar100/data/train'

        cifar100_metadata_path = os.path.join(_CIFAR_DATASET_DIR, "cifar-100-python", "meta")
        all_category_names = pickle.load(open(cifar100_metadata_path, "rb"))[
            "fine_label_names"
        ]
        print("all_category_names--fine_label_names",all_category_names)
        print("all_category_names--fine_label_names--len",len(all_category_names))#100
        train_category_names =read_categories(
            os.path.join(_CIFAR_CATEGORY_SPLITS_DIR, "train.txt")
        )
        val_category_names =read_categories(
            os.path.join(_CIFAR_CATEGORY_SPLITS_DIR, "val.txt")
        )
        test_category_names = read_categories(
            os.path.join(_CIFAR_CATEGORY_SPLITS_DIR, "test.txt")
        )

        train_category_ids = get_label_ids(train_category_names)
        val_category_ids = get_label_ids(val_category_names)#[55, 48, 87, 40, 27, 73, 14, 4, 7, 89, 32, 47, 15, 26, 71, 19]
        test_category_ids = get_label_ids(test_category_names)#[2, 5, 8, 21, 34, 42, 46, 58, 60, 62, 69, 70, 77, 83, 84, 86, 94, 95, 98, 99]

        count_image, count_class, data_train_list = 0, 0, []
        for label in train_category_names:
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
        task_data = torch.cat([torch.unsqueeze(self.read_image(one[2], self.transform), dim=0) for one in task_list])
        task_label = torch.Tensor([int(one_tuple[1] == now_label) for one_tuple in c_way_k_shot_tuple_list])
        task_index = torch.Tensor([one[0] for one in task_list]).long()
        # print("task_data, task_label, task_index")
        # print("task_data, task_label, task_index",task_data, task_label, task_index)

        return task_data, task_label, task_index

    @staticmethod
    def read_image(image_path, transform=None):
        image = Image.open(image_path).convert(Config.convert)
        if transform is not None:
            image = transform(image)
        return image

    pass


##############################################################################################################


class Runner(object):

    def __init__(self):
        self.best_accuracy = 0.0

        # all data
        self.data_train = CifarfsDataset.get_data_all(Config.data_root)
        self.task_train = CifarfsDataset(self.data_train, Config.num_way, Config.num_shot)
        self.task_train_loader = DataLoader(self.task_train, Config.batch_size, shuffle=True, num_workers=Config.num_workers)

        # model
        self.matching_net = RunnerTool.to_cuda(Config.matching_net)
        RunnerTool.to_cuda(self.matching_net.apply(RunnerTool.weights_init))

        # loss
        self.loss = RunnerTool.to_cuda(nn.MSELoss())
        self.loss_ce = RunnerTool.to_cuda(nn.CrossEntropyLoss())

        # optim
        self.matching_net_optim = torch.optim.Adam(self.matching_net.parameters(), lr=Config.learning_rate)
        self.matching_net_scheduler = StepLR(self.matching_net_optim, Config.train_epoch // 3, gamma=0.5)

        self.test_tool = TestTool(self.matching_test, data_root=Config.data_root,
                                  num_way=Config.num_way,  num_shot=Config.num_shot,
                                  episode_size=Config.episode_size, test_episode=Config.test_episode,
                                  transform=self.task_train.transform_test)
        pass

    def load_model(self):
        if os.path.exists(Config.mn_dir):
            self.matching_net.load_state_dict(torch.load(Config.mn_dir))
            Tools.print("load proto net success from {}".format(Config.mn_dir))
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
        support_magnitude = torch.sum(torch.pow(z_support, 2), 2).clamp(1e-10, float("inf")).rsqrt()
        similarities = support_magnitude * torch.sum(z_support * z_query_expand, 2)
        similarities_softmax = torch.softmax(similarities, dim=1)
        similarities_softmax = similarities_softmax.view(z_batch_size, Config.num_way, Config.num_shot)
        predicts = torch.sum(similarities_softmax, dim=2)
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
        z_support_magnitude = torch.sum(torch.pow(z_support_expand, 2), 2).clamp(1e-10, float("inf")).rsqrt()
        similarities = z_support_magnitude * torch.sum(z_support_expand * z_query_expand, 2)
        similarities_softmax = torch.softmax(similarities, dim=1)
        similarities_softmax = similarities_softmax.view(batch_num, Config.num_way, Config.num_shot)
        predicts = torch.sum(similarities_softmax, dim=2)
        return predicts

    def train(self):
        Tools.print()
        Tools.print("Training...")

        for epoch in range(Config.train_epoch):
            self.matching_net.train()

            Tools.print()
            all_loss = 0.0
            for task_data, task_labels, task_index in tqdm(self.task_train_loader):
                task_data, task_labels = RunnerTool.to_cuda(task_data), RunnerTool.to_cuda(task_labels)

                # 1 calculate features
                predicts = self.matching(task_data)

                # 2 loss
                if Config.loss_is_mse:
                    loss = self.loss(predicts, task_labels)
                else:
                    targets = torch.argmax(task_labels, dim=1) // Config.num_shot
                    loss = self.loss_ce(predicts, targets)
                    pass

                all_loss += loss.item()

                # 3 backward
                self.matching_net.zero_grad()
                loss.backward()
                self.matching_net_optim.step()
                ###########################################################################
                pass

            ###########################################################################
            # print
            Tools.print("{:6} loss:{:.3f} lr:{}".format(
                epoch + 1, all_loss / len(self.task_train_loader), self.matching_net_scheduler.get_last_lr()))

            self.matching_net_scheduler.step()
            ###########################################################################

            ###########################################################################
            # Val
            if epoch % Config.val_freq == 0:
                Tools.print()
                Tools.print("Test {} {} .......".format(epoch, Config.model_name))
                self.matching_net.eval()

                val_accuracy = self.test_tool.val(episode=epoch, is_print=True)
                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    torch.save(self.matching_net.state_dict(), Config.mn_dir)
                    Tools.print("Save networks for epoch: {}".format(epoch))
                    pass
                pass
            ###########################################################################
            pass

        pass

    pass




##############################################################################################################


"""
180_64_64_64_mse_mn_5way_1shot
2020-12-06 20:16:16 load proto net success from ../models_mn/fsl/180_64_64_64_mse_mn_5way_1shot.pkl
2020-12-06 20:17:57 Train 180 Accuracy: 0.7238888888888888
2020-12-06 20:17:57 Val   180 Accuracy: 0.5101111111111111
2020-12-06 20:21:32 episode=180, Mean Test accuracy=0.5105999999999999

180_64_64_64_ce_mn_5way_1shot
2020-12-06 20:11:16 load proto net success from ../models_mn/fsl/180_64_64_64_ce_mn_5way_1shot.pkl
2020-12-06 20:12:58 Train 180 Accuracy: 0.6807777777777777
2020-12-06 20:12:58 Val   180 Accuracy: 0.503
2020-12-06 20:17:20 episode=180, Mean Test accuracy=0.5060533333333334

180_64_64_64_png_mse_mn_5way_1shot
2020-12-07 04:10:24 load proto net success from ../models_mn/fsl/180_64_64_64__pngmse_mn_5way_1shot.pkl
2020-12-07 04:12:29 Train 180 Accuracy: 0.7603333333333333
2020-12-07 04:12:29 Val   180 Accuracy: 0.5257777777777778
2020-12-07 04:17:42 episode=180, Mean Test accuracy=0.5266533333333333



2020-12-13 23:31:13 load proto net success from ../models_mn/fsl/180_64_64_64_mse_png_mn_5way_1shot.pkl
2020-12-13 23:32:53 Train 180 Accuracy: 0.7493333333333334
2020-12-13 23:32:53 Val   180 Accuracy: 0.5289999999999999
2020-12-13 23:32:53 Test1 180 Accuracy: 0.5251111111111111
2020-12-13 23:32:53 Test2 180 Accuracy: 0.5231111111111111
2020-12-13 23:37:07 episode=180, Test accuracy=0.5283555555555556
2020-12-13 23:37:07 episode=180, Test accuracy=0.5249111111111112
2020-12-13 23:37:07 episode=180, Test accuracy=0.5183111111111112
2020-12-13 23:37:07 episode=180, Test accuracy=0.5208444444444444
2020-12-13 23:37:07 episode=180, Test accuracy=0.5205777777777777
2020-12-13 23:37:07 episode=180, Mean Test accuracy=0.5226

"""


if __name__ == '__main__':
    runner = Runner()
    # runner.load_model()

    # runner.matching_net.eval()
    # runner.test_tool.val(episode=0, is_print=True)

    runner.train()

    runner.load_model()
    runner.matching_net.eval()# RunnerTool
    runner.test_tool.val(episode=Config.train_epoch, is_print=True)
    runner.test_tool.test(test_avg_num=5, episode=Config.train_epoch, is_print=True)
    pass