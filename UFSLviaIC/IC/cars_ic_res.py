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
from tool_ic_test import ICTestTool
from alisuretool.Tools import Tools
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, resnet34, resnet50, vgg16_bn


##############################################################################################################


class DatasetIC(Dataset):

    def __init__(self, data_list, image_size=84):
        self.data_list = data_list
        self.train_label = [one[1] for one in self.data_list]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225])

        # self.transform = transforms.Compose([
        #     transforms.Resize([int(image_size * 1.25), int(image_size * 1.25)]),
        #     transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.)),
        #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.4), transforms.RandomGrayscale(0.2),
        #     transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        # self.transform_test = transforms.Compose([transforms.Resize([int(image_size * 1.25), int(image_size * 1.25)]),
        #                                           transforms.CenterCrop(image_size), transforms.ToTensor(), normalize])
        self.transform = transforms.Compose([
            transforms.Resize([int(image_size * 1.50), int(image_size * 1.50)]),
            transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4), transforms.RandomGrayscale(0.2),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        self.transform_test = transforms.Compose([transforms.Resize([int(image_size * 1.50), int(image_size * 1.50)]),
                                                  transforms.CenterCrop(image_size), transforms.ToTensor(), normalize])
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        idx, label, image_filename = self.data_list[idx]
        image = Image.open(image_filename).convert('RGB')
        image = self.transform(image)
        return image, label, idx

    @staticmethod
    def get_data_all(data_root):
        train_folder = os.path.join(data_root, "train")

        count_image, count_class, data_train_list = 0, 0, []
        image_list = os.listdir(train_folder)
        for label in image_list:
            now_class_path = os.path.join(train_folder, label)
            if os.path.isdir(now_class_path):
                for time in range(Config.ic_times):
                    for name in os.listdir(now_class_path):
                        data_train_list.append((count_class, os.path.join(now_class_path, name)))
                        pass
                    pass
                count_class += 1
            pass

        np.random.shuffle(data_train_list)
        data_train_list_final = []
        for index, data_train in enumerate(data_train_list):
            data_train_list_final.append((index, data_train[0], data_train[1]))
            pass

        return data_train_list_final

    pass


##############################################################################################################


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
        pass

    def forward(self, x, dim=1):
        norm = x.pow(self.power).sum(dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class ICResNet(nn.Module):

    def __init__(self, low_dim=512, modify_head=False, resnet=None, vggnet=None):
        super().__init__()
        self.is_res = True if resnet else False
        self.is_vgg = True if vggnet else False

        if self.is_res:
            self.resnet = resnet(num_classes=low_dim)
            if modify_head:
                self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                pass
        elif self.is_vgg:
            self.vggnet = vggnet()
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512, low_dim)
            pass
        else:
            raise Exception("......")

        self.l2norm = Normalize(2)
        pass

    def forward(self, x):
        if self.is_res:
            out_logits = self.resnet(x)
        elif self.is_vgg:
            features = self.vggnet.features(x)
            features = self.avgpool(features)
            features = torch.flatten(features, 1)
            out_logits = self.fc(features)
            pass
        else:
            raise Exception("......")

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
        self.class_num = np.zeros(shape=(self.out_dim, ), dtype=np.int)
        self.classes = np.zeros(shape=(self.n_sample, ), dtype=np.int)
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
        top_k = out.data.topk(self.out_dim, dim=1)[1].cpu()
        indexes_cpu = indexes.cpu()

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

        # data
        self.data_train = DatasetIC.get_data_all(Config.data_root)
        self.ic_train_loader = DataLoader(DatasetIC(self.data_train), Config.batch_size,
                                          shuffle=True, num_workers=Config.num_workers)

        # IC
        self.produce_class = ProduceClass(len(self.data_train), Config.ic_out_dim, Config.ic_ratio)
        self.produce_class.init()

        # model
        self.ic_model = self.to_cuda(ICResNet(Config.ic_out_dim, modify_head=Config.modify_head,
                                              resnet=Config.resnet, vggnet=Config.vggnet))
        self.ic_loss = self.to_cuda(nn.CrossEntropyLoss())

        self.ic_model_optim = torch.optim.SGD(
            self.ic_model.parameters(), lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)

        # Eval
        self.test_tool_ic = ICTestTool(feature_encoder=None, ic_model=self.ic_model,
                                       data_root=Config.data_root, batch_size=Config.batch_size,
                                       num_workers=Config.num_workers, ic_out_dim=Config.ic_out_dim,
                                       transform=self.ic_train_loader.dataset.transform_test, k=Config.knn)
        pass

    @staticmethod
    def to_cuda(x):
        return x.cuda() if torch.cuda.is_available() else x

    @staticmethod
    def _weights_init(m):
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif class_name.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif class_name.find('Linear') != -1:
            m.weight.data.normal_(0, 0.01)
            # m.bias.data = torch.ones(m.bias.data.size())
            pass
        pass

    @staticmethod
    def adjust_learning_rate1(optimizer, epoch):

        def _get_lr(_base_lr, now_epoch, _t_epoch=Config.t_epoch, _eta_min=1e-05):
            return _eta_min + (_base_lr - _eta_min) * (1 + math.cos(math.pi * now_epoch / _t_epoch)) / 2

        t_epoch = Config.t_epoch
        first_epoch = Config.first_epoch
        init_learning_rate = Config.learning_rate
        if epoch < first_epoch + t_epoch * 0:  # 0-200
            learning_rate = init_learning_rate
        elif epoch < first_epoch + t_epoch * 1:  # 200-300
            learning_rate = init_learning_rate / 2
        elif epoch < first_epoch + t_epoch * 2:  # 300-400
            learning_rate = init_learning_rate / 4
        elif epoch < first_epoch + t_epoch * 3:  # 400-500
            learning_rate = _get_lr(init_learning_rate / 2.0, epoch - first_epoch - t_epoch * 2)
        elif epoch < first_epoch + t_epoch * 4:  # 500-600
            learning_rate = _get_lr(init_learning_rate / 4.0, epoch - first_epoch - t_epoch * 3)
        elif epoch < first_epoch + t_epoch * 5:  # 600-700
            learning_rate = _get_lr(init_learning_rate / 8.0, epoch - first_epoch - t_epoch * 4)
        elif epoch < first_epoch + t_epoch * 6:  # 700-800
            learning_rate = _get_lr(init_learning_rate / 16., epoch - first_epoch - t_epoch * 5)
        elif epoch < first_epoch + t_epoch * 7:  # 800-900
            learning_rate = _get_lr(init_learning_rate / 32., epoch - first_epoch - t_epoch * 6)
        else:  # 900-1000
            learning_rate = _get_lr(init_learning_rate / 64., epoch - first_epoch - t_epoch * 7)
            pass

        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
            pass

        return learning_rate

    @staticmethod
    def adjust_learning_rate2(optimizer, epoch):

        def _get_lr(_base_lr, now_epoch, _t_epoch=Config.t_epoch, _eta_min=1e-05):
            return _eta_min + (_base_lr - _eta_min) * (1 + math.cos(math.pi * now_epoch / _t_epoch)) / 2

        t_epoch = Config.t_epoch
        first_epoch = Config.first_epoch
        init_learning_rate = Config.learning_rate

        if epoch < first_epoch + t_epoch * 0:  # 0-500
            learning_rate = init_learning_rate
        elif epoch < first_epoch + t_epoch * 1:  # 500-1000
            learning_rate = init_learning_rate / 10
        else:  # 1000-1500
            learning_rate = init_learning_rate / 100
            pass

        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
            pass

        return learning_rate

    def load_model(self):
        if os.path.exists(Config.ic_dir):
            self.ic_model.load_state_dict(torch.load(Config.ic_dir))
            Tools.print("load ic model success from {}".format(Config.ic_dir))
        pass

    def train(self):
        Tools.print()
        Tools.print("Training...")

        # Init Update
        try:
            self.ic_model.eval()
            Tools.print("Init label {} .......")
            self.produce_class.reset()
            for image, label, idx in tqdm(self.ic_train_loader):
                image, idx = self.to_cuda(image), self.to_cuda(idx)
                ic_out_logits, ic_out_l2norm = self.ic_model(image)
                self.produce_class.cal_label(ic_out_l2norm, idx)
                pass
            Tools.print("Epoch: {}/{}".format(self.produce_class.count, self.produce_class.count_2))
        finally:
            pass

        for epoch in range(1, 1 + Config.train_epoch):
            self.ic_model.train()

            Tools.print()
            ic_lr = self.adjust_learning_rate(self.ic_model_optim, epoch)
            Tools.print('Epoch: [{}] ic_lr={}'.format(epoch, ic_lr))

            all_loss = 0.0
            self.produce_class.reset()
            for image, label, idx in tqdm(self.ic_train_loader):
                image, label, idx = self.to_cuda(image), self.to_cuda(label), self.to_cuda(idx)

                ###########################################################################
                # 1 calculate features
                ic_out_logits, ic_out_l2norm = self.ic_model(image)

                # 2 calculate labels
                ic_targets = self.produce_class.get_label(idx)
                self.produce_class.cal_label(ic_out_l2norm, idx)

                # 3 loss
                loss = self.ic_loss(ic_out_logits, ic_targets)
                all_loss += loss.item()

                # 4 backward
                self.ic_model.zero_grad()
                loss.backward()
                self.ic_model_optim.step()
                ###########################################################################
                pass

            ###########################################################################
            # print
            Tools.print("{:6} loss:{:.3f}".format(epoch, all_loss / len(self.ic_train_loader)))
            Tools.print("Train: [{}] {}/{}".format(epoch, self.produce_class.count, self.produce_class.count_2))
            ###########################################################################

            ###########################################################################
            # Val
            if epoch % Config.val_freq == 0:
                self.ic_model.eval()

                val_accuracy = self.test_tool_ic.val(epoch=epoch)
                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    torch.save(self.ic_model.state_dict(), Config.ic_dir)
                    Tools.print("Save networks for epoch: {}".format(epoch))
                    pass
                pass
            ###########################################################################
            pass

        pass

    pass


##############################################################################################################


"""
2020-12-24 01:09:25 load ic model success from ../cars/models/ic_res_xx/2_resnet_18_64_512_1_1700_300_200_False_ic.pkl
2020-12-24 01:09:25 Test 1700 .......
2020-12-24 01:09:36 Epoch: 1700 Train 0.1559/0.3656 0.0000
2020-12-24 01:09:36 Epoch: 1700 Val   0.2543/0.5729 0.0000
2020-12-24 01:09:36 Epoch: 1700 Test  0.1899/0.4250 0.0000


2020-12-25 04:36:54 load ic model success from ../cars/models/ic_res_xx/2_256_png_7_resnet_18_64_512_1_1700_300_200_False_transform2_ic.pkl
2020-12-25 04:36:54 Test 1700 .......
2020-12-25 04:37:02 Epoch: 1700 Train 0.1843/0.4068 0.0000
2020-12-25 04:37:02 Epoch: 1700 Val   0.3357/0.6706 0.0000
2020-12-25 04:37:02 Epoch: 1700 Test  0.2199/0.5438 0.0000


2020-12-25 08:14:47 load ic model success from ../cars/models/ic_res_xx/3_256_png_7_resnet_34_64_512_1_1700_300_200_True_transform2_ic.pkl
2020-12-25 08:14:47 Test 1700 .......
2020-12-25 08:14:56 Epoch: 1700 Train 0.1920/0.4251 0.0000
2020-12-25 08:14:56 Epoch: 1700 Val   0.3445/0.6810 0.0000
2020-12-25 08:14:56 Epoch: 1700 Test  0.2406/0.5645 0.0000


2020-12-25 16:42:52 load ic model success from ../cars/models/ic_res_xx/2_256_png_7_resnet_18_64_512_1_1700_300_200_False_ic.pkl
2020-12-25 16:42:52 Test 1700 .......
2020-12-25 16:43:01 Epoch: 1700 Train 0.1638/0.3723 0.0000
2020-12-25 16:43:01 Epoch: 1700 Val   0.3152/0.6513 0.0000
2020-12-25 16:43:01 Epoch: 1700 Test  0.2199/0.5072 0.0000


2020-12-25 20:37:25 load ic model success from ../cars/models/ic_res_xx/3_256_png_7_resnet_34_64_512_1_1700_300_200_True_ic.pkl
2020-12-25 20:37:25 Test 1700 .......
2020-12-25 20:37:34 Epoch: 1700 Train 0.1745/0.3927 0.0000
2020-12-25 20:37:34 Epoch: 1700 Val   0.3298/0.6823 0.0000
2020-12-25 20:37:34 Epoch: 1700 Test  0.2368/0.5609 0.0000


2020-12-26 22:19:01 load ic model success from ../cars/models/ic_res_xx/2_256_png_7_resnet_34_64_512_1_1700_300_200_True_ic.pkl
2020-12-26 22:19:01 Test 1700 .......
2020-12-26 22:19:12 Epoch: 1700 Train 0.2073/0.4512 0.0000
2020-12-26 22:19:12 Epoch: 1700 Val   0.3579/0.6940 0.0000
2020-12-26 22:19:12 Epoch: 1700 Test  0.2616/0.5723 0.0000


2020-12-27 06:18:22 load ic model success from ../cars/models/ic_res_xx/2_256_png_7_resnet_34_64_512_1_1700_300_200_True_1_ic.pkl
2020-12-27 06:18:22 Test 1700 .......
2020-12-27 06:18:34 Epoch: 1700 Train 0.1884/0.4233 0.0000
2020-12-27 06:18:34 Epoch: 1700 Val   0.3562/0.6982 0.0000
2020-12-27 06:18:34 Epoch: 1700 Test  0.2398/0.5697 0.0000
"""


##############################################################################################################


class Config(object):
    gpu_id = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    num_workers = 8
    val_freq = 10

    knn = 50

    batch_size = 64

    ic_out_dim = 512
    ic_ratio = 1
    ic_times = 1

    # resnet, vggnet, net_name = resnet18, None, "resnet_18"
    resnet, vggnet, net_name = resnet34, None, "resnet_34"

    # modify_head = False
    modify_head = True

    # class_split = "256_png"
    class_split = "256_png_7"

    learning_rate = 0.01
    train_epoch = 1700
    first_epoch, t_epoch = 300, 200
    adjust_learning_rate = Runner.adjust_learning_rate1

    model_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
        gpu_id, class_split, net_name, batch_size, ic_out_dim, ic_ratio, train_epoch, first_epoch, t_epoch, modify_head, ic_times)

    ic_dir = Tools.new_dir("../cars/models/ic_res_xx/{}_ic.pkl".format(model_name))
    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/UFSL/Cars'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/UFSL/Cars'
    else:
        data_root = "F:\\data\\Cars"
    data_root = os.path.join(data_root, class_split)

    Tools.print(model_name)
    Tools.print(data_root)
    Tools.print(ic_dir)
    pass


if __name__ == '__main__':
    runner = Runner()
    # runner.load_model()

    runner.ic_model.eval()
    runner.test_tool_ic.val(epoch=0, is_print=True)

    runner.train()

    runner.load_model()
    runner.ic_model.eval()
    runner.test_tool_ic.val(epoch=Config.train_epoch, is_print=True)
    pass
