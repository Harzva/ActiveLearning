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
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, resnet34, resnet50, vgg16_bn


##############################################################################################################


# class MiniImageNetIC(Dataset):

#     def __init__(self, data_list, image_size=84):
#         self.data_list = data_list
#         self.train_label = [one[1] for one in self.data_list]

#         norm = transforms.Normalize(mean=[x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]],
#                                     std=[x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])
#         self.transform = transforms.Compose([transforms.CenterCrop(size=image_size), transforms.ToTensor(), norm])
#         self.transform2 = transforms.Compose([transforms.ToTensor()])
#         pass
# class MiniImageNetIC(Dataset):
    
#     def __init__(self, data_list, image_size=32):
#         self.data_list = data_list
#         self.train_label = [one[1] for one in self.data_list]
#         normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
#                                          std= [x / 255.0 for x in [63.0, 62.1, 66.7]])
#         self.transform = transforms.Compose(
#                 [transforms.ToTensor(), normalize]
#         )
#         self.transform2 =transforms.Compose(
#                 [transforms.ToTensor()]
#         )
#         pass
class MiniImageNetIC(Dataset):
    
    def __init__(self, data_list, image_size=28):
        self.data_list = data_list
        self.train_label = [one[1] for one in self.data_list]


        normalize = transforms.Normalize(mean=[0.92206], std=[0.08426])
        # self.transform = transforms.Compose([transforms.RandomRotation(30),
        #                                     transforms.Resize(image_size),
        #                                     transforms.RandomCrop(image_size, padding=4, fill=255),
        #                                     transforms.ToTensor(), normalize])
        # self.transform_test = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), normalize])

        self.transform = transforms.Compose(
                [transforms.Resize(image_size),transforms.ToTensor(), normalize]
        )
        self.transform2 =transforms.Compose(
                [transforms.Resize(image_size),transforms.ToTensor()]
        )
        pass


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        idx, label, image_filename = self.data_list[idx]
        image = Image.open(image_filename).convert('RGB')
        image_transform = self.transform(image)
        image_transform2 = self.transform2(image)
        return image_transform, image_transform2, label, idx

    @staticmethod
    def get_data_all(data_root):
        train_folder = os.path.join(data_root, "train")
        val_folder = os.path.join(data_root, "val")
        test_folder = os.path.join(data_root, "test")

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

        count_image, count_class, data_val_list = 0, 0, []
        for label in os.listdir(val_folder):
            now_class_path = os.path.join(val_folder, label)
            if os.path.isdir(now_class_path):
                for name in os.listdir(now_class_path):
                    data_val_list.append((count_image, count_class, os.path.join(now_class_path, name)))
                    count_image += 1
                    pass
                count_class += 1
            pass

        count_image, count_class, data_test_list = 0, 0, []
        for label in os.listdir(test_folder):
            now_class_path = os.path.join(test_folder, label)
            if os.path.isdir(now_class_path):
                for name in os.listdir(now_class_path):
                    data_test_list.append((count_image, count_class, os.path.join(now_class_path, name)))
                    count_image += 1
                    pass
                count_class += 1
            pass

        return data_train_list, data_val_list, data_test_list

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


# class ICResNet(nn.Module):

#     def __init__(self, low_dim=512, modify_head=False, resnet=None, vggnet=None):
#         super().__init__()
#         self.is_res = True if resnet else False
#         self.is_vgg = True if vggnet else False

#         if self.is_res:
#             self.resnet = resnet(num_classes=low_dim)
#             if modify_head:
#                 self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#                 pass
#         elif self.is_vgg:
#             self.vggnet = vggnet()
#             self.avgpool = nn.AdaptiveAvgPool2d(1)
#             self.fc = nn.Linear(512, low_dim)
#             pass
#         else:
#             raise Exception("......")

#         self.l2norm = Normalize(2)
#         pass

#     def forward(self, x):
#         if self.is_res:
#             out_logits = self.resnet(x)
#         elif self.is_vgg:
#             features = self.vggnet.features(x)
#             features = self.avgpool(features)
#             features = torch.flatten(features, 1)
#             out_logits = self.fc(features)
#             pass
#         else:
#             raise Exception("......")

#         out_l2norm = self.l2norm(out_logits)
#         return out_logits, out_l2norm

#     def __call__(self, *args, **kwargs):
#         return super().__call__(*args, **kwargs)

#     pass
class C4Net(nn.Module):
    
    def __init__(self, hid_dim, z_dim, has_norm=False):
        super().__init__()
        self.conv_block_1 = nn.Sequential(nn.Conv2d(3, hid_dim, 3, padding=1),
                                          nn.BatchNorm2d(hid_dim), nn.ReLU(), nn.MaxPool2d(2))  # 41
        self.conv_block_2 = nn.Sequential(nn.Conv2d(hid_dim, hid_dim, 3, padding=1),
                                          nn.BatchNorm2d(hid_dim), nn.ReLU(), nn.MaxPool2d(2))  # 21
        self.conv_block_3 = nn.Sequential(nn.Conv2d(hid_dim, hid_dim, 3, padding=1),
                                          nn.BatchNorm2d(hid_dim), nn.ReLU(), nn.MaxPool2d(2))  # 10
        self.conv_block_4 = nn.Sequential(nn.Conv2d(hid_dim, z_dim, 3, padding=1),
                                          nn.BatchNorm2d(z_dim), nn.ReLU(), nn.MaxPool2d(2))  # 5

        self.has_norm = has_norm
        if self.has_norm:
            self.norm = Normalize(2)
        pass

    def forward(self, x):
        out = self.conv_block_1(x)
        out = self.conv_block_2(out)
        out = self.conv_block_3(out)
        out = self.conv_block_4(out)

        if self.has_norm:
            out = out.view(out.shape[0], -1)
            out = self.norm(out)
        return out

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
class EncoderC4(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.encoder = C4Net(64, 64)
        self.out_dim = 64
        pass

    def forward(self, x):
        out = self.encoder(x)
        out = torch.flatten(out, 1)
        return out

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass

class ICResNet(nn.Module):

    def __init__(self, encoder, low_dim=512):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(self.encoder.out_dim, low_dim)
        self.l2norm = Normalize(2)
        pass

    def forward(self, x):
        out = self.encoder(x)
        out_logits = self.fc(out)
        out_l2norm = self.l2norm(out_logits)
        return out_logits, out_l2norm

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


##############################################################################################################


class Runner(object):

    def __init__(self):
        # data
        self.data_train, self.data_val, self.data_test = MiniImageNetIC.get_data_all(Config.data_root)
        self.train_loader = DataLoader(MiniImageNetIC(self.data_train), Config.batch_size, False, num_workers=Config.num_workers)
        self.val_loader = DataLoader(MiniImageNetIC(self.data_val), Config.batch_size, False, num_workers=Config.num_workers)
        self.test_loader = DataLoader(MiniImageNetIC(self.data_test), Config.batch_size, False, num_workers=Config.num_workers)

        # model
        # self.ic_model = self.to_cuda(ICResNet(Config.ic_out_dim, modify_head=Config.modify_head,
        #                                       resnet=Config.resnet, vggnet=Config.vggnet))
        self.ic_model = self.to_cuda(ICResNet(low_dim=Config.ic_out_dim, encoder=Config.ic_net))
        pass

    @staticmethod
    def to_cuda(x):
        return x.cuda() if torch.cuda.is_available() else x

    def load_model(self):
        if os.path.exists(Config.ic_dir):
            try:
                self.ic_model.load_state_dict(torch.load(Config.ic_dir))
                # self.ic_model  = torch.nn.DataParallel(self.ic_model , device_ids=[0,1])
            except:
                temp=torch.load(Config.ic_dir)
                print('*'*60,list(temp.keys())[0])
                # print('*'*60,type(list(temp.keys())[0]))

                # print('*'*60,list(temp.keys())[0].replace('module','resnet'))
                temp={'resnet.'+i:temp[i] for i in temp }
                # temp={i.replace('module.',''):temp[i] for i in temp }
                # print(temp,'*'*60)
                self.ic_model.load_state_dict(temp)
            Tools.print("load ic model success from {}".format(Config.ic_dir))
        pass

    def vis(self, split="train"):
        Tools.print()
        Tools.print("Vis ...")

        loader = self.test_loader if split == "test" else self.train_loader
        loader = self.val_loader if split == "val" else loader

        feature_list = []
        self.ic_model.eval()
        for image_transform, image, label, idx in tqdm(loader):
            ic_out_logits, ic_out_l2norm = self.ic_model(self.to_cuda(image_transform))

            image_data = np.asarray(image.permute(0, 2, 3, 1) * 255, np.uint8)
            cluster_id = np.asarray(torch.argmax(ic_out_logits, -1).cpu())
            for i in range(len(idx)):
                feature_list.append([int(idx[i]), int(label[i]), int(cluster_id[i]),
                                     np.array(ic_out_logits[i].cpu().detach().numpy()),
                                     np.array(ic_out_l2norm[i].cpu().detach().numpy())])
                temp=int(ic_out_l2norm[i].cpu().detach().numpy()[cluster_id[i]]*100)
                
                result_path = Tools.new_dir(os.path.join(Config.vis_dir, split, str(cluster_id[i])))
                Image.fromarray(image_data[i]).save(os.path.join(result_path, "{}_{}_{}.png".format(temp,label[i], idx[i])))
                pass
            pass

        Tools.write_to_pkl(os.path.join(Config.vis_dir, "{}.pkl".format(split)), feature_list)
        pass

    pass


##############################################################################################################


class Config(object):
    gpu_id = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    num_workers = 8
    batch_size = 8

    # resnet, vggnet, net_name = resnet18, None, "resnet_18"
    resnet, vggnet, net_name = resnet34, None, "resnet_34"
    # resnet, vggnet, net_name = resnet50, None, "resnet_50"
    # resnet, vggnet, net_name = None, vgg16_bn, "vgg16_bn"

    # modify_head = False
    modify_head = True

    is_png = True
    # is_png = False

    # ic
#####################################3miniImageNet_png
    # ic_dir = "/home/ubuntu/Documents/hzh/ActiveLearning/UFSLviaIC/IC/IC_model/3_resnet_34_64_512_1_2100_500_200_ic.pkl"
    # ic_out_dim = 512
    # vis_dir = Tools.new_dir("UFSLviaIC/IC/IC_result/3_resnet_34_64_512_1_2100_500_200")
    # dataset='miniImageNet'
    # data_root = f'/home/ubuntu/Dataset/Partition1/hzh/data/{dataset}'
    # if not os.path.exists(data_root):
    #     data_root = f'/home/ubuntu/Dataset/Partition1/hzh/data/{dataset}'
    # data_root = os.path.join(data_root, "miniImageNet_png") if is_png else data_root
    # Tools.print(data_root)
##########################################tiered-imagenet
    # ic_dir='/home/ubuntu/Documents/hzh/ActiveLearning/UFSLviaIC/IC/IC_model/123_res34_head_1200_384_2048_conv4_100_5_1_288_ic_tiered.pkl'
    # ic_out_dim = 2048
    # vis_dir = Tools.new_dir("UFSLviaIC/IC/IC_result/123_res34_head_1200_384_2048_conv4_100_5_1_288_ic_tiered_score")
    # dataset='tiered-imagenet'
    # data_root = f'/home/ubuntu/Dataset/Partition1/hzh/data/{dataset}'
    # if not os.path.exists(data_root):
    #     data_root = f'/home/ubuntu/Dataset/Partition1/hzh/data/{dataset}'
    # pass
##########################################tiered-imagenet
    # ic_dir='/home/ubuntu/Documents/hzh/ActiveLearning/UFSLviaIC/my_MN/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli_CIFARFS/eval-res12_1_2100_64_5_1_500_200_512_1_1.0_1.0_head_png_res12_ic_CIFARFS/1_2100_64_5_1_500_200_512_1_1.0_1.0_head_png_res12_ic_CIFARFS.pkl'
    # ic_out_dim = 512
    # vis_dir = Tools.new_dir("UFSLviaIC/IC/IC_result/1_2100_64_5_1_500_200_512_1_1.0_1.0_head_png_res12_ic_CIFARFS")
    # dataset='CIFARFS'
    # data_root = f'/home/ubuntu/Dataset/Partition1/hzh/data/{dataset}'
    # if not os.path.exists(data_root):
    #     data_root = f'/home/ubuntu/Dataset/Partition1/hzh/data/{dataset}'
    # pass
##########################################omniglot
    ic_dir='/home/ubuntu/Documents/hzh/ActiveLearning/UFSLviaIC/IC/IC_model/2_28_ICConv4_64_1024_1_1600_1000_300_ic_omniglot.pkl'
    ic_out_dim = 1024
    ic_net, net_name = EncoderC4(), "ICConv4"
    vis_dir = Tools.new_dir("UFSLviaIC/IC/IC_result/2_28_ICConv4_64_1024_1_1600_1000_300_ic_omniglot")
    dataset='omniglot_single'
    data_root = f'/home/ubuntu/Dataset/Partition1/hzh/data/{dataset}'
    if not os.path.exists(data_root):
        data_root = f'/home/ubuntu/Dataset/Partition1/hzh/data/{dataset}'
    pass

if __name__ == '__main__':
    runner = Runner()
    runner.load_model()

    runner.vis(split="train")
    runner.vis(split="val")
    runner.vis(split="test")

    pass
