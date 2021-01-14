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
from torchvision.models import resnet18, resnet34
sys.path.append("../Common")
from UFSLTool import MyTransforms, MyDataset, C4Net, Normalize, RunnerTool
from UFSLTool import ResNet12Small, FSLEvalTool, EvalFeatureDataset


##############################################################################################################


class Runner(object):

    def __init__(self, config):
        self.config = config

        # model
        self.norm = Normalize(2)
        self.matching_net = RunnerTool.to_cuda(self.config.matching_net)
        pass

    def load_model(self):
        mn_dir = self.config.mn_checkpoint
        print(mn_dir)
        if os.path.exists(mn_dir):
            try:
                self.matching_net.load_state_dict(torch.load(mn_dir))
            except Exception:
                temp=torch.load(mn_dir)
                # print(temp)
                temp = {"module.{}".format(key): temp[key] for key in temp}
                self.matching_net.load_state_dict(temp)
            Tools.print("load matching net success from {}".format(mn_dir), txt_path=self.config.log_file)
        else:
            print('you no load model')
        pass

    def matching_test(self, sample_z, batch_z, num_way, num_shot):
        batch_num, _, _, _ = batch_z.shape
        
        # sample_z = self.matching_net(samples)  # 5x64*5*5   2*512*1*1
        # batch_z = self.matching_net(batches)  # 75x64*5*5   6*512*1*1
        z_support = sample_z.view(num_way * num_shot, -1)#2*512*1*1>>>>>>2*512
        z_query = batch_z.view(batch_num, -1)#6*512*1*1>>>>>6*512
        _, z_dim = z_query.shape
        #  batch_num 6  num_way 2 num_shot 1  z_dim512 z_support_expand z_support: 2*512>>>>> 1*2*512>>>6*2*512  
        z_support_expand = z_support.unsqueeze(0).expand(batch_num, num_way * num_shot, z_dim)
        z_query_expand = z_query.unsqueeze(1).expand(batch_num, num_way * num_shot, z_dim)
        #z_query:[6,512]>>>>>> [6,1,512]>>>>>z_query_expand[6,2,512] 

        # 相似性
        z_support_expand = self.norm(z_support_expand)
        similarities = torch.sum(z_support_expand * z_query_expand, -1) #6*2*1
        similarities = torch.softmax(similarities, dim=1)
        similarities = similarities.view(batch_num, num_way, num_shot)
        predicts = torch.mean(similarities, dim=-1)
        return predicts

    def get_test_tool(self, image_features):
        test_tool_fsl = FSLEvalTool(model_fn=self.matching_test, data_root=self.config.data_root,
                                    num_way=self.config.num_way, num_shot=self.config.num_shot,
                                    episode_size=self.config.episode_size, test_episode=self.config.test_episode,
                                    image_features=image_features, txt_path=self.config.log_file)
        return test_tool_fsl

    def get_features(self):
        Tools.print("get_features")

        output_feature = {}
        with torch.no_grad():
            self.matching_net.eval()

            _, _, transform_test = MyTransforms.get_transform(
                dataset_name=self.config.dataset_name, has_ic=True, is_fsl_simple=True, is_css=False)
            data_test = MyDataset.get_data_split(self.config.data_root, split=self.config.split)#图片地址list 12000
            loader = DataLoader(EvalFeatureDataset(data_test, transform_test),
                                self.config.batch_size, False, num_workers=self.config.num_workers)
            for image, image_name in tqdm(loader):#image 64*3*32*32   image_name：图片地址list64
                output = self.matching_net(image.cuda()).data.cpu().numpy()#output 64*512*1*1
                for output_one, image_name_one in zip(output, image_name):
                    output_feature[image_name_one] = output_one#
                pass
            pass
        return output_feature #64 个图片对应的特征

    pass


##############################################################################################################


class Config(object):

    def __init__(self, gpu_id='1', name=None, is_conv_4=True, mn_checkpoint=None,
                 dataset_name=MyDataset.dataset_name_miniimagenet, result_dir="result",
                 split=MyDataset.dataset_split_test, is_check=False):
        self.gpu_id = gpu_id
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_id

        self.name = name
        self.split = split
        self.is_conv_4 = is_conv_4
        self.dataset_name = dataset_name
        self.num_way = 5
        self.num_shot = 1
        self.num_workers = 8
        self.episode_size = 15
        self.test_episode = 600
        self.mn_checkpoint = mn_checkpoint

        ###############################################################################################
        if self.is_conv_4:
            self.matching_net, self.batch_size = C4Net(hid_dim=64, z_dim=64), 64
        else:
            self.matching_net, self.batch_size = ResNet12Small(avg_pool=True, drop_rate=0.1), 64
        if len(self.gpu_id)>=2:
            Tools.print('DataParallel is True')
            self.matching_net  = torch.nn.DataParallel(self.matching_net , device_ids=range(torch.cuda.device_count()))
        ###############################################################################################

        self.is_check = is_check
        if self.is_check:
            self.log_file = None
            return

        # self.log_file = Tools.new_dir(os.path.join("/home/ubuntu/Documents/hzh/ActiveLearning/UFSLviaIC/models_abl/{}/mn/{}".format(self.dataset_name, result_dir),
        #                                            "{}_{}_{}.txt".format(split, self.name, Tools.get_format_time())))
        self.log_file = Tools.new_dir(os.path.join("/home/ubuntu/Documents/hzh/ActiveLearning/UFSLviaIC/models_abl/{}/mn/{}".format(self.dataset_name, result_dir),
                                                   "{}_{}.txt".format(split, self.name)))
        print(os.path.join("/home/ubuntu/Documents/hzh/ActiveLearning/UFSLviaIC/models_abl/{}/mn/{}".format(self.dataset_name, result_dir),"{}_{}.txt".format(split, self.name)))
        ###############################################################################################
        self.is_png = True
        self.data_root = MyDataset.get_data_root(dataset_name=self.dataset_name, is_png=self.is_png)
        _, _, self.transform_test = MyTransforms.get_transform(
            dataset_name=self.dataset_name, has_ic=True, is_fsl_simple=True, is_css=False)
        ###############################################################################################
        pass

    pass


##############################################################################################################


def final_eval(gpu_id, name, mn_checkpoint, dataset_name, is_conv_4,
               test_episode=1000, result_dir="result", split=MyDataset.dataset_split_test):
    config = Config(gpu_id, dataset_name=dataset_name, is_conv_4=is_conv_4,
                    name=name, mn_checkpoint=mn_checkpoint, result_dir=result_dir, split=split)
    runner = Runner(config=config)

    runner.load_model()
    runner.matching_net.eval()
    image_features = runner.get_features()
    test_tool_fsl = runner.get_test_tool(image_features=image_features)

    # ways, shots = MyDataset.get_ways_shots(dataset_name=dataset_name, split=split)
    # ways, shots = [5,20],[5]
    # for index, way in enumerate(ways):
    #     Tools.print("{}/{} way={}".format(index, len(ways), way))
    #     m, pm = test_tool_fsl.eval(num_way=way, num_shot=1, episode_size=15, test_episode=test_episode, split=split)
    #     Tools.print("way={},shot=1,acc={:.2f},con={:.2f}".format(way, m*100, pm*100), txt_path=config.log_file)
    # for index, shot in enumerate(shots):
    #     Tools.print("{}/{} shot={}".format(index, len(shots), shot))
    #     m, pm = test_tool_fsl.eval(num_way=5, num_shot=shot, episode_size=15, test_episode=test_episode, split=split)
    #     Tools.print("way=5,shot={},acc={:.2f},con={:.2f}".format(shot, m*100, pm*100), txt_path=config.log_file)
    way_shots=[[5,1],[5,5],[20,1],[20,5]]
    for index, way_shot in enumerate(way_shots):
        way,shot=way_shot[0],way_shot[1]
        Tools.print("{}/{} shot={}".format(index,len(way_shots),shot))
        m, pm = test_tool_fsl.eval(num_way=way, num_shot=shot, episode_size=15, test_episode=test_episode, split=split)
        Tools.print("way={},shot={},acc={:.2f},con={:.2f}".format(way,shot, m*100, pm*100), txt_path=config.log_file)

    pass
def select_eval(gpu_id, name, mn_checkpoint, dataset_name, is_conv_4,
               test_episode=1000, result_dir="result", split=MyDataset.dataset_split_test):
    config = Config(gpu_id, dataset_name=dataset_name, is_conv_4=is_conv_4,
                    name=name, mn_checkpoint=mn_checkpoint, result_dir=result_dir, split=split)
    runner = Runner(config=config)

    runner.load_model()
    runner.matching_net.eval()
    image_features = runner.get_features()
    test_tool_fsl = runner.get_test_tool(image_features=image_features)

    ways, shots =[5],[1]
    for index, way in enumerate(ways):
        Tools.print("{}/{} way={}".format(index, len(ways), way))
        m, pm = test_tool_fsl.eval(num_way=way, num_shot=1, episode_size=15, test_episode=test_episode, split=split)
        Tools.print("way={},shot=1,acc={},con={}".format(way, m, pm), txt_path=config.log_file)
    for index, shot in enumerate(shots):
        Tools.print("{}/{} shot={}".format(len(shots),index , shot))
        m, pm = test_tool_fsl.eval(num_way=5, num_shot=shot, episode_size=15, test_episode=test_episode, split=split)
        Tools.print("way=5,shot={},acc={},con={}".format(shot, m, pm), txt_path=config.log_file)
    pass


# elif dataset_name == MyDataset.dataset_name_select:
#     if split == MyDataset.dataset_split_test:
#         ways = [5]
#         shots = [1]
#     else:
#             raise Exception(".")
def FC100_final_eval(gpu_id='0', result_dir="result_table"):
    dataset_name = MyDataset.dataset_name_FC100
    checkpoint_path = "/home/ubuntu/Documents/hzh/ActiveLearning/UFSLviaIC/models_abl/{}/mn".format(dataset_name)

    param_list = [
        {"name": "cluster_conv4", "is_conv_4": True,'gpu':'0',
         "mn": os.path.join(checkpoint_path, "cluster", "0_cluster_conv4_300_64_5_1_100_100_png.pkl")},#
        {"name": "css_conv4", "is_conv_4": True,'gpu':'1',
         "mn": os.path.join(checkpoint_path, "css", "1_css_conv4_300_64_5_1_100_100_png.pkl")},#
        {"name": "random_conv4", "is_conv_4": True,'gpu':'0',
         "mn": os.path.join(checkpoint_path, "random", "0_random_conv4_300_64_5_1_100_100_png.pkl")},#
        {"name": "label_conv4", "is_conv_4": True,'gpu':'01',
         "mn": os.path.join(checkpoint_path, "label", "2_400_64_5_1_200_100_png_mn_5way_1shot_FC100.pkl")},
        {"name": "ufsl_res34_conv4", "is_conv_4": True,'gpu':'1',
         "mn": os.path.join(checkpoint_path, "ufsl", "1_2100_64_5_1_500_200_512_1_1.0_1.0_head_png_mn_5way_1shot_FC100.pkl")},


        {"name": "cluster_res12", "is_conv_4": False,'gpu':'0',
         "mn": os.path.join(checkpoint_path, "cluster", "0_cluster_res12_100_32_5_1_60_80_png.pkl")},#
        {"name": "css_res12", "is_conv_4": False,'gpu':'1',
         "mn": os.path.join(checkpoint_path, "css", "1_css_res12_100_32_5_1_60_80_png.pkl")},#
        {"name": "random_res12", "is_conv_4": False,'gpu':'0',
         "mn": os.path.join(checkpoint_path, "random", "0_random_res12_100_32_5_1_60_80_png.pkl")},#
        {"name": "label_res12", "is_conv_4": False,'gpu':'0',
         "mn": os.path.join(checkpoint_path, "label", "fsl-(150)EP400_BS128_mn_5way_1shot_DR0.3_res12_FC100.pkl")},

        {"name": "ufsl_res34head_res12", "is_conv_4": False,'gpu':'0',
         "mn": os.path.join(checkpoint_path, "ufsl", "fsl-1080EP2100_BS128_ft500_200_mn_5w1s_DR0.1_res12_lr2_FC100_RGB.pkl")},
    ]
        # "fsl-(150)EP400_BS128_mn_5way_1shot_DR0.3_res12_FC100.pkl"
        #  fsl-250EP600_BS64_ft200_100_mn_5w1s_DR0.2_res12_lr3_FC100_RGB.pkl
    for index, param in enumerate(param_list):
        Tools.print("Check: {} / {}".format(index, len(param_list)))
        Runner(config=Config(gpu_id=param['gpu'], dataset_name=dataset_name, is_conv_4=param["is_conv_4"],
                             name=param["name"], mn_checkpoint=param["mn"], is_check=True)).load_model()
        pass

    for index, param in enumerate(param_list):
        Tools.print("{} / {}".format(index, len(param_list)))
        final_eval(gpu_id=param['gpu'], name=param["name"], mn_checkpoint=param["mn"],
                   dataset_name=dataset_name, is_conv_4=param["is_conv_4"], result_dir=result_dir)
        pass

    pass

def CIFARFS_final_eval(gpu_id='0', result_dir="result"):
    dataset_name = MyDataset.dataset_name_CIFARFS
    checkpoint_path = "/home/ubuntu/Documents/hzh/ActiveLearning/UFSLviaIC/models_abl/{}/mn".format(dataset_name)

    param_list = [
        {"name": "cluster_conv4", "is_conv_4": True,'gpu':'1',
         "mn": os.path.join(checkpoint_path, "cluster", "1_cluster_conv4_300_64_5_1_100_100_png.pkl")},#
        {"name": "css_conv4", "is_conv_4": True,'gpu':'0',
         "mn": os.path.join(checkpoint_path, "css", "0_css_conv4_300_64_5_1_100_100_png.pkl")},#
        {"name": "random_conv4", "is_conv_4": True,'gpu':'0',
         "mn": os.path.join(checkpoint_path, "random", "0_random_conv4_300_64_5_1_100_100_png.pkl")},#
        {"name": "label_conv4", "is_conv_4": True,'gpu':'1',
         "mn": os.path.join(checkpoint_path, "label", "2_400_64_5_1_200_100_png_mn_5way_1shot_CIFARFS.pkl")},
        {"name": "ufsl_res34_conv4", "is_conv_4": True,'gpu':'0',
         "mn": os.path.join(checkpoint_path, "ufsl", "1_2100_64_5_1_500_200_512_1_1.0_1.0_head_png_mn_5way_1shot_CIFARFS.pkl")},


        {"name": "cluster_res12", "is_conv_4": False,'gpu':'1',
         "mn": os.path.join(checkpoint_path, "cluster", "1_cluster_res12_100_32_5_1_60_80_png.pkl")},#
        {"name": "css_res12", "is_conv_4": False,'gpu':'1',
         "mn": os.path.join(checkpoint_path, "css", "1_css_res12_100_32_5_1_60_80_png.pkl")},#
        {"name": "random_res12", "is_conv_4": False,'gpu':'0',
         "mn": os.path.join(checkpoint_path, "random", "0_random_res12_100_32_5_1_60_80_png.pkl")},#
        {"name": "label_res12", "is_conv_4": False,'gpu':'01',
         "mn": os.path.join(checkpoint_path, "label", "fsl-(130)EP400_BS128_mn_5way_1shot_DR0.1_res12_CIFARFS.pkl")},
        {"name": "ufsl_res34_res12", "is_conv_4": False,'gpu':'0',
         "mn": os.path.join(checkpoint_path, "ufsl", "1_2100_64_5_1_500_200_512_1_1.0_1.0_head_png_res12_mn_5way_1shot_CIFARFS.pkl")},
    ]

    for index, param in enumerate(param_list):
        Tools.print("Check: {} / {}".format(index, len(param_list)))
        Runner(config=Config(gpu_id=param['gpu'], dataset_name=dataset_name, is_conv_4=param["is_conv_4"],
                             name=param["name"], mn_checkpoint=param["mn"], is_check=True)).load_model()
        pass

    for index, param in enumerate(param_list):
        Tools.print("{} / {}".format(index, len(param_list)))
        final_eval(gpu_id=param['gpu'], name=param["name"], mn_checkpoint=param["mn"],
                   dataset_name=dataset_name, is_conv_4=param["is_conv_4"], result_dir=result_dir)
        pass

    pass

def miniimagenet_our_eval(gpu_id='0', result_dir="result_our"):
    dataset_name = MyDataset.dataset_name_FC100
    checkpoint_path = "/home/ubuntu/Documents/hzh/ActiveLearning/UFSLviaIC/models_abl/{}/mn".format(dataset_name)

    param_list = [
        {"name": "ufsl_res18_conv4", "is_conv_4": True,
         "mn": os.path.join(checkpoint_path, "ufsl", "1_2100_64_5_1_500_200_512_1_1.0_1.0_mn.pkl")},
        {"name": "ufsl_res34head_conv4", "is_conv_4": True,
         "mn": os.path.join(checkpoint_path, "ufsl", "3_2100_64_5_1_500_200_512_1_1.0_1.0_head_png_mn.pkl")},

        {"name": "ufsl_res34head_res12", "is_conv_4": False,
         "mn": os.path.join(checkpoint_path, "ufsl", "1_R12S_1500_32_5_1_300_200_512_1_1.0_1.0_head_png_mn.pkl")},
        {"name": "ufsl_res34head_res12", "is_conv_4": False,
         "mn": os.path.join(checkpoint_path, "ufsl", "2_R12S_1500_32_5_1_500_200_512_1_1.0_1.0_head_png_mn.pkl")},
    ]

    for index, param in enumerate(param_list):
        Tools.print("Check: {} / {}".format(index, len(param_list)))
        Runner(config=Config(gpu_id, dataset_name=dataset_name, is_conv_4=param["is_conv_4"],
                             name=param["name"], mn_checkpoint=param["mn"], is_check=True)).load_model()
        pass

    for index, param in enumerate(param_list):
        Tools.print("{} / {}".format(index, len(param_list)))
        final_eval(gpu_id, name=param["name"], mn_checkpoint=param["mn"], dataset_name=dataset_name,
                   is_conv_4=param["is_conv_4"], result_dir=result_dir, split=MyDataset.dataset_split_train)
        final_eval(gpu_id, name=param["name"], mn_checkpoint=param["mn"], dataset_name=dataset_name,
                   is_conv_4=param["is_conv_4"], result_dir=result_dir, split=MyDataset.dataset_split_val)
        final_eval(gpu_id, name=param["name"], mn_checkpoint=param["mn"], dataset_name=dataset_name,
                   is_conv_4=param["is_conv_4"], result_dir=result_dir, split=MyDataset.dataset_split_test)
        pass

    pass

def select_model_label_Ufsl_FC100(gpu_id='0', result_dir="result_select"):
    dataset_name = MyDataset.dataset_name_FC100
    checkpoint_path = "/home/ubuntu/Documents/hzh/ActiveLearning/UFSLviaIC/models_abl/{}/mn".format(dataset_name)

    param_list = [
        {"name": "select_FC100", "is_conv_4": False,'gpu':'1',
         "mn": os.path.join(checkpoint_path, "label", "fsl-(150)EP400_BS128_mn_5way_1shot_DR0.3_res12_FC100.pkl")},#
        {"name": "select_FC100", "is_conv_4": False,'gpu':'01',
         "mn": os.path.join(checkpoint_path, "label", "fsl-250EP600_BS64_ft200_100_mn_5w1s_DR0.2_res12_lr3_FC100_RGB.pkl")},#
        {"name": "select_FC100", "is_conv_4": False,'gpu':'01',
         "mn": os.path.join(checkpoint_path, "label", "140-400_2_64_5_1_200_100_png_mn_5way_1shot_FC100.pkl")},#
        {"name": "select_FC100", "is_conv_4": True,'gpu':'01',
         "mn": os.path.join(checkpoint_path, "label", "2_400_64_5_1_200_100_png_mn_5way_1shot_FC100.pkl")},#

        {"name": "select_FC100", "is_conv_4": True,'gpu':'0',
         "mn": os.path.join(checkpoint_path, "ufsl", "1_2100_64_5_1_500_200_512_1_1.0_1.0_head_png_mn_5way_1shot_FC100.pkl")},#
        {"name": "select_FC100", "is_conv_4": True,'gpu':'0',
         "mn": os.path.join(checkpoint_path, "ufsl", "4_2100_64_5_1_500_200_512_1_1.0_1.0_head_png_mn_5way_1shot_FC100.pkl")},#
        {"name": "select_FC100", "is_conv_4": False,'gpu':'0',
         "mn": os.path.join(checkpoint_path, "ufsl", "1_2100_64_5_1_500_200_512_1_1.0_1.0_head_png_res12_mn_5way_1shot_FC100.pkl")},#
        {"name": "select_FC100", "is_conv_4": False,'gpu':'0',
         "mn": os.path.join(checkpoint_path, "ufsl", "fsl-1080EP2100_BS128_ft500_200_mn_5w1s_DR0.1_res12_lr2_FC100_RGB.pkl")},#

    ]

    for index, param in enumerate(param_list):
        Tools.print("Check: {} / {}".format(index, len(param_list)))
        Runner(config=Config(gpu_id=param['gpu'], dataset_name=dataset_name, is_conv_4=param["is_conv_4"],
                             name=param["name"], mn_checkpoint=param["mn"], is_check=True)).load_model()
        pass

    for index, param in enumerate(param_list):
        Tools.print("{} / {}".format(index, len(param_list)))
        select_eval(gpu_id=param['gpu'], name=param["name"], mn_checkpoint=param["mn"],
                   dataset_name=dataset_name, is_conv_4=param["is_conv_4"], result_dir=result_dir)
        pass

    pass

##############################################################################################################
def select_model_label_Ufsl_CIFARFS(gpu_id='0', result_dir="result_select"):
    dataset_name = MyDataset.dataset_name_CIFARFS
    checkpoint_path = "/home/ubuntu/Documents/hzh/ActiveLearning/UFSLviaIC/models_abl/{}/mn".format(dataset_name)

    param_list = [
        {"name": "select_CIFARFS", "is_conv_4": False,'gpu':'01',
         "mn": os.path.join(checkpoint_path, "label", "fsl-(130)EP400_BS128_mn_5way_1shot_DR0.1_res12_CIFARFS.pkl")},#
        {"name": "select_CIFARFS", "is_conv_4": True,'gpu':'1',
         "mn": os.path.join(checkpoint_path, "label", "2_400_64_5_1_200_100_png_mn_5way_1shot_CIFARFS.pkl")},#

        {"name": "select_CIFARFS", "is_conv_4": True,'gpu':'0',
         "mn": os.path.join(checkpoint_path, "ufsl", "4_2100_64_5_1_500_200_512_1_1.0_1.0_head_png_mn_5way_1shot_CIFARFS.pkl")},#
        {"name": "select_CIFARFS", "is_conv_4": True,'gpu':'0',
         "mn": os.path.join(checkpoint_path, "ufsl", "1_2100_64_5_1_500_200_512_1_1.0_1.0_head_png_mn_5way_1shot_CIFARFS.pkl")},#
        {"name": "select_CIFARFS", "is_conv_4": False,'gpu':'0',
         "mn": os.path.join(checkpoint_path, "ufsl", "1_2100_64_5_1_500_200_512_1_1.0_1.0_head_png_res12_mn_5way_1shot_CIFARFS.pkl")},#

    ]

    for index, param in enumerate(param_list):
        Tools.print("Check: {} / {}".format(index, len(param_list)))
        Runner(config=Config(gpu_id=param['gpu'], dataset_name=dataset_name, is_conv_4=param["is_conv_4"],
                             name=param["name"], mn_checkpoint=param["mn"], is_check=True)).load_model()
        pass

    for index, param in enumerate(param_list):
        Tools.print("{} / {}".format(index, len(param_list)))
        select_eval(gpu_id=param['gpu'], name=param["name"], mn_checkpoint=param["mn"],
                   dataset_name=dataset_name, is_conv_4=param["is_conv_4"], result_dir=result_dir)
        pass

    pass

if __name__ == '__main__':
    # miniimagenet_our_eval()
    # miniimagenet_final_eval()

    # CIFARFS_final_eval()
    FC100_final_eval()

    # select_model_label_Ufsl_FC100()
    # select_model_label_Ufsl_CIFARFS()
    pass
