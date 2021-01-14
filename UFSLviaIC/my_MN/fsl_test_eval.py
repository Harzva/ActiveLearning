import os
import math
import torch
import random
import platform
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from alisuretool.Tools import Tools
from mn_fsl_test_tool import FSLTestTool
from mn_tool import MatchingNet, Normalize, RunnerTool,ResNet12Small,CNNEncoder,Classifier
##############################################################################################################
import argparse

class Runner(object):

    def __init__(self):
        self.matching_net = RunnerTool.to_cuda(Config.matching_net)
        pass

    def load_model(self):
        if os.path.exists(Config.mn_dir):
            self.matching_net.load_state_dict(torch.load(Config.mn_dir))
            Tools.print("load proto network success from {}".format(Config.mn_dir))
        pass

    def compare_fsl_test(self, samples, batches, num_way, num_shot):
        batch_num, _, _, _ = batches.shape

        sample_z = self.matching_net(samples)  # 5x64*5*5
        batch_z = self.matching_net(batches)  # 75x64*5*5
        sample_z = sample_z.view(num_way, num_shot, -1)
        batch_z = batch_z.view(batch_num, -1)
        _, z_dim = batch_z.shape

        z_proto = sample_z.mean(1)
        z_proto_expand = z_proto.unsqueeze(0).expand(batch_num, num_way, z_dim)
        z_query_expand = batch_z.unsqueeze(1).expand(batch_num, num_way, z_dim)

        dists = torch.pow(z_query_expand - z_proto_expand, 2).sum(2)
        log_p_y = F.log_softmax(-dists, dim=1)
        return log_p_y

    def eval_one(self, num_way=5, num_shot=1):
        Tools.print("way:{} shot:{}".format(num_way, num_shot))

        self.matching_net.eval()
        test_tool_fsl = FSLTestTool(self.compare_fsl_test, Config.data_root, num_way=num_way, num_shot=num_shot,
                                 episode_size=Config.episode_size, test_episode=Config.test_episode)
        test_tool_fsl.val(is_print=True)
        test_tool_fsl.test(test_avg_num=5, is_print=True)
        pass

    def eval(self):
        self.eval_one(num_way=5, num_shot=1)
        self.eval_one(num_way=5, num_shot=5)
        self.eval_one(num_way=5, num_shot=10)
        self.eval_one(num_way=5, num_shot=20)
        self.eval_one(num_way=10, num_shot=1)
        self.eval_one(num_way=10, num_shot=5)
        self.eval_one(num_way=10, num_shot=10)
        self.eval_one(num_way=10, num_shot=20)
        pass

    pass


##############################################################################################################

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
    parser.add_argument('--convert',type=str, default='L',help=' Image.open(x).convert(RGB)') 
    parser.add_argument('--dataset','-ds',type=str,default='omniglot_single',help='dataset _name')     
    args = parser.parse_args()
    return args

class Config(object):
    args = parse_args()
    gpu_id = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    num_workers = 8
    episode_size = 15
    test_episode = 600
    test_avg_num = 5
    drop_rate=args.drop_rate
    hid_dim = 64
    z_dim = 64

    # has_norm = True
    has_norm = False
    model_path = "./models_mn_eval/two_ic_ufsl_2net_res_sgd_acc_duli"
    # model_pn_name = "2_2100_64_5_1_64_64_500_200_512_1_1.0_1.0_norm_pn_5way_1shot.pkl"
    model_pn_name = "/home/ubuntu/Documents/hzh/ActiveLearning/UFSLviaIC/models_mn/fsl_sgd_modify/Dec23_13-55-32_EP400_BS128_mn_5way_1shot_DR0.1_res12_CIFARFS/fsl-(130)EP400_BS128_mn_5way_1shot_DR0.1_res12_CIFARFS.pkl"
    mn_dir = Tools.new_dir(os.path.join(model_path, model_pn_name))
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

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/miniImagenet'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
    else:
        data_root = "F:\\data\\miniImagenet"

    pass


"""
../models_pn/two_ic_ufsl_2net_res_sgd_acc_duli/1_2100_64_5_1_64_64_500_200_512_1_1.0_1.0_pn_5way_1shot.pkl
2020-12-01 09:12:39 way:5 shot:1
2020-12-01 09:14:18 Train 0 Accuracy: 0.4772222222222222
2020-12-01 09:14:18 Val   0 Accuracy: 0.4313333333333332
2020-12-01 09:18:17 episode=0, Mean Test accuracy=0.44584
2020-12-01 09:18:17 way:5 shot:5
2020-12-01 09:20:41 Train 0 Accuracy: 0.6297333333333333
2020-12-01 09:20:41 Val   0 Accuracy: 0.5904
2020-12-01 09:25:18 episode=0, Mean Test accuracy=0.6101822222222222
2020-12-01 09:25:18 way:5 shot:10
2020-12-01 09:28:27 Train 0 Accuracy: 0.6782666666666667
2020-12-01 09:28:27 Val   0 Accuracy: 0.6444666666666666
2020-12-01 09:34:10 episode=0, Mean Test accuracy=0.658831111111111
2020-12-01 09:34:10 way:5 shot:20
2020-12-01 09:38:45 Train 0 Accuracy: 0.7033999999999999
2020-12-01 09:38:45 Val   0 Accuracy: 0.6714
2020-12-01 09:46:33 episode=0, Mean Test accuracy=0.6877244444444444
2020-12-01 09:46:33 way:10 shot:1
2020-12-01 09:49:41 Train 0 Accuracy: 0.3311111111111111
2020-12-01 09:49:41 Val   0 Accuracy: 0.2951666666666667
2020-12-01 09:57:11 episode=0, Mean Test accuracy=0.30263555555555555
2020-12-01 09:57:11 way:10 shot:5
2020-12-01 10:01:54 Train 0 Accuracy: 0.5
2020-12-01 10:01:54 Val   0 Accuracy: 0.43606666666666666
2020-12-01 10:11:15 episode=0, Mean Test accuracy=0.4631955555555556
2020-12-01 10:11:15 way:10 shot:10
2020-12-01 10:17:29 Train 0 Accuracy: 0.5481333333333334
2020-12-01 10:17:29 Val   0 Accuracy: 0.49
2020-12-01 10:28:47 episode=0, Mean Test accuracy=0.5140933333333334
2020-12-01 10:28:47 way:10 shot:20
2020-12-01 10:38:02 Train 0 Accuracy: 0.5807666666666668
2020-12-01 10:38:02 Val   0 Accuracy: 0.5299333333333334
2020-12-01 10:53:48 episode=0, Mean Test accuracy=0.5488977777777777

"""


if __name__ == '__main__':
    runner = Runner()

    runner.load_model()
    runner.eval()
    pass
