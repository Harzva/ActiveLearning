import os
import torch
import random
import numpy as np
import torch.nn as nn
from PIL import Image
from alisuretool.Tools import Tools
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm 
##############################################################################################################


class ClassBalancedSampler(Sampler):
    """ Samples 'num_inst' examples each from 'num_cl' pools of examples of size 'num_per_class' """

    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        self.num_per_class = num_per_class #1  2
        self.num_cl = num_cl # 5  
        self.num_inst = num_inst #1  2
        self.shuffle = shuffle
        # print('*'*60,'num_cl,num_inst',num_per_class,num_cl,num_inst)
        pass

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]]
                     for j in range(self.num_cl)]
        else:
            batch = [[i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]]
                     for j in range(self.num_cl)]
            pass

        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
            pass

        return iter(batch)

    def __len__(self):
        return 1

    pass


class ClassBalancedSamplerTest(Sampler):
    """ Samples 'num_inst' examples each from 'num_cl' pools of examples of size 'num_per_class' """

    def __init__(self, num_cl, num_inst, shuffle=True):
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle
        pass

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batches = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)] for j in range(self.num_cl)]
        else:
            batches = [[i + j * self.num_inst for i in range(self.num_inst)] for j in range(self.num_cl)]
            pass

        batches = [[batches[j][i] for j in range(self.num_cl)] for i in range(self.num_inst)]

        if self.shuffle:
            random.shuffle(batches)
            for sublist in batches:
                random.shuffle(sublist)
                pass
            pass

        batches = [item for sublist in batches for item in sublist]
        # print('*'*60,batches)
        return iter(batches)

    def __len__(self):
        return 1

    pass


class DatasetTask(object):

    def __init__(self, character_folders, num_classes, train_num, test_num):
        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num
        # print(character_folders)
        # print('*'*60,len(self.character_folders), self.num_classes)
        class_folders = random.sample(self.character_folders, self.num_classes)
        labels = dict(zip(class_folders, np.array(range(len(class_folders)))))

        samples = dict()
        self.train_roots = []
        self.test_roots = []
        for c in class_folders:
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])

            self.train_roots += samples[c][:train_num]
            self.test_roots += samples[c][train_num:train_num + test_num]
            pass

        self.train_labels = [labels[os.path.split(x)[0]] for x in self.train_roots]
        self.test_labels = [labels[os.path.split(x)[0]] for x in self.test_roots]

        # if len(self.train_roots) == 0:
        #     print()
        pass

    pass


# class myDataset(Dataset):

#     def __init__(self, task, transform=None, target_transform=None,split='',Config=''):
#         self.task = task
#         self.split = split
#         self.transform = transform
#         self.target_transform = target_transform
#         self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels
#         self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
#         self.Config=Config
#         pass

#     def __len__(self):
#         return len(self.image_roots)

#     def __getitem__(self, idx):
#         # print('*'*60,len(self.image_roots),idx)
#         # try:
#         image = Image.open(self.image_roots[idx]).convert(self.Config.convert)
#         # except Exception:
#         #     print()

#         # print('*'*60,type(image))
#         if self.transform is not None:
#             # if 
#             image = self.transform(image)

#         label = self.labels[idx]
#         if self.target_transform is not None:
#             label = self.target_transform(label)
#         return image, label

#     @staticmethod
#     def folders(data_root):
#         train_folder = os.path.join(data_root, "train")
#         val_folder = os.path.join(data_root, "val")
#         test_folder = os.path.join(data_root, "test")

#         folders_train = [os.path.join(train_folder, label) for label in os.listdir(train_folder)
#                          if os.path.isdir(os.path.join(train_folder, label))]
#         folders_val = [os.path.join(val_folder, label) for label in os.listdir(val_folder)
#                        if os.path.isdir(os.path.join(val_folder, label))]
#         folders_test = [os.path.join(test_folder, label) for label in os.listdir(test_folder)
#                         if os.path.isdir(os.path.join(test_folder, label))]

#         random.seed(1)
#         random.shuffle(folders_train)
#         random.shuffle(folders_val)
#         random.shuffle(folders_test)
#         return folders_train, folders_val, folders_test

#     @staticmethod
#     def get_data_loader(self,num_per_class=1,sampler_test=False, shuffle=False):
#         #     def get_data_loader(self,task, num_per_class=1, split='train', sampler_test=False, shuffle=False, transform=None):
#         if self.split  == 'train':
#             sampler = ClassBalancedSampler(num_per_class, self.task.num_classes, self.task.train_num, shuffle=shuffle)
#         else:
#             if not sampler_test:
#                 sampler = ClassBalancedSampler(num_per_class, self.task.num_classes, self.task.test_num, shuffle=shuffle)
#             else:  # test
#                 sampler = ClassBalancedSamplerTest(self.task.num_classes, self.task.test_num, shuffle=shuffle)
#                 pass
#             pass

#         assert self.transform,'transform is must'
#             # normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
#             # transform = transforms.Compose([transforms.ToTensor(), normalize])
#             # pass
#             #    def __init__(self, task, transform=None, target_transform=None,split='',Config=''):
#         dataset = myDataset(self.task, split=self.split, transform=self.transform,Config=self.Config)
#         return DataLoader(dataset, batch_size=num_per_class * self.task.num_classes, sampler=sampler)

#     pass
class myDataset(Dataset): 

    def __init__(self, task, split='train', transform=None, target_transform=None,Config=''):
        self.task = task
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.Config=Config
        pass

    def __len__(self):#len(myDataset())
        return len(self.image_roots)

    def __getitem__(self, idx):
        image = Image.open(self.image_roots[idx]).convert(self.Config.convert)
        # print('^'*60)
        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

    @staticmethod
    def folders(data_root):
        train_folder = os.path.join(data_root, "train")
        val_folder = os.path.join(data_root, "val")
        test_folder = os.path.join(data_root, "test")

        folders_train = [os.path.join(train_folder, label) for label in os.listdir(train_folder)
                         if os.path.isdir(os.path.join(train_folder, label))]
        folders_val = [os.path.join(val_folder, label) for label in os.listdir(val_folder)
                       if os.path.isdir(os.path.join(val_folder, label))]
        folders_test = [os.path.join(test_folder, label) for label in os.listdir(test_folder)
                        if os.path.isdir(os.path.join(test_folder, label))]

        random.seed(1)
        random.shuffle(folders_train)
        random.shuffle(folders_val)
        random.shuffle(folders_test)
        return folders_train, folders_val, folders_test

    @staticmethod
    def get_data_loader(task, num_per_class=1, split='train', sampler_test=False, shuffle=False, transform=None,Config=''):
        dataset = myDataset(task, split=split, transform=transform,Config=Config)
        if split == 'train':
            sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num, shuffle=shuffle)
        else:
            if not sampler_test:
                sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num, shuffle=shuffle)
            else:  # test
                sampler = ClassBalancedSamplerTest(task.num_classes, task.test_num, shuffle=shuffle)
                pass
            pass
        assert transform,'transform is must'

        return DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)

    pass


##############################################################################################################
def compute_confidence_interval(data):
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

class FSLTestTool(object):
    
    def __init__(self, model_fn, data_root, num_way=5, num_shot=1, episode_size=15, test_episode=600, transform=None,Config=''):
        self.model_fn = model_fn
        self.transform = transform


        self.folders_train, self.folders_val, self.folders_test = myDataset.folders(data_root)

        self.test_episode = test_episode
        self.num_way = num_way
        self.num_shot = num_shot
        self.episode_size = episode_size
        self.Config=Config
        self.Config.logger.info(f'transform_test:{self.transform}')
        pass

    def val_train(self):
        self.Config.logger.info('val_train---------eval train dataset')
        return self._val(self.folders_train, sampler_test=False, all_episode=self.test_episode)

    def val_val(self):
        self.Config.logger.info('val_val---------eval val dataset')
        return self._val(self.folders_val, sampler_test=False, all_episode=self.test_episode)

    def val_test(self):
        self.Config.logger.info('val_test---------eval test dataset')
        return self._val(self.folders_test, sampler_test=False, all_episode=self.test_episode)

    def val_test2(self):
        self.Config.logger.info('val_test---------eval test dataset again')
        return self._val(self.folders_test, sampler_test=True, all_episode=self.test_episode)

    def test(self, test_avg_num, epoch=0, is_print=True):
        acc_list = []
        test_pm_list=[]
        # acc_list=np.zeros(test_avg_num)
        # test_pm_list=np.zeros(test_avg_num)

        for _ in range(test_avg_num):
            self.Config.logger.info('test-----true test')
            acc,test_pm= self._val(self.folders_test, sampler_test=True, all_episode=self.test_episode)
            self.Config.logger.info("epoch={}, Test accuracy={} ± {}".format(epoch, acc,test_pm))
            # acc_list[_]=acc
            # test_pm_list[_]=test_pm
            acc_list.append(acc)
            test_pm_list.append(test_pm)
        mean_acc =np.mean(np.array(acc_list))
        mean_pm=np.mean(np.array(test_pm_list))

        self.Config.logger.info("epoch={}, Mean Test accuracy={}".format(epoch, mean_acc,mean_pm))

        return mean_acc,mean_pm

    def val(self, epoch=0, is_print=True,save_based_test=False):
        acc_train,train_pm= self.val_train()
        acc_val,val_pm= self.val_val()
        acc_test1,test1_pm = self.val_test()
        acc_test2,test2_pm  = self.val_test2()

        if is_print:
            self.Config.logger.info("fsl_Train {} Accuracy: {}±{}".format(epoch, acc_train,train_pm))
            self.Config.logger.info("fsl_Val   {} Accuracy: {}±{}".format(epoch, acc_val,val_pm))
            self.Config.logger.info("fsl_Test1 {} Accuracy: {}±{}".format(epoch, acc_test1,test1_pm))
            self.Config.logger.info("fsl_Test2 {} Accuracy: {}±{}".format(epoch, acc_test2,test2_pm))
            self.Config.writer.add_scalars('fsl_acc', {'fsl_train_acc': acc_train,
                            'fsl_val_acc': acc_val,
                            'fsl_test1_acc':  acc_test1,
                            'fsl_test2_acc':  acc_test2}, epoch)
            pass
        return acc_val if save_based_test else (acc_test1+acc_test2)/2
        # if save_based_test:
        #     return acc_val
            
        # else: 
        #     return (acc_test1+acc_test2)/2

    def _val(self, folders, sampler_test, all_episode):
        accuracies = []
        # pbar = tqdm(total=all_episode, ncols=70)
        # print(self.num_way, self.num_shot)
        for i in range(all_episode):
            total_rewards = 0
            counter = 0
            #len(folders)=64 
            # 随机选5类，每类中取出1个作为训练样本，每类取出15个作为测试样本 每次20个样本len(folders)=20---5 way 1 shot 15 query 
            #5way 15query
            #train (num_classes)5*(train_num)2=10 query:(num_classes)5*(test_num)15=75
            #  def __init__(self, character_folders, num_classes, train_num, test_num):
            task = DatasetTask(folders, self.num_way, self.num_shot, self.episode_size)
            #每个情景  查询集里每一类num_per_class有多少个图像
            # sample_data_loader = myDataset(task,transform=self.transform,split="train",Config=self.Config).get_data_loader(
            #                                 num_per_class=1,sampler_test=sampler_test,shuffle=False)
            # batch_data_loader =myDataset(task,transform=self.transform,split="val",Config=self.Config).get_data_loader(
            #                                 num_per_class=3, sampler_test=sampler_test,shuffle=True)
            # num_per_class = 5 if self.num_shot> 1 else 3     #num_per_class=5  

            sample_data_loader = myDataset.get_data_loader(task, self.num_shot, "train", sampler_test=sampler_test,
                                                              shuffle=False, transform=self.transform,Config=self.Config)                                             
            batch_data_loader = myDataset.get_data_loader(task, 3, "val", sampler_test=sampler_test,
                                                             shuffle=True, transform=self.transform,Config=self.Config)
 
            samples, labels = sample_data_loader.__iter__().next()
            '''
            temp=list(batch_data_loader.__iter__()) or list(batch_data_loader)
            # samples:2*5=10*3*28*28  # 0011223344 5way 2shot sample_data_loader--batchsize and labels:10，sample_data_loader：2
            #tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
            #if len>1则next内容不一样并且这样写就更不一样，固定batch_data_loader也不行 add .next()=list#tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
            # for samples, labels  in sample_data_loader:#   5way*1shot=5task/(npc(numshot)=1*way=5)=1
            #     print(samples.shape) ##torch.Size([10, 3, 28, 28])  tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]) 
            #     print("*"*80,'len(sample_data_loader)',len(sample_data_loader.__iter__()))#1 \
            # print("*"*80,len(sample_data_loader),len(batch_data_loader))#1 1
            
            # print("out_batch_labels",type(batch_data_loader.__iter__()))
            # print("out_batch_labels",batch_data_loader.__iter__().next())
            # tensor([1, 3, 0, 2, 4, 3, 1, 0, 2, 4, 4, 1, 3, 0, 2, 4, 2, 0, 3, 1, 2, 1, 4, 3,0])
            '''
            samples = self.to_cuda(samples)
            print('#'*60,len(batch_data_loader),flush=True)
            for batches, batch_labels in batch_data_loader:#batches（5w1shot）3*5=15*3*32*32batches（5w2shot）5*5=25*3*32*32  #batch_labels batch_data_loader--batchsize 25  batch_data_loader：5
                # results = self.model_fn(samples, self.to_cuda(batches))#10support  match 25 query>>>results：25*5
                results = self.model_fn(samples, self.to_cuda(batches), num_way_test=self.num_way, num_shot=self.num_shot)
                '''
                query：每个batch5*5=25个task   3counter
                support：，日个batch 2*5=10个tast 
                print("batch_labels",batch_labels)#5way*15shot=75task/(npc=5*way=5)=3 5way*15shot=75task/(npc=3*way=5)=5
                '''
                _, predict_labels = torch.max(results.data, 1)
                batch_size = batch_labels.shape[0]
                print('^'*60,"batch_labels",batch_labels,flush=True)

                
                rewards = [1 if predict_labels[j].cpu() == batch_labels[j] else 0 for j in range(batch_size)]
                # batch_labels:tensor([0, 4, 1, 2, 3, 4, 2, 0, 3, 1, 1, 2, 4, 0, 3])
                # predi labels:tensor([4, 3, 3, 1, 2, 4, 3, 4, 1, 4, 4, 4, 4, 0, 0]
                total_rewards += np.sum(rewards)

                counter += batch_size
                pass
            accuracies.append(total_rewards / 1.0 / counter)#counter一般为75
        #     pbar.update(1)
        # pbar.close()
        #    return np.mean(np.array(accuracies, dtype=np.float)), 
        m,pm=compute_confidence_interval(accuracies)[0],compute_confidence_interval(accuracies)[1]
        return m,pm

    @staticmethod
    def to_cuda(x):
        return x.cuda() if torch.cuda.is_available() else x

    pass


if __name__ == '__main__':
    test_tool = FSLTestTool(model_fn=None, data_root=None, num_way=5, num_shot=1, episode_size=15, test_episode=600)
    _acc = test_tool.val(epoch=0, is_print=True)
    test_tool.test(5, epoch=0, is_print=True)
    pass
