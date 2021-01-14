import os
import time
from glob import glob
import matplotlib.pyplot as plt  # matplotlib.colors.BASE_COLORS
from alisuretool.Tools import Tools
# 2021-01-06 19:59:14 way=12,shot=1,acc=0.1589111111111111,con=0.0021892133181342327

# txt_path = "/mnt/4T/ALISURE/ActiveLearning/UFSLviaIC/models_abl/ic_res_xx"
dataset_name = "FC100"
result_dir = "result"

txt_path = "/home/ubuntu/Documents/hzh/ActiveLearning/UFSLviaIC/models_abl/{}/mn/{}".format(dataset_name, result_dir)
all_txt = glob(os.path.join(txt_path, "test*.txt"))
all_txt_name = []
all_txt_content = []
for txt in all_txt:
    with open(txt) as file:
        txt_content = file.readlines()
        all_txt_content.append(txt_content)
        pass
    # txt_name_list = os.path.basename(txt).split("_")[2:]
    txt_name_list = os.path.basename(txt)[:-4].split("_")
    all_txt_name.append(txt_name_list)
    pass


acc_result_dict = {}
for index, txt_content in enumerate(all_txt_content):
    txt_name = all_txt_name[index]
    method,backbone=txt_name[1],txt_name[1]
    

    # time_list = [time.strptime(txt.split(" E")[0], "%Y-%m-%d %H:%M:%S")
    #              for txt in txt_content if "Epoch" in txt and "ic_lr" in txt][100: 110]
    # now_time = (time.mktime(time_list[-1]) - time.mktime(time_list[0])) // (len(time_list) - 1)
    acc_dict = {}
    #2021-01-06 19:59:14 way=12,shot=1,acc=0.1589111111111111,con=0.0021892133181342327
    for split in ["Train", "Val", "Test"]:
        txt_acc = [txt.split(" ")[-2].split("/") for txt in txt_content if "Epoch" in txt and split in txt]
        txt_acc = [[float(acc[0]), float(acc[1])] for acc in txt_acc]
        top_1, top_5 = txt_acc[-1]
        acc_dict[split] = [top_1, top_5]
        pass

    if txt_name[0] == "vgg16":
        d = int(txt_name[3])
        name = "VGG-16"
    elif txt_name[0] == "conv4":
        d = int(txt_name[2])
        name = "Conv-4"
    else:
        d = int(txt_name[2])
        if txt_name[0] == "res50":
            name = "ResNet-50"
        elif txt_name[0] == "res34":
            name = "ResNet-34"
        elif txt_name[0] == "res18":
            name = "ResNet-18"
        else:
            raise Exception("")
        name = name + head
        pass

    if name not in acc_result_dict:
        acc_result_dict[name] = {}
    acc_result_dict[name][d] = {"acc": acc_dict}
    # acc_result_dict[name][d] = {"acc": acc_dict, "time": now_time}
    pass


# Conv-4 & 32.02 & 76.5 & 32.5 & 76.5 & 32.5 & 76.5 & 128 \\
dim = 512
result_str = []
for acc_dict_key in ["Conv-4", "ResNet-18", "ResNet-34", "ResNet-50", "VGG-16"]:
    acc_dict = acc_result_dict[acc_dict_key][dim]
    acc = acc_dict["acc"]
    time = acc_dict["time"]
    now_str = "{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {} \\\\".format(
        acc_dict_key, acc["Train"][0] * 100, acc["Train"][1] * 100,
        acc["Val"][0] * 100, acc["Val"][1] * 100, acc["Test"][0] * 100, acc["Test"][1] * 100, int(time))
    result_str.append(now_str)
    pass

for i in result_str:
    print(i)

Tools.print()
