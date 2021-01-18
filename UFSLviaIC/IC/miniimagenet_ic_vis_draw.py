import os
from glob import glob
from PIL import Image
from collections import Counter
import time
import socket
from datetime import datetime

class Config(object):
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')# or 

    split = "all"
    # split = "train"
    # split = "val"
    # split = "test"
    is_all = split == "all"

    ################################################################################
    # ic_id_list = [4, 5, 7, 28, 29, 8, 41, 26, 12, 19, 45, 46, 56, 64, 65]
    # image_num_train = 10
    # image_num_val = 5
    # image_num_test = 5
    ################################################################################

    ################################################################################
    # ic_id_list = [4, 5, 28, 29, 8, 41, 26, 12, 19, 45, 46, 56, 65]
    # 29 ,19, 45,41, 26,12
    # ic_id_list = [6,17,103,142,177,618,
    #             522,238,590,1242,619,717,743,
    #             685,403,479,538,424,434,780,
    #               94,592,703,394,703,76,245,247]
    # ic_id_list=list(range(512))
    ic_id_list = [155,39,44,54,56,77,80,84,105,189,188,79,71,290,163,166,177,209,450,503,210,122,267,379,300,380,504,505]
    image_num_train = 10
    image_num_val = 5
    image_num_test = 5
    #
    # vis_root = f"/home/ubuntu/Documents/hzh/ActiveLearning/UFSLviaIC/IC/IC_result/123_res34_head_1200_384_2048_conv4_100_5_1_288_ic_tiered"
    vis_root = f'/home/ubuntu/Documents/hzh/ActiveLearning/UFSLviaIC/IC/IC_result/1_2100_64_5_1_500_200_512_1_1.0_1.0_head_png_res12_ic_CIFARFS'
    # {image_num_train}_{image_num_val}_{image_num_test}{current_time}
    ################################################################################

    image_size = 32
    margin_image = 2
    margin_split = 16

    # image_size = 84
    # margin_image = 4
    # margin_split = 32
    sort = True
    # sort = False

    if is_all:
        image_num_list = [image_num_train, image_num_val, image_num_test]
        image_num = sum(image_num_list)
        vis_ic_path_list = [os.path.join(vis_root, "train"),
                            os.path.join(vis_root, "val"),
                            os.path.join(vis_root, "test")]
        a = image_size * image_num_train + (image_num_train - 1) * margin_image
        b = image_size * image_num_val + (image_num_val - 1) * margin_image
        c = image_size * image_num_test + (image_num_test - 1) * margin_image
        start_list = [0, a + margin_split, a + margin_split + b + margin_split]
        result_size = (a + margin_split + b + margin_split + c,
                       len(ic_id_list) * image_size + (len(ic_id_list) - 1) * margin_image)
    elif split == "train":
        image_num = image_num_train
        vis_ic_path = os.path.join(vis_root, split)
        result_size = (image_size * image_num_train + (image_num_train - 1) * margin_image,
                       len(ic_id_list) * image_size + (len(ic_id_list) - 1) * margin_image)
    elif split == "val":
        image_num = image_num_val
        vis_ic_path = os.path.join(vis_root, split)
        result_size = (image_size * image_num_val + (image_num_val - 1) * margin_image,
                       len(ic_id_list) * image_size + (len(ic_id_list) - 1) * margin_image)
    elif split == "test":
        image_num = image_num_test
        vis_ic_path = os.path.join(vis_root, split)
        result_size = (image_size * image_num_test + (image_num_test - 1) * margin_image,
                       len(ic_id_list) * image_size + (len(ic_id_list) - 1) * margin_image)
    else:
        raise Exception(".......")

    result_path = os.path.join(vis_root, "{}_{}_{}_{}_{}_{}.png".format(split, image_size, image_num, margin_image,len(ic_id_list),current_time))
    pass


def get_image_y_split(vis_ic_path, ic_id, image_num):
    ic_image_file = glob(os.path.join(vis_ic_path, str(ic_id), "*.png"))

    if Config.sort:
        ic_class = [os.path.basename(ic).split("_")[0] for ic in ic_image_file]
        ic_count_sorted = sorted(Counter(ic_class).items(), key=lambda x: x[1], reverse=True)
        ic_image_file_sorted = []
        for ic_count_one in ic_count_sorted:
            ic_image_file_sorted.extend(glob(os.path.join(vis_ic_path, str(ic_id), "{}_*.png".format(ic_count_one[0]))))
            pass
        ic_image_file = ic_image_file_sorted
        # ic_image_file=sorted(ic_image_file)
        pass

    im_list_result = [Image.open(image_file) for image_file in ic_image_file[:image_num]]
    return im_list_result


def get_image_by_id(ic_id):
    if Config.is_all:
        im_list_result = []
        for vis_ic_path, image_num in zip(Config.vis_ic_path_list, Config.image_num_list):
            im_list_result.append(get_image_y_split(vis_ic_path, ic_id=ic_id, image_num=image_num))
            pass
    else:
        im_list_result = get_image_y_split(Config.vis_ic_path, ic_id=ic_id, image_num=Config.image_num)
        pass
    return im_list_result


# if __name__ == '__main__':
#     im_result = Image.new("RGB", size=Config.result_size, color=(255, 255, 255))
#     for i in range(len(Config.ic_id_list)):
#         im_list_list = get_image_by_id(Config.ic_id_list[i])
#         if Config.is_all:
#             for split_id in range(len(im_list_list)):
#                 im_list = im_list_list[split_id]
#                 for j in range(len(im_list)):
#                     im_result.paste(im_list[j].resize((Config.image_size, Config.image_size)),
#                                     box=(Config.start_list[split_id] + j * (Config.image_size + Config.margin_image),
#                                          i * (Config.image_size + Config.margin_image)))
#                 pass
#         else:
#             im_list = im_list_list
#             for j in range(len(im_list)):
#                 im_result.paste(im_list[j].resize((Config.image_size, Config.image_size)),
#                                 box=(j * (Config.image_size + Config.margin_image),
#                                      i * (Config.image_size + Config.margin_image)))
#             pass
#         pass
#     im_result.save(Config.result_path)
#     pass
if __name__ == '__main__':
    from PIL import ImageFont

    from PIL import ImageDraw

    im_result = Image.new("RGB", size=Config.result_size, color=(255, 255, 255))
    for i in range(len(Config.ic_id_list)):
        im_list_list = get_image_by_id(Config.ic_id_list[i])
        if Config.is_all:
            for split_id in range(len(im_list_list)):
                im_list = im_list_list[split_id]
                for j in range(len(im_list)):
                    im_result.paste(im_list[j].resize((Config.image_size, Config.image_size)),
                                    box=(Config.start_list[split_id] + j * (Config.image_size + Config.margin_image),
                                         i * (Config.image_size + Config.margin_image)))
                pass
        # else:
        #     im_list = im_list_list
        #     for j in range(len(im_list)):
        #         im_result.paste(im_list[j].resize((Config.image_size, Config.image_size)),
        #                         box=(j * (Config.image_size + Config.margin_image),
        #                              i * (Config.image_size + Config.margin_image)))
        #     pass
        # pass
        # font = ImageFont.truetype( '/usr/share/fonts/truetype/ubuntu/Ubuntu-RI.ttf',10)#设置字体
        # draw = ImageDraw.Draw(im_result)
        # draw.text((10,(Config.image_size+Config.margin_image)*i), f'{i}',font=font)
        # draw.text((10,(Config.image_size+Config.margin_image)*i), f'{Config.ic_id_list[i]}',font=font)

    im_result.save(Config.result_path)
    pass