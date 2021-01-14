11111111111111111111111111111111111111111111
#

[Few-Shot Classification Leaderboard](https://few-shot.yyliu.net/miniimagenet.html)
test-pull
pip install -r requirements   -i https://pypi.mirrors.ustc.edu.cn/simple/ 

首先不能覆盖上次提交的

看看是否会被先继承再更新覆盖
pip install -r requirements.txt   -i https://pypi.mirrors.ustc.edu.cn/simple/ 



先继承再更新 

先pull再push
  所以在改变之前先pull,并且不在其他服务器上改变,避免merge

  是不是所有的fsl数据增强都要变化，加上色彩和灰度
  采用提交格式
pull 有冲突时，会先拉下来，而git改变区域则为询问是否保留本地改变，commit push即和提交合并本地改变
12.25

  in 1-1080  xxxxi11080
  python mn_FC100_fsl_sgd_modify_tensorboard.py -bs 128 -r12 -te 600  finished
  python mn_CIFARFS_fsl_sgd_modify_tensorboard.py -bs 128 -r12 -te 600 WILL   12.26 跑  实验后期提高分数 going
  in 2-1080  i21080
  实验1
python mn_omniglot_two_ic_ufsl_2net_res_sgd_acc_duli.py -bs 128   going finished
  实验2
python mn_omniglot_fsl_sgd_modify.py -bs 64 -g 01 -w 20 -s 1     going    finished

  out 2-1080  xxx --out

  实验1
python mn_omniglot_fsl_sgd_modify.py -bs 128 -g 01 -w 5 -s 1  finished
  实验2 
python  mn_omniglot_fsl_sgd_modify_dataAUG.py -bs 128 -g 01 -w 5 -s 1   finished


目前有监督fsl不可以测试20wayoneshot，因为训练测试为一致的类别，测试时的类别始终为5way即可。
所以 如果改变 num_way！=5，那么必须有num_waytest=5，
无监督 跳过，所有的不改变即可，调用师兄PNET程序
有监督20way只为对比
aaa

12.26
out
python  mn_omniglot_fsl_sgd_modify_dataAUG.py -bs 128 -g 01 -w 5 -s 1 lr 3 finished
python  mn_omniglot_fsl_sgd_modify_dataAUG.py -bs 128 -g 01 -w 5 -s 1 wegit init   finished  
python  mn_omniglot_fsl_sgd_modify_dataAUG.py -bs 64  -w 5 -s 1 -g 1 -te 1000 --lr 3    normal  finished
python  pn_omniglot_fsl_sgd.py -bs 64  -w 5 -s 1 -te 1000 --lr 2   finished
python  pn_omniglot_fsl_sgd.py -bs 64  -w 5 -s 1 -te 1000 --lr 3    finished
in1

in2



12.27
  out
python  mn_omniglot_fsl_sgd_modify_dataAUG.py -bs 128 -g 1 -w 5 -s 1 --lr 3 finished
python  mn_omniglot_fsl_sgd_modify_dataAUG.py -bs 128 -g 0 -w 5 -s 1 --lr 3 -fb mnc4    convert  finished
python  mn_omniglot_fsl_sgd_modify_dataAUG.py -bs 128 -g 1 -w 5 -s 1 --lr 3 -fb rnc4   finished

python  mn_omniglot_fsl_sgd_modify_dataAUG.py -bs 128 -g 0 -w 5 -s 1 --lr 3 -fb mnc4  -te 1000  convert  finished
python  mn_omniglot_fsl_sgd_modify_dataAUG.py -bs 128 -g 1 -w 5 -s 1 --lr 3 -fb rnc4  -te 1000 finished

python  mn_omniglot_fsl_sgd_modify_dataAUG.py -bs 128  -w 5 -s 1 --lr 3 -res12  -te 1000 finished
python  mn_omniglot_fsl_sgd_modify.py -bs 128  -w 5 -s 1 --lr 3 -res12  -te 1000  finished  
in 2
python mn_omniglot_two_ic_ufsl_2net_res_sgd_acc_duli.py  -g 01 -bs 128 wrong finished


in1
 python mn_CIFARFS_fsl_sgd_modify.py -bs 128 -r12 -te 600  finished


12.29
out
python mn_omniglot_fsl_sgd_modify_dataAUG.py -bs 128 -g 1 -w 5 -s 1 --lr 3 -fb c4  
python mn_FC100_two_ic_ufsl_2net_res_sgd_acc_duli_tensorboard.py   going do  
in1
python mn_FC100_fsl_sgd_modify_tensorboard.py -bs 128 -fb res12 -te 600  --lr 3  going   do  debug --
in2
python mn_omniglot_two_ic_ufsl_2net_res_sgd_acc_duli.py  -g 0 -bs 128 
python mn_omniglot_two_ic_ufsl_2net_res_sgd_acc_duli.py  -g 0 -bs 128 -fb rnc4
y2 market
假如因为忘记pull不能push(已经 缓存),那就补上pull就可以再push 最后会merge


sync
本地删除 远端删除后  不能同步

远端删除 本地不能同步，且远端不能编辑

不能只同步一个文件

担心：远端生成文件同步到本地将本地覆盖
报错本地：ssss
远程：test 不能同步both,但可以local 到remote 且 远端不会删除本地没有的test
或者 remote 到local 且本地不会删除没有的ssss  一般不用 都是local 到remote
可以重命名但没有删除
"autoUpload": true 修改文件内容自动上传
"autoDelete": #删除文件创建文件自动上传而非文件夹
e3bm 文件夹也不能同步所以 乖乖用git
不能随便 sync LR 防止L把最新的权重覆盖
测试三个是否同时同步
不可以同步
代码用git
大文件同sync  

train
python mn_fsl_sgd_modify_dataAUG.py -ds CIFARFS --convert RGB  --lr 3 -fb res12 -g 01 -te 600 -dr 0.2

you
python mn_fsl_sgd_modify_dataAUG.py -ds CIFARFS --convert RGB  --lr 3 -fb rnc4 -g 01 -te 600 -dr 0.2

in2
python mn_fsl_sgd_modify_dataAUG.py -ds FC100 --convert RGB  --lr 3 -fb res12 -g 01 -te 600 -dr 0.2

in1
python mn_fsl_sgd_modify_dataAUG.py -ds FC100 --convert RGB  --lr 3 -fb rnc4 -g 01 -te 600 -dr 0.2



1.4
first
python mn_two_ic_ufsl_2net_res_sgd_acc_duli_eval.py -ds CIFARFS --convert RGB -fb res12 -dr 0.2 -g 01 
second
python mn_two_ic_ufsl_2net_res_sgd_acc_duli_eval.py -ds FC100 --convert RGB -fb c4 -dr 0.2 -g 01

eval
FC100-FSL

FC100-UFSL
**************res12
python mn_two_ic_ufsl_2net_res_sgd_acc_duli_eval.py -v /home/ubuntu/Documents/hzh/ActiveLearning/UFSLviaIC/my_MN/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli_FC100/eval-res12/1_2100_64_5_1_500_200_512_1_1.0_1.0_head_png_res12_ic_FC100.pkl -ds FC100 -fb res12 -dr 0.1 -g 0 -d
**************c4
python mn_two_ic_ufsl_2net_res_sgd_acc_duli_eval.py -v /home/ubuntu/Documents/hzh/ActiveLearning/UFSLviaIC/my_MN/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli_FC100/eval-c4/1_2100_64_5_1_500_200_512_1_1.0_1.0_head_png_ic_FC100.pkl -ds FC100 -fb c4 -dr 0.1 -g 1 -d

python mn_fsl_sgd_modify_dataAUG_eval.py -v /home/ubuntu/Documents/hzh/ActiveLearning/UFSLviaIC/my_MN/models_mn/fsl_sgd_modify_FC100/eval-res12_Jan03_13-09-12_EP600_BS64_ft200_100_mn_5w1s_DR0.2_res12_lr3_FC100_RGB/fsl-250EP600_BS64_ft200_100_mn_5w1s_DR0.2_res12_lr3_FC100_RGB.pkl -ds FC100 -fb res12 -dr 0.2 -g 01 --convert RGB

python mn_fsl_sgd_modify_dataAUG_eval.py -v /home/ubuntu/Documents/hzh/ActiveLearning/UFSLviaIC/my_MN/models_mn/fsl_sgd_modify_FC100/eval-res12_Dec23_11-42-14_EP400_BS128_mn_5way_1shot_DR0.3_res12_FC100_finished/'fsl-(150)EP400_BS128_mn_5way_1shot_DR0.3_res12_FC100.pkl' -ds FC100 -fb res12 -dr 0.2 -g 01 -d --convert RGB



#############################CIFARFS-FSL
python mn_fsl_sgd_modify_dataAUG_eval.py -ds CIFARFS --convert RGB -fb res12 -dr 0.2 \
  -v /home/ubuntu/Documents/hzh/ActiveLearning/UFSLviaIC/my_MN/models_mn/fsl_sgd_modify_CIFARFS/Jan03_00-40-32_EP600_BS64_ft200_100_mn_5w1s_DR0.2_res12_lr3_CIFARFS_RGB/fsl-210EP600_BS64_ft200_100_mn_5w1s_DR0.2_res12_lr3_CIFARFS_RGB.pkl -d -g 01
#############################CIFARFS-UFSL
***************c4
python mn_two_ic_ufsl_2net_res_sgd_acc_duli_eval.py -ds CIFARFS --convert RGB -fb c4\
  -v /home/ubuntu/Documents/hzh/ActiveLearning/UFSLviaIC/my_MN/models_mn/old_ic_CIFARFS/1_2100_64_5_1_500_200_512_1_1.0_1.0_head_png_ic_CIFARFS/1_2100_64_5_1_500_200_512_1_1.0_1.0_head_png_ic_CIFARFS.pkl  -g 1
***************res12
python mn_two_ic_ufsl_2net_res_sgd_acc_duli_eval.py -ds CIFARFS --convert RGB -fb res12\
  -v /home/ubuntu/Documents/hzh/ActiveLearning/UFSLviaIC/my_MN/models_mn/old_ic_CIFARFS/1_2100_64_5_1_500_200_512_1_1.0_1.0_head_png_res12_ic_CIFARFS/1_2100_64_5_1_500_200_512_1_1.0_1.0_head_png_res12_mn_5way_1shot_CIFARFS.pkl  -g 0 


omniglot-FSL
python mn_fsl_sgd_modify_dataAUG_eval.py 
omniglot-UFSL
python mn_two_ic_ufsl_2net_res_sgd_acc_duli_eval.py

python mn_fsl_sgd_modify_dataAUG_eval.py -v /home/ubuntu/Documents/hzh/ActiveLearning/UFSLviaIC/my_MN/models_mn/fsl_sgd_modify_CIFARFS/eval-Dec27_00-25-18_EP600_BS128_ft200_100_mn_5w1s_DR0.1_res12_CIFARFS/fsl-220EP600_BS128_ft200_100_mn_5w1s_DR0.1_res12_CIFARFS.pkl -ds CIFARFS -dr 0.1 -fb res12 -g 0

############################in2---FC100   ["random", "css", "cluster"]

python mn_ufsl_random_and_css.py  -ds FC100 -bt 0 -fb c4

python mn_ufsl_random_and_css.py  -ds FC100 -bt 1 -fb c4

python mn_ufsl_random_and_css.py  -ds FC100 -bt 2 -fb c4

python mn_ufsl_random_and_css.py  -ds FC100 -bt 0 -fb res12

python mn_ufsl_random_and_css.py  -ds FC100 -bt 1 -fb res12

python mn_ufsl_random_and_css.py  -ds FC100 -bt 2 -fb res12

############################out---CIFARFS
python mn_ufsl_random_and_css.py  -ds CIFARFS -bt 0 -fb res12 -g 0

python mn_ufsl_random_and_css.py  -ds CIFARFS -bt 1 -fb res12 -g 1 

python mn_ufsl_random_and_css.py  -ds CIFARFS -bt 2 -fb res12 -g 1

python mn_ufsl_random_and_css.py  -ds CIFARFS -bt 0 -fb c4 -g 01

git pull test

python mn_two_ic_ufsl_2net_res_sgd_acc_duli_eval.py --convert RGB -bs 64 --lr 3 -fb res12N 



