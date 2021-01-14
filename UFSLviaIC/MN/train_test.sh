echo "CIFARFS--UFSL"  
python mn_CIFARFS_two_ic_ufsl_2net_res_sgd_acc_duli.py -d  -te 1
echo "CIFARFS--FSL"  
python mn_CIFARFS_fsl_sgd_modify.py -d -te 2
echo "FC100--FSL" 
python mn_FC100_fsl_sgd_modify_tensorboard.py -d -te 1
echo "FC100--UFSL"  
python mn_FC100_two_ic_ufsl_2net_res_sgd_acc_duli_tensorboard.py -d -te 1
echo "omniglot--FSL"
python mn_omniglot_fsl_sgd_modify_dataAUG.py -d -te 1
echo "omniglot--UFSL"
python mn_omniglot_two_ic_ufsl_2net_res_sgd_acc_duli.py -d  -te 1