def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--drop_rate', '-dr',type=float, default=0.2, help=' drop_rate')
    parser.add_argument('--gpu', '-g',type=str, default='0', help=' gpu')
    parser.add_argument('--batch_size', '-bs',type=int, default=64, help=' batchsize')
    parser.add_argument('--train_epoch', '-te',type=int, default=400, help='train_epoch')
    parser.add_argument('--debug','-d',  action='store_true', help=' debug')
    parser.add_argument('--res12', '-r12',action='store_true', help='fsl_backbone is res12')
    parser.add_argument('--num_way', '-w',type=int, default=5, help=' num_way=5')
    parser.add_argument('--num_shot', '-s',type=int, default=1, help=' num_shot=1')
    parser.add_argument('--val', '-v',type=str, default='',help=' only val wegit _dir')
    parser.add_argument('--lr',type=int, default=2,help=' lr function id')  
      
    args = parser.parse_args()
    return args

class Config(object):
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    batch_size = args.batch_size
    train_epoch = args.train_epoch
    num_way =args.num_way
    num_shot = args.num_shot
    num_workers = 8
    learning_rate = 0.01
    num_way_test = 5
    # num_shot_test = 0
    val_freq=1 if args.debug else 10
    episode_size = 15#MiniImageNetTask
    # episode_size = 100#my
    test_episode = 600#600代
    first_epoch, t_epoch = 200, 100
    if args.lr==1:
        adjust_learning_rate = RunnerTool.adjust_learning_rate1
    elif args.lr==2:
        adjust_learning_rate = RunnerTool.adjust_learning_rate2
    elif args.lr==3:
        adjust_learning_rate = RunnerTool.adjust_learning_rate3
    hid_dim = 64
    z_dim = 64
    is_png = True
    # is_png = False
    ###############################################################################################
    drop_rate=args.drop_rate
    dataset='omniglot_single'
    if  args.res12:
        fsl_backbone='res12'
        matching_net = ResNet12Small(avg_pool=True, drop_rate=drop_rate,inplanes=len(args.convert))
        commit=f'{num_way}w{num_shot}s_DR{drop_rate}_{fsl_backbone}_lr{args.lr}_{dataset}_{convert}'
    else:
        matching_net = MatchingNet(hid_dim=hid_dim, z_dim=z_dim)#fsl conv4->res12
        fsl_backbone='c4'
        commit=f'{num_way}w{num_shot}s_{fsl_backbone}_lr{args.lr}_{dataset}_weights_init_normal'
    DataParallel=True if len(args.gpu)>=2 else False

    ###############################################################################################
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')# or 
    model_name =f'EP{train_epoch}_BS{batch_size}_ft{first_epoch}_{t_epoch}_mn_{commit}'


    data_root = f'/home/ubuntu/Dataset/Partition1/hzh/data/{dataset}'
    if not os.path.exists(data_root):
        data_root = f'/home/test/Documents/hzh/ActiveLearning/data/{dataset}'
    
    _root_path = "../models_mn/fsl_sgd_modify"
    # _root_path = "../models_rn/two_ic_ufsl_2net_res_sgd_acc_duli"
################################################################################################down is same
    if not args.debug:
        debug=""
        for i in glob.glob(_root_path+'/debug*'):
            shutil.rmtree(i)
            print(f'delete {i}')
    else:
            debug="debug"
            print(f'you are debugging ')
    date_dir=f'{_root_path}/{debug+current_time}_{model_name}'
    mn_dir=Tools.new_dir(f"{date_dir}/{model_name}.pkl") if not args.val else args.val
    writer = SummaryWriter(date_dir+'/runs')

    shutil.copy(os.path.abspath(sys.argv[0]),date_dir)

    log_file = os.path.join(date_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file,name=f"mn-{commit}")
    logger.info(model_name)
    logger.info(f'DataParallel is {DataParallel}')
    logger.info(f"platform.platform{platform.platform()}")
    logger.info(f"config:   ")
    logger.info(f"args.gpu :{args.gpu} ,is_png:   {is_png},num_way_test:   {num_way_test}, test_episode:   {test_episode}") 
    logger.info(f"first_epoch:   {first_epoch},t_epoch:   {t_epoch}, val_freq:   {val_freq},episode_size:   {episode_size}")
    logger.info(f'hid_dim:   {hid_dim},z_dim:   {z_dim} , is_png:   {is_png},input_dim: {input_dim}')
    ABSPATH=os.path.abspath(sys.argv[0])

    pass