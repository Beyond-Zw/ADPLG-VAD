import options
class Config(object):
    def __init__(self, args):
        self.ID = args.ID
        self.gpus = args.gpus
        self.root_dir = args.root_dir
        self.modal = args.modal
        self.lr_v = eval(args.lr_v)
        self.num_iters = len(self.lr_v)
        self.lr_s = eval(args.lr_s)
        self.num_iters = len(self.lr_s)
        self.sch_step_s = args.sch_step_s
        self.sch_gamma_s = args.sch_gamma_s
        self.len_feature = args.len_feature
        self.batch_size = args.batch_size
        self.model_path = args.model_path
        self.model_file = args.model_file
        self.num_workers = args.num_workers
        self.seed = args.seed
        self.num_segments = args.num_segments
        self.ls_s_w = args.ls_s_w
        self.ls_v_w = args.ls_v_w
        self.ls_tri_w = args.ls_tri_w
        self.ls_cos_w = args.ls_cos_w
        self.ls_sp_w = args.ls_sp_w
        self.ls_sm_w = args.ls_sm_w
        self.warm_up = args.warm_up
        self.thre_pesudo_a = args.thre_pesudo_a
        self.layer_ts = args.layer_ts
        self.law1 = args.law1
        self.law2 = args.law2
        self.law3 = args.law3
        self.law4 = args.law4
        self.thre_var_a = args.thre_var_a
        self.pl_his_num = args.pl_his_num
        self.warm_up = args.warm_up





if __name__ == "__main__":
    args=options.parse_args()
    conf=Config(args)
    print(conf.lr)  

