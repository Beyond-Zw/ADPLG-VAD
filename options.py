import argparse
from random import seed
import os

def parse_args():
    descript = 'Pytorch Implementation of ADPLG-VAD'
    parser = argparse.ArgumentParser(description = descript)
    parser.add_argument('--root_dir', type=str, default='outputs/')
    parser.add_argument('--modal', type = str, default = 'RGB',choices = ["RGB,RGB+audio"])
    parser.add_argument('--model_path', type = str, default = 'outputs/models/')
    parser.add_argument('--model_file', type=str, default="",help='the path of pre-trained model file')
    parser.add_argument('--lr_v', type = str, default = '[0.0001]*20000', help = 'learning rates for steps(list form)')
    parser.add_argument('--lr_s', type=str, default='[0.001]*20000', help='learning rates for steps(list form)')
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--num_workers', type = int, default = 8)
    parser.add_argument('--num_segments', type = int, default = 64)
    parser.add_argument('--len_feature', type=int, default = 1024, help = 'length of feature vector, 1152 for RGB+audio')
    parser.add_argument('--seed', type = int, default = 42, help = 'random seed (-1 for no manual seed)')
    parser.add_argument('--ls_s_w', type=float, default=1, help='ls_i_w')
    parser.add_argument('--ls_v_w', type=float, default=1, help='ls_v_w')
    parser.add_argument('--ls_tri_w', type=float, default=1, help='ls_tri_w')
    parser.add_argument('--ls_cos_w', type=float, default=1, help='ls_cos_w')
    parser.add_argument('--ls_sp_w', type=float, default=1, help='ls_sp_w')
    parser.add_argument('--ls_sm_w', type=float, default=1, help='ls_sm_w')
    parser.add_argument('--warm_up', type=int, default=1500, help='Steps of warm up')
    parser.add_argument('--thre_pesudo_a', type=float, default=0.15, help='thre_pesudo_a')
    parser.add_argument('--thre_var_a', type=float, default=0.5, help='thre_var_a')
    parser.add_argument('--pl_his_num', type=int, default=5)
    parser.add_argument('--sch_step_s', type=int, default=4000)
    parser.add_argument('--sch_gamma_s', type=float, default=0.9)
    parser.add_argument('--layer_ts', type=int, default=2, help='n_layer of global_atte')
    parser.add_argument('--law1', type=int, default=2, help='Local window size')
    parser.add_argument('--law2', type=int, default=4, help='Local window size')
    parser.add_argument('--law3', type=int, default=6, help='Local window size')
    parser.add_argument('--law4', type=int, default=8, help='Local window size')

    parser.add_argument('--ID', type=str, default='xd_testing', help='ID of model')
    parser.add_argument('--gpus', default=0, type=int, choices=[0, 1, 2, 3], help='gpus')

    return init_args(parser.parse_args())

def init_args(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    return args
