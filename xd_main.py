import numpy as np
import torch.utils.data as data
import utils
from options import *
from config import *
from train import *
from xd_test import test
from model.model import Video_Classifer, Snippet_Classifer

from dataset_loader import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

if __name__ == "__main__":

    args = parse_args()

    config = Config(args)
    worker_init_fn = None
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{config.gpus}"

    if config.seed >= 0:
        utils.set_seed(config.seed)
        worker_init_fn = np.random.seed(config.seed)

    net_v = Video_Classifer(config, input_dim=config.len_feature)
    net_s = Snippet_Classifer(config, input_dim=config.len_feature)

    net_v = net_v.cuda()
    net_s = net_s.cuda()

    normal_train_loader = DataLoaderX(
        XDVideo(root_dir = config.root_dir, mode = 'Train',modal = config.modal, num_segments = config.num_segments, len_feature = config.len_feature, is_normal = True),
            batch_size = config.batch_size,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    abnormal_train_loader = DataLoaderX(
        XDVideo(root_dir = config.root_dir, mode='Train', modal = config.modal, num_segments = config.num_segments, len_feature = config.len_feature, is_normal = False),
            batch_size = config.batch_size,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    test_loader = DataLoaderX(
        XDVideo(root_dir = config.root_dir, mode = 'Test', modal = config.modal, num_segments = config.num_segments, len_feature = config.len_feature),
            batch_size = 5,
            shuffle = False, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn)

    # normal_train_loader = DataLoaderX(
    #     XDVideo_CLIP(root_dir = config.root_dir, mode = 'Train',modal = config.modal, num_segments = config.num_segments, len_feature = config.len_feature, is_normal = True),
    #         batch_size = config.batch_size,
    #         shuffle = True, num_workers = config.num_workers,
    #         worker_init_fn = worker_init_fn, drop_last = True)
    # abnormal_train_loader = DataLoaderX(
    #     XDVideo_CLIP(root_dir = config.root_dir, mode='Train', modal = config.modal, num_segments = config.num_segments, len_feature = config.len_feature, is_normal = False),
    #         batch_size = config.batch_size,
    #         shuffle = True, num_workers = config.num_workers,
    #         worker_init_fn = worker_init_fn, drop_last = True)
    # test_loader = DataLoaderX(
    #     XDVideo_CLIP(root_dir = config.root_dir, mode = 'Test', modal = config.modal, num_segments = config.num_segments, len_feature = config.len_feature),
    #         batch_size = 5,
    #         shuffle = False, num_workers = config.num_workers,
    #         worker_init_fn = worker_init_fn)

    test_info = {"step": [], "auc": [],"ap":[],"ac":[]}

    optimizer_v = torch.optim.Adam(net_v.parameters(), lr = config.lr_v[0], betas = (0.9, 0.999), weight_decay = 0.005)
    optimizer_s = torch.optim.Adam(net_s.parameters(), lr = config.lr_s[0], betas = (0.9, 0.999), weight_decay = 0.005)
    scheduler_s = StepLR(optimizer_s, config.sch_step_s, config.sch_gamma_s)

    pl_dict = {}
    for step in tqdm(
            range(1, config.num_iters + 1),
            total = config.num_iters,
            dynamic_ncols = True
        ):

        if (step - 1) % len(normal_train_loader) == 0:
            normal_loader_iter = iter(normal_train_loader)

        if (step - 1) % len(abnormal_train_loader) == 0:
            abnormal_loader_iter = iter(abnormal_train_loader)
        cost_v, loss_cls_s, lost_triplet, loss_cls_v, cos_simi_loss = train(net_v, net_s, normal_loader_iter,abnormal_loader_iter, optimizer_v, optimizer_s, step, config, pl_dict, scheduler_s)
        if step % 50 == 0:
            print(
                f"[Step {step}] Total loss: {cost_v.item():.4f}, VC Loss: {loss_cls_v.item():.4f}, Triplet: {lost_triplet.item():.4f}, SC Loss: {loss_cls_s.item():.4f}, COS Loss: {cos_simi_loss.item():.4f}")
        if step % 1000 == 0 and step > config.warm_up:
            test(net_s, test_loader, test_info, step, model_file = None)
            torch.save(net_s.state_dict(), os.path.join(args.model_path, f"xd_model_s_{config.ID}_step_{step}_ap_{test_info['ap'][-1]:4f}.pkl"))
            torch.save(net_v.state_dict(), os.path.join(args.model_path, f"xd_model_v_{config.ID}_step_{step}_ap_{test_info['ap'][-1]:4f}.pkl"))




