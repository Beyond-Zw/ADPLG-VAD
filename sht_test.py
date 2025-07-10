from dataset_loader import *
from sklearn.metrics import roc_curve,auc,precision_recall_curve
from options import *
from config import *
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from model.model import Video_Classifer, Snippet_Classifer
import warnings
warnings.filterwarnings("ignore")
class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def test(net, test_loader, test_info, step, model_file = None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        load_iter = iter(test_loader)
        frame_gt = np.load("frame_label/sh-gt.npy")
        frame_predict = None

        temp_predict = torch.zeros((0)).cuda()

        for i in range(len(test_loader.dataset)):
            _data, _label, name = next(load_iter)
            _data = _data.cuda()

            res = net(_data)
            a_predict = res
            temp_predict = torch.cat([temp_predict, a_predict], dim=0)
            if (i + 1) % 10 == 0 :
                a_predict = temp_predict.mean(0).cpu().numpy()

                fpre_ = np.repeat(a_predict, 16)
                if frame_predict is None:
                    frame_predict = fpre_
                else:
                    frame_predict = np.concatenate([frame_predict, fpre_])
                temp_predict = torch.zeros((0)).cuda()

        fpr,tpr,_ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)
        print("auc",auc_score)

        precision, recall, th = precision_recall_curve(frame_gt, frame_predict,)
        ap_score = auc(recall, precision)
        print("ap",ap_score)

        test_info["step"].append(step)
        test_info["auc"].append(auc_score)
        test_info["ap"].append(ap_score)


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


    test_loader = DataLoaderX(
        SHT(root_dir=config.root_dir, mode='Test', modal=config.modal, num_segments=config.num_segments,
                len_feature=config.len_feature),
        batch_size=1,
        shuffle=False, num_workers=config.num_workers,
        worker_init_fn=worker_init_fn)

    test_info = {"step": [], "auc": [], "ap": [], "ac": []}
    step = 0
    model_file = config.model_file
    test(net_s, test_loader, test_info, step, model_file=None)

