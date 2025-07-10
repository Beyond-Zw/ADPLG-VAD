import torch
import torch.nn as nn
from utils import max_min_one_dim_norm,max_min_two_dim_norm
import copy


def norm(data):
    l2=torch.norm(data, p = 2, dim = -1, keepdim = True)
    return torch.div(data, l2)

def sparsity(arr, batch_size, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2*loss


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]
    loss = torch.sum((arr2-arr)**2)
    return lamda1*loss


def l1_penalty(var):
    return torch.mean(torch.norm(var, dim=0))

class VC_Loss(torch.nn.Module):
    def __init__(self):
        super(VC_Loss, self).__init__()
        self.criterion_v = torch.nn.BCELoss()

    def forward(self, config, scores_v, labels_v):

        random_index = torch.randperm(config.batch_size*2).cuda()
        scores_v = scores_v.squeeze(-1)[random_index]
        labels_v = labels_v[random_index]
        loss_cls_v = self.criterion_v(scores_v, labels_v)  # BCE loss in the score space

        return loss_cls_v

class SC_Loss(torch.nn.Module):
    def __init__(self):
        super(SC_Loss, self).__init__()

        self.criterion_s = torch.nn.BCELoss(reduction='none')
    def forward(self, config, score_s, a_scores_att_one_norm):
        score_s=score_s.squeeze(-1)

        labels_s = torch.zeros(config.batch_size*2, config.num_segments).cuda()
        score_s_temp = torch.ones(config.batch_size * 2, config.num_segments).cuda()
        a_scores_att_one_binary = a_scores_att_one_norm.detach()

        hard_mask = a_scores_att_one_binary > 10
        a_scores_att_one_binary = torch.where(hard_mask, 0, a_scores_att_one_binary)
        score_s_a_mask = torch.where(hard_mask, 0, 1)
        score_s_temp[config.batch_size:,:] = score_s_a_mask
        labels_s[config.batch_size:,:] = a_scores_att_one_binary
        score_s = score_s*score_s_temp
        loss_cls_s = self.criterion_s(score_s, labels_s)

        mask_zero = loss_cls_s==0
        loss_cls_s = torch.where(mask_zero, torch.nan, loss_cls_s)

        loss_cls_s = torch.nanmean(loss_cls_s)

        labels_s_vis = labels_s[config.batch_size:, :]

        return loss_cls_s, labels_s_vis


def train(net_v, net_s, normal_loader, abnormal_loader, optimizer_v, optimizer_s, step, config, pl_dict, scheduler_s):
    net_v.train()
    net_s.train()
    ninput, nlabel, vid_n_name = next(normal_loader)
    ainput, alabel, vid_a_name = next(abnormal_loader)
    for key in vid_a_name:
        pl_dict.setdefault(key, [])
    _data = torch.cat((ninput, ainput), 0)
    _label = torch.cat((nlabel, alabel), 0)
    _data = _data.cuda()
    _label = _label.cuda()

    scores_v, a_scores_att_one_norm, a_n_scores_att_one_norm, a_scores_att, a_n_scores_att, feats_v, a_feats, a_n_feats, n_feats = \
        net_v(_data)

    a_scores_att_one_norm_copy = copy.deepcopy(a_scores_att_one_norm.detach())
    i = 0
    for key in vid_a_name:

        if len(pl_dict[key]) < config.pl_his_num:
            pl_dict[key].append(a_scores_att_one_norm_copy[i])
            i += 1
        else:
            pl_dict[key].append(a_scores_att_one_norm_copy[i])

            del pl_dict[key][0]

            pl_dict_mean = torch.stack(pl_dict[key]).mean(0)
            # print("pl_dict_mean", pl_dict_mean)
            pl_dict_mean_norm = max_min_one_dim_norm(pl_dict_mean)
            pl_mean_a = torch.where(pl_dict_mean_norm >= config.thre_pesudo_a, 0.5, -0.5)
            # print("pl_dict_mean_norm", pl_dict_mean_norm)
            pl_dict_var = torch.stack(pl_dict[key]).var(0)
            pl_dict_var_norm = max_min_one_dim_norm(pl_dict_var)
            # ****************** static hard samples *******************
            num_hard_this_video = torch.sum(pl_dict_var_norm > config.thre_var_a).item()
            # ************************************************
            # print(pl_dict_var_norm)
            pl_var_a = torch.where(pl_dict_var_norm <= config.thre_var_a, 0.5, 1e8)
            # print("pl_dict_var_norm:",pl_dict_var_norm)
            pl_a = pl_mean_a+pl_var_a
            # print("pl_a", pl_a)
            # temp = pl_a.detach().cpu().numpy()
            a_scores_att_one_norm[i] = pl_a
            # print(i, key)
            i = i + 1

    loss_smooth = smooth(a_scores_att, 1)
    loss_sparsity = sparsity(a_scores_att, config.batch_size, 1)

    # Compactness-Separation Loss
    triplet_loss = nn.TripletMarginLoss(margin=1, p=2)
    lost_triplet = triplet_loss(n_feats, a_n_feats, a_feats)

    # Distributional dissimilarity loss
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_simi_loss = torch.sum(cos(a_scores_att, a_n_scores_att)) / config.batch_size

    # Video Classification Loss
    VC_Loss_criterion = VC_Loss()
    loss_cls_v = VC_Loss_criterion(config, scores_v, _label)

    cost_v = config.ls_v_w * loss_cls_v + config.ls_tri_w * lost_triplet + \
             config.ls_cos_w * cos_simi_loss + config.ls_sm_w * loss_smooth + config.ls_sp_w * loss_sparsity

    optimizer_v.zero_grad()
    cost_v.backward()
    optimizer_v.step()

    # *****************************************training snippets classfier**********************************************#
    loss_cls_s = torch.tensor(0)
    if step > config.warm_up:
        score_s = net_s(_data)
        # Snippet Classification Loss
        SC_Loss_criterion = SC_Loss()
        loss_cls_s, labels_s_vis = SC_Loss_criterion(config, score_s, a_scores_att_one_norm)
        cost_s = config.ls_s_w * loss_cls_s
        optimizer_s.zero_grad()
        cost_s.backward()
        optimizer_s.step()
        scheduler_s.step()

    return cost_v, loss_cls_s, lost_triplet, loss_cls_v, cos_simi_loss
