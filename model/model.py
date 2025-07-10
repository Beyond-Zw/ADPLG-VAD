import sys
import torch,types
import torch.nn as nn
import torch.nn.init as torch_init
import math
from model.global_attetion import Transformer
from local_attention.transformer import LocalTransformer
from multiprocessing import Process, Queue
torch.set_default_tensor_type('torch.FloatTensor')

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

class embedding(nn.Module):
    def __init__(self, input_size, out_size):
        super(embedding, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3,
                    stride=1, padding=1),
            nn.ReLU(),
        )
    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = x.permute(0, 2, 1)

        return x


class VSSF(nn.Module):
    def __init__(self,args, input_size=2048, nonlinear=False):
        super(VSSF, self).__init__()
        self.args = args
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU(), nn.Linear(input_size, input_size), nn.Tanh())
            self.k = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU(), nn.Linear(input_size, input_size), nn.Tanh())
            self.v = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU(), nn.Linear(input_size, input_size), nn.Tanh())

            self.k_a_n = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU(), nn.Linear(input_size, input_size), nn.Tanh())
            # self.v_a_n = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU(), nn.Linear(input_size, input_size), nn.Tanh())

        else:
            # self.q = nn.Linear(input_size, input_size)
            # self.k = nn.Linear(input_size, input_size)
            # self.v = nn.Linear(input_size, input_size)
            self.q = nn.Conv1d(in_channels=input_size, out_channels=512, kernel_size=1, padding=0)
            self.k = nn.Conv1d(in_channels=input_size, out_channels=512, kernel_size=1, padding=0)
            self.v = nn.Conv1d(in_channels=input_size, out_channels=512, kernel_size=1, padding=0)

    def forward(self, feats):

        B, N, C = feats.shape
        a_feats = feats[int(B/2):]
        n_feats = feats[:int(B/2)]

        a_feats = a_feats.permute(0, 2, 1)
        n_feats = n_feats.permute(0, 2, 1)

        Q_a = self.q(a_feats).permute(0, 2, 1).reshape(int(B/2), N, 512)
        K_a = self.k(a_feats).permute(0, 2, 1).reshape(int(B/2), N, 512)
        V_a = self.v(a_feats).permute(0, 2, 1).reshape(int(B/2), N, 512)
        # V_a = a_feats

        # Q_a_n = self.q(a_feats)
        K_a_n = self.k(a_feats).permute(0, 2, 1).reshape(int(B/2), N, 512)
        V_a_n = self.v(a_feats).permute(0, 2, 1).reshape(int(B/2), N, 512)
        # V_a_n = a_feats

        Q_n = self.q(n_feats).permute(0, 2, 1).reshape(int(B/2), N, 512)
        K_n = self.k(n_feats).permute(0, 2, 1).reshape(int(B/2), N, 512)
        V_n = self.v(n_feats).permute(0, 2, 1).reshape(int(B/2), N, 512)
        # V_n = n_feats

        # ******************* fuse A *********************
        a_att = (Q_a @ K_a.transpose(-1, -2)) / math.sqrt(Q_a.shape[-1])
        # print('att', att)
        # att_a = att_a.softmax(dim=2)
        a_scores_att = torch.diagonal(a_att, offset=0, dim1=1, dim2=2)
        # scores_att_temp = scores_att_a.detach().cpu().numpy()
        a_scores_att = a_scores_att.softmax(dim=1)
        # print('scores_att_one', scores_att_one)
        a_scores_att_one_max = torch.max(a_scores_att, dim=1)[0].unsqueeze(-1).repeat(1, self.args.num_segments)
        a_scores_att_one_min = torch.min(a_scores_att, dim=1)[0].unsqueeze(-1).repeat(1, self.args.num_segments)
        a_scores_att_one_norm = (a_scores_att - a_scores_att_one_min) / (
                    a_scores_att_one_max - a_scores_att_one_min + 1e-8)
        # print('scores_att_one_norm',scores_att_one_norm)
        # a_scores_att_ten_norm = a_scores_att_one_norm.unsqueeze(1).repeat(1, 10, 1)
        a_feats = (a_scores_att.unsqueeze(-2) @ V_a).squeeze(1)

        # ******************* fuse A_N *********************
        a_n_att = (Q_n @ K_a_n.transpose(-1, -2)) / math.sqrt(Q_n.shape[-1])
        # print('att', att)
        # att_a = att_a.softmax(dim=2)
        a_n_scores_att = torch.diagonal(a_n_att, offset=0, dim1=1, dim2=2)
        # scores_att_temp = scores_att_a.detach().cpu().numpy()
        a_n_scores_att = a_n_scores_att.softmax(dim=1)
        # print('scores_att_one', scores_att_one)
        a_n_scores_att_one_max = torch.max(a_n_scores_att, dim=1)[0].unsqueeze(-1).repeat(1, self.args.num_segments)
        a_n_scores_att_one_min = torch.min(a_n_scores_att, dim=1)[0].unsqueeze(-1).repeat(1, self.args.num_segments)
        a_n_scores_att_one_norm = (a_n_scores_att - a_n_scores_att_one_min) / (a_n_scores_att_one_max - a_n_scores_att_one_min +1e-8)
        # print('scores_att_one_norm',scores_att_one_norm)

        a_n_feats = (a_n_scores_att.unsqueeze(-2) @ V_a_n).squeeze(1)

        # ******************* fuse N *********************
        n_att = (Q_n @ K_n.transpose(-1, -2)) / math.sqrt(Q_n.shape[-1])
        # print('att', att)
        # att_a = att_a.softmax(dim=2)
        n_scores_att = torch.diagonal(n_att, offset=0, dim1=1, dim2=2)
        # scores_att_temp = scores_att_a.detach().cpu().numpy()
        n_scores_att = n_scores_att.softmax(dim=1)
        n_feats = (n_scores_att.unsqueeze(-2) @ V_n).squeeze(1)
        # n_feats = n_feats.mean(2)

        feats = torch.cat((n_feats, a_feats), dim=0)

        return feats, a_scores_att_one_norm, a_n_scores_att_one_norm, a_scores_att, a_n_scores_att, a_feats, a_n_feats, n_feats

class Video_Classifer(nn.Module):
    def __init__(self, args, input_dim=1024):
        super(Video_Classifer, self).__init__()
        self.args = args
        self.classifier_v = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        # self.modal_fusion = nn.Sequential(
        #     nn.Linear(input_dim, 1024),
        # )

        self.Snippts_Fuse = VSSF(args, input_size=1024)
        self.weight_init()

    def weight_init(self):
        for layer in self.classifier_v:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        # x = self.modal_fusion(x)
        feats_v, a_scores_att_one_norm, a_n_scores_att_one_norm, a_scores_att, a_n_scores_att, a_feats, a_n_feats, n_feats = self.Snippts_Fuse(x)
        scores_v = self.classifier_v(feats_v)

        return scores_v, a_scores_att_one_norm, a_n_scores_att_one_norm, a_scores_att, a_n_scores_att, feats_v, a_feats, a_n_feats, n_feats#

class Snippet_Classifer(nn.Module):
    def __init__(self, args, input_dim=1024):
        super(Snippet_Classifer, self).__init__()
        self.args = args

        self.classifier_s = nn.Sequential(
            nn.Linear(1024, 32),  # 0814-1-3
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # self.modal_fusion = nn.Sequential(
        #            nn.Linear(input_dim, 1024),
        #        )

        self.weight_init()
        self.vars_s = nn.ParameterList()
        for i, param in enumerate(self.classifier_s.parameters()):
            self.vars_s.append(param)

        self.LSA1 = LocalTransformer(num_tokens=256, dim=1024, local_attn_window_size=args.law1).cuda()
        self.LSA2 = LocalTransformer(num_tokens=256, dim=1024, local_attn_window_size=args.law2).cuda()
        self.LSA3 = LocalTransformer(num_tokens=256, dim=1024, local_attn_window_size=args.law3).cuda()
        self.LSA4 = LocalTransformer(num_tokens=256, dim=1024, local_attn_window_size=args.law4).cuda()

        self.embedding = embedding(input_size=1024, out_size=1024)
        self.norm = nn.LayerNorm(1024)
        self.selfatt = Transformer(1024, args.layer_ts, 4, 256, 1024, dropout=0.5)

    def weight_init(self):
        for layer in self.classifier_s:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        # x = self.modal_fusion(x)
        bs, t, f = x.size()
        x = self.embedding(x)

        # x = self.to_out(x)
        # x = x.permute(1, 0, 2)
        # x1 = self.to_out(x)
        # x1, x2, x3, x4= self.LSA1(x), self.LSA2(x),self.LSA3(x), self.LSA4(x)
        x1 = self.LSA1(x)
        x2 = self.LSA2(x)
        x3 = self.LSA3(x)
        x4 = self.LSA4(x)
        x = torch.cat((x1, x2, x3, x4), dim=2)

        x = self.selfatt(x)

        score_s = self.classifier_s(x)
        # print("score_s_out", score_s[0])
        return score_s



