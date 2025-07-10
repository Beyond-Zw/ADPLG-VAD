import torch
import torch.utils.data as data
import os
import numpy as np
import utils


class UCF_crime(data.DataLoader):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        split_path = os.path.join('list', 'UCF_{}.list'.format(self.mode))
        split_file = open(split_path, 'r')
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.split())
        split_file.close()
        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[8100:]
            elif is_normal is False:
                self.vid_list = self.vid_list[:8100]
            else:
                assert (is_normal == None)
                print("Please sure is_normal=[True/False]")
                self.vid_list = []

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):

        if self.mode == "Test":
            data, label, name = self.get_data(index)
            return data, label, name
        else:
            data, label, name = self.get_data(index)
            return data, label, name

    def get_data(self, index):
        vid_info = self.vid_list[index][0]
        name = vid_info.split("/")[-1].split("_x264")[0]
        video_feature = np.load(vid_info).astype(np.float32)

        if "Normal" in vid_info.split("/")[-1]:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)
        if self.mode == "Train":
            new_feat = np.zeros((self.num_segments, video_feature.shape[1])).astype(np.float32)
            r = np.linspace(0, len(video_feature), self.num_segments + 1, dtype=np.int64)
            for i in range(self.num_segments):
                if r[i] != r[i + 1]:
                    new_feat[i, :] = np.mean(video_feature[r[i]:r[i + 1], :], 0)
                else:
                    new_feat[i:i + 1, :] = video_feature[r[i]:r[i] + 1, :]
            video_feature = new_feat
        if self.mode == "Test":
            return video_feature, label, name
        else:
            return video_feature, label, name

class XDVideo(data.DataLoader):
    def __init__(self, root_dir, mode, modal, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.data_path = root_dir
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        if self.modal == 'RGB':
            self.feature_path = []
            if self.mode == "Train":
                self.feature_path.append(os.path.join(self.data_path, "i3d-features", 'RGB'))
            else:
                self.feature_path.append(os.path.join(self.data_path, "i3d-features", 'RGBTest'))
        elif self.modal == 'RGB+audio':
            self.RGB_feature_path = []
            self.audio_feature_path = []
            if self.mode == "Train":
                self.RGB_feature_path.append(os.path.join(self.data_path, "i3d-features", "RGB"))
                self.audio_feature_path.append(os.path.join(self.data_path, "vggish-features", "train"))
            else:
                self.RGB_feature_path.append(os.path.join(self.data_path, "i3d-features", "RGBTest"))
                self.audio_feature_path.append(os.path.join(self.data_path, "vggish-features", "test"))
        else:
            self.feature_path = os.path.join(self.data_path, modal)
        vid_split_path = os.path.join("list", 'XD_{}.list'.format(self.mode))
        vid_split_file = open(vid_split_path, 'r', encoding="utf-8")
        self.vid_list = []
        for line in vid_split_file:
            self.vid_list.append(line.split())
        vid_split_file.close()

        audio_split_path = os.path.join("list", 'XD_audio_{}.list'.format(self.mode))
        audio_split_file = open(audio_split_path, 'r', encoding="utf-8")
        self.audio_list = []
        for line in audio_split_file:
            self.audio_list.append(line.split())
        audio_split_file.close()

        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[9525:]
                self.audio_list = self.audio_list[1905:]
            elif is_normal is False:
                self.vid_list = self.vid_list[:9525]
                self.audio_list = self.audio_list[:1905]
            else:
                assert (is_normal == None)
                print("Please sure is_normal = [True/False]")
                self.vid_list = []
                self.audio_list = []

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data, label, vid_name = self.get_data(index)
        return data, label, vid_name

    def get_data(self, index):
        vid_name = self.vid_list[index][0]
        audio_name = self.audio_list[index // 5][0]
        label = torch.tensor(0.0)
        if "_label_A" not in vid_name:
            label = torch.tensor(1.0)
        video_feature = np.load(vid_name).astype(np.float32)
        audio_feature = np.load(audio_name).astype(np.float32)
        if self.modal == 'RGB+audio':
            video_feature = np.concatenate((video_feature, audio_feature), axis=1)
        else:
            video_feature = video_feature

        if self.mode == "Train":
            new_feat = np.zeros((self.num_segments, video_feature.shape[1])).astype(np.float32)
            r = np.linspace(0, len(video_feature), self.num_segments + 1, dtype=np.int64)
            for i in range(self.num_segments):
                if r[i] != r[i + 1]:
                    new_feat[i, :] = np.mean(video_feature[r[i]:r[i + 1], :], 0)
                else:
                    new_feat[i:i + 1, :] = video_feature[r[i]:r[i] + 1, :]
            video_feature = new_feat
        if self.mode == "Test":
            return video_feature, label, vid_name
        else:
            return video_feature, label, vid_name
        return video_feature, label, vid_name

class XDVideo_per_class(data.DataLoader):
    def __init__(self, root_dir, mode, modal, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.data_path=root_dir
        self.mode=mode
        self.modal=modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        if self.modal == 'RGB':
            self.feature_path = []
            if self.mode == "Train":
                self.feature_path.append(os.path.join(self.data_path, "i3d-features", 'RGB'))
            else:
                self.feature_path.append(os.path.join(self.data_path, "i3d-features", 'RGBTest'))
        else:
            self.feature_path = os.path.join(self.data_path, modal)
        split_path = os.path.join("list/XD_category_lists_raw",'XD_G_{}.list'.format(self.mode))
        split_file = open(split_path, 'r',encoding="utf-8")
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.split())
        split_file.close()
        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[9525:]
            elif is_normal is False:
                self.vid_list = self.vid_list[:9525]
            else:
                assert (is_normal == None)
                print("Please sure is_normal = [True/False]")
                self.vid_list=[]

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data,label, vid_name = self.get_data(index)
        return data, label, vid_name

    def get_data(self, index):
        vid_name = self.vid_list[index][0]
        label=torch.tensor(0.0)
        if "_label_A" not in vid_name:
            label=torch.tensor(1.0)
        video_feature = np.load(os.path.join(self.feature_path[0],
                                vid_name )).astype(np.float32)
        if self.mode == "Train":
            new_feature = np.zeros((self.num_segments,self.len_feature)).astype(np.float32)
            sample_index = utils.random_perturb(video_feature.shape[0],self.num_segments)
            for i in range(len(sample_index)-1):
                if sample_index[i] == sample_index[i+1]:
                    new_feature[i,:] = video_feature[sample_index[i],:]
                else:
                    new_feature[i,:] = video_feature[sample_index[i]:sample_index[i+1],:].mean(0)

            video_feature = new_feature
        return video_feature, label, vid_name

class SHT(data.DataLoader):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        split_path = os.path.join('list', 'SHT_{}.list'.format(self.mode))
        split_file = open(split_path, 'r')
        self.anomaly_vid_list = []
        self.normal_vid_list = []
        self.vid_list = []
        for line in split_file:
            if line.split()[-1]=="1.0":
                self.anomaly_vid_list.append(line.split())
            elif line.split()[-1]=="0.0":
                self.normal_vid_list.append(line.split())
            else:
                self.vid_list.append(line.split())
        split_file.close()
        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.normal_vid_list
            elif is_normal is False:
                self.vid_list = self.anomaly_vid_list
            else:
                assert (is_normal == None)
                print("Please sure is_normal=[True/False]")
                self.vid_list = []

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):

        if self.mode == "Test":
            data, label, name = self.get_data(index)
            return data, label, name
        else:
            data, label, name = self.get_data(index)
            return data, label, name

    def get_data(self, index):
        vid_info = self.vid_list[index]
        name = vid_info[0].split("/")[-1].split(".")[0]
        video_feature = np.load(vid_info[0]).astype(np.float32)

        if vid_info[-1]=="1.0" :
            label = torch.tensor(1.0)
        else:
            label = torch.tensor(0.0)
        if self.mode == "Train":
            new_feat = np.zeros((self.num_segments, video_feature.shape[1])).astype(np.float32)
            r = np.linspace(0, len(video_feature), self.num_segments + 1, dtype=np.int64)
            for i in range(self.num_segments):
                if r[i] != r[i + 1]:
                    new_feat[i, :] = np.mean(video_feature[r[i]:r[i + 1], :], 0)
                else:
                    new_feat[i:i + 1, :] = video_feature[r[i]:r[i] + 1, :]
            video_feature = new_feat
        if self.mode == "Test":
            return video_feature, label, name
        else:
            return video_feature, label, name