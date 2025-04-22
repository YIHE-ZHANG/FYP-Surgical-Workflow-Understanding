import os
import os.path
import numpy as np
import random
import torch
import json
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

import csv

class Breakfast(object):
    def __init__(self,
                 root='./data/breakfast',
                 transform=None, mode='val',
                 num_frames=16, ds=1, ol=0.5,
                 small_test=False,
                 frame_dir='./data/breakfast/frames/',
                 label_dir='./data/breakfast/action_ids/',
                 class_dir='./data/breakfast/bf_mapping.json',
                 ext_class_dir='./data/breakfast/bf_mapping_new.json',
                 pretrain=True, n_split=1):
        self.root = root
        self.transform = transform
        self.mode = mode
        self.num_frames = num_frames
        self.ds = ds
        self.overlap = ol
        self.small_test = small_test
        self.frame_dir = frame_dir
        self.label_dir = label_dir
        self.class_dir = class_dir
        self.ext_class_dir = ext_class_dir
        self.pretrain = pretrain
        self.n_split = n_split

        # if self.mode == 'train':
        with open(self.class_dir, 'r') as f:
            self.classes = json.load(f)
            self.classes = {int(k): v for k, v in self.classes.items()}
        # else:
        #     with open(self.ext_class_dir, 'r') as f:
        #         self.classes = json.load(f)
        #         self.classes = {int(k): v for k, v in self.classes.items()}

        if not self.small_test:
            if self.mode == 'train':
                self.train_split = np.load(
                    os.path.join(root, 'splits',
                                 f'train_split{self.n_split}_nf{self.num_frames}_ol{self.overlap}_ds{self.ds}_all.npy'))
            else:
                self.train_split = np.load(
                    os.path.join(root, 'splits',
                                 f'test_split{self.n_split}_nf{self.num_frames}_ol{self.overlap}_ds{self.ds}_all.npy'))
        else:
            self.train_split = np.load(
                os.path.join(root, 'splits', f'smalltest_split1_nf{self.num_frames}_ol{self.overlap}_ds{self.ds}.npy'))

    def frame_sampler(self, videoname, vlen):
        start_idx = int(videoname[1])
        seq_idx = np.arange(self.num_frames) * self.ds + start_idx
        seq_idx = np.where(seq_idx < vlen, seq_idx, vlen - 1)
        return seq_idx

    def __getitem__(self, index):
        videoname = self.train_split[index]
        vsplt = videoname[0].split('_', 2)
        vname_splt = np.copy(vsplt)
        if vsplt[1] == 'stereo':
            vname_splt[1] = 'stereo01'
            vname_splt[2] = vsplt[2][:-4]
        vpath = os.path.join(self.frame_dir, vsplt[0], vsplt[1], vsplt[2])
        vlen = len([f for f in os.listdir(vpath) if os.path.isfile(os.path.join(vpath, f))])
        vlabel = np.load(
            os.path.join(self.label_dir, vname_splt[0] + '_' + vname_splt[1] + '_' + vname_splt[2] + '.npy'))
        diff = vlabel.size - vlen
        if diff > 0:
            vlabel = vlabel[:-diff]
        elif diff < 0:
            vlabel = np.pad(vlabel, (0, -diff), 'constant', constant_values=(0, vlabel[-1]))
        path_list = os.listdir(vpath)
        path_list.sort(key=lambda x: int(x[4:-4]))
        frame_index = self.frame_sampler(videoname, vlen)
        seq = [Image.open(os.path.join(vpath, path_list[i])).convert('RGB') for i in frame_index]
        vid = vlabel[frame_index]
        if self.pretrain:
            vid = torch.from_numpy(vid)
            vid = torch.unique_consecutive(vid)
            vid = vid.numpy()
            vid = np.pad(vid, (0, 10 - vid.shape[0]), 'constant', constant_values=(0, -1))

        if self.transform is not None:
            seq = self.transform(seq)
        else:
            convert_tensor = transforms.ToTensor()
            seq = [convert_tensor(img) for img in seq]
            seq = torch.stack(seq)
        # seq = torch.stack(seq, 1)
        # seq = seq.permute(1, 0, 2, 3)
        return seq, vid

    def __len__(self):
        # return 2
        return len(self.train_split)


class Breakfast_feat(object):
    def __init__(self,
                 root='./data/breakfast',
                 transform=None, mode='train',
                 num_frames=8, n_seg=64,
                 small_test=False,
                 frame_dir='./data/breakfast/frames/',
                 class_dir="./data/breakfast/id2acti.txt",
                 label_dir="./data/breakfast/acti2id.json",
                 pretrain=True, n_split=5):
        self.root = root
        self.transform = transform
        self.mode = mode
        self.num_frames = num_frames
        self.n_seg = n_seg
        self.small_test = small_test
        self.frame_dir = frame_dir
        self.label_dir = label_dir
        self.class_dir = class_dir
        self.pretrain = pretrain
        self.n_split = n_split
        self.feat_dir = "./data/breakfast/features_dir/breakfast_vit_acti_ls/"

        with open(self.label_dir, 'r') as f:
            self.cls2id = json.load(f)
            self.cls2id = {k: int(v) for k, v in self.cls2id.items()}
        self.classes = {}
        with open(self.class_dir, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                line = line.split(' ', 1)
                self.classes[line[0]] = line[1]
                self.classes = {int(k): v for k, v in self.classes.items()}
        if self.mode == 'train':
            self.splt_dir = "./data/breakfast/splits/train.split" + str(self.n_split) + ".bundle"
        else:
            self.splt_dir = "./data/breakfast/splits/test.split" + str(self.n_split) + ".bundle"

        file_ptr = open(self.splt_dir, 'r')
        self.train_split = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

        with open('./data/breakfast/splits/breakfast_acti_vid2idx.json', 'r') as openf:
            self.act_splt = json.load(openf)

    def frame_sampler(self, vlen):
        if vlen > self.num_frames * self.n_seg:
            seq_idx = np.arange(0, vlen - self.num_frames, self.num_frames)
            seq_idx = np.append(seq_idx, vlen - self.num_frames)
            sorted_sample = [seq_idx[k] for k in sorted(random.sample(range(len(seq_idx)), self.n_seg))]
            result = [np.arange(ii, ii + self.num_frames) for ii in sorted_sample]
        else:
            result = [np.arange(i[0], i[0] + self.num_frames)
                      for i in np.array_split(range(vlen - self.num_frames), self.n_seg - 1)]
            result.append(np.arange(vlen - self.num_frames, vlen))

        return result

    def __getitem__(self, index):
        videoname = self.train_split[index]
        videoname = videoname.split('.', 2)[0]
        vsplt = videoname.split('_')
        cls_id = self.cls2id[vsplt[3]]
        seq = np.zeros((self.n_seg, self.num_frames, 768))
        n = 0
        for vid in self.act_splt[videoname]:
            seq[n, :, :] = np.load(os.path.join(self.feat_dir, videoname + '_' + vid + '.npy'))
            n += 1

        return seq, cls_id

    def __len__(self):
        # return 1
        return len(self.train_split)


class Breakfast_acti(data.Dataset):
    def __init__(self,
                 root='./data/breakfast',
                 transform=None, mode='val',
                 num_frames=32, ds=1, ol=0.5,
                 small_test=False,
                 frame_dir='./data/breakfast/frames/',
                 label_dir='./data/breakfast/action_ids/',
                 class_dir='./data/breakfast/bf_mapping.json',
                 id2acti_dir="./data/breakfast/id2acti.txt",
                 acti2id_dir="./data/breakfast/acti2id.json",
                 pretrain=True, n_split=1):
        self.root = root
        self.transform = transform
        self.mode = mode
        self.num_frames = num_frames
        self.ds = ds
        self.overlap = ol
        self.small_test = small_test
        self.frame_dir = frame_dir
        self.label_dir = label_dir
        self.class_dir = class_dir
        self.pretrain = pretrain
        self.n_split = n_split
        self.acti2id_dir = acti2id_dir
        self.id2acti_dir = id2acti_dir

        # if self.mode == 'train':
        with open(self.class_dir, 'r') as f:
            self.classes = json.load(f)
            self.classes = {int(k): v for k, v in self.classes.items()}

        with open(self.acti2id_dir, 'r') as f:
            self.cls2id = json.load(f)
            self.cls2id = {k: int(v) for k, v in self.cls2id.items()}
        self.acti_classes = {}
        with open(self.id2acti_dir, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                line = line.split(' ', 1)
                self.acti_classes[line[0]] = line[1]
                self.acti_classes = {int(k): v for k, v in self.acti_classes.items()}
        # else:
        #     with open(self.ext_class_dir, 'r') as f:
        #         self.classes = json.load(f)
        #         self.classes = {int(k): v for k, v in self.classes.items()}

        if not self.small_test:
            if self.mode == 'train':
                self.train_split = np.load(
                    os.path.join(root, 'splits',
                                 f'train_split{self.n_split}_nf{self.num_frames}_ol{self.overlap}_ds{self.ds}_all.npy'))
            else:
                self.train_split = np.load(
                    os.path.join(root, 'splits',
                                 f'test_split{self.n_split}_nf{self.num_frames}_ol{self.overlap}_ds{self.ds}_all.npy'))
        else:
            self.train_split = np.load(
                os.path.join(root, 'splits', f'smalltest_split1_nf{self.num_frames}_ol{self.overlap}_ds{self.ds}.npy'))

    def frame_sampler(self, videoname, vlen):
        start_idx = int(videoname[1])
        seq_idx = np.arange(self.num_frames) * self.ds + start_idx
        seq_idx = np.where(seq_idx < vlen, seq_idx, vlen - 1)
        return seq_idx

    def __getitem__(self, index):
        videoname = self.train_split[index]
        vsplt = videoname[0].split('_', 3)
        acti_name = vsplt[1]
        vd_name = vsplt[3]
        if acti_name == 'stereo':
            acti_name += '01'
            vd_name = vd_name[:-4]
        cls_id = self.cls2id[vd_name]
        vpath = os.path.join(self.frame_dir, vsplt[0], vsplt[1], vsplt[2] + '_' + vsplt[3])
        vlen = len([f for f in os.listdir(vpath) if os.path.isfile(os.path.join(vpath, f))])
        vlabel = np.load(
            os.path.join(self.label_dir, vsplt[0] + '_' + acti_name + '_' + vsplt[2] + '_' + vd_name + '.npy'))
        # diff = vlabel.size - vlen
        # if diff > 0:
        #     vlabel = vlabel[:-diff]
        # elif diff < 0:
        #     vlabel = np.pad(vlabel, (0, -diff), 'constant', constant_values=(0, vlabel[-1]))
        path_list = os.listdir(vpath)
        path_list.sort(key=lambda x: int(x[4:-4]))
        frame_index = self.frame_sampler(videoname, vlen)
        seq = [Image.open(os.path.join(vpath, path_list[i])).convert('RGB') for i in frame_index]
        vid = vlabel[frame_index]
        if self.pretrain:
            vid = torch.from_numpy(vid)
            vid = torch.unique_consecutive(vid)
            vid = vid.numpy()
            vid = np.pad(vid, (0, 10 - vid.shape[0]), 'constant', constant_values=(0, -1))

        if self.transform is not None:
            seq = self.transform(seq)
        else:
            convert_tensor = transforms.ToTensor()
            seq = [convert_tensor(img) for img in seq]
            seq = torch.stack(seq)
        # seq = torch.stack(seq, 1)
        # seq = seq.permute(1, 0, 2, 3)
        return seq, vid, cls_id

    def __len__(self):
        # return 1
        return len(self.train_split)


class Breakfast_FRAMES(data.Dataset):
    def __init__(self,
                 root='./data/breakfast',
                 small_test=False,
                 frame_dir='./data/breakfast/frames/',
                 save_feat_dir='bf_vit_features',
                 num_frames=32,
                 transforms=None):
        self.root = root
        self.small_test = small_test
        self.frame_dir = frame_dir
        self.save_feat_dir = save_feat_dir
        self.num_frames = num_frames
        self.transform = transforms
        #
        # self.data_lst = np.load(
        #     os.path.join(root, 'splits', 'breakfast_acti.npy'))

        self.data_lst = np.load(
            os.path.join(root, 'splits', 'breakfast_exfm.npy'))

    def frame_sampler(self, videoname, vlen):
        start_idx = int(videoname[1])
        seq_idx = np.arange(self.num_frames) + start_idx
        seq_idx = np.where(seq_idx < vlen, seq_idx, vlen - 1)
        return seq_idx

    def __getitem__(self, index):
        videoname = self.data_lst[index]
        vroot = videoname[0]
        vsplt = vroot.split('_', 2)
        vname_splt = np.copy(vsplt)
        if vsplt[1] == 'stereo':
            vname_splt[1] = 'stereo01'
            vname_splt[2] = vsplt[2][:-4]
        vpath = os.path.join(self.frame_dir, *vsplt)
        vlen = len([f for f in os.listdir(vpath) if os.path.isfile(os.path.join(vpath, f))])
        path_list = os.listdir(vpath)
        path_list.sort(key=lambda x: int(x[4:-4]))
        frame_index = self.frame_sampler(videoname, vlen)
        seq = [Image.open(os.path.join(vpath, path_list[i])).convert('RGB') for i in frame_index]
        if self.transform is not None:
            seq = self.transform(seq)
        else:
            convert_tensor = transforms.ToTensor()
            seq = [convert_tensor(img) for img in seq]
            seq = torch.stack(seq)
        fname = vname_splt[0] + '_' + vname_splt[1] + '_' + vname_splt[2] + '_' + videoname[1] + '.npy'
        return seq, fname

    def __len__(self):
        # return 1
        return len(self.data_lst)


class GTEA(data.Dataset):
    def __init__(self,
                 root='./data/gtea',
                 transform=None, mode='val',
                 num_frames=16, ds=None, ol=None,
                 small_test=False,
                 frame_dir="./data/gtea/frames/",
                 label_dir="./data/gtea/action_descriptions_id/",
                 class_dir="./data/gtea/gtea_id2act.json",
                 pretrain=True,
                 n_split=None):
        if ds is None:
            ds = [1, 2, 4]
        if ol is None:
            ol = [1, 1, 0.5]
        self.root = root
        self.transform = transform
        self.mode = mode
        self.num_frames = num_frames
        self.ds = ds
        self.overlap = ol
        self.small_test = small_test
        self.frame_dir = frame_dir
        self.label_dir = label_dir
        self.class_dir = class_dir
        self.pretrain = pretrain
        self.n_split = n_split

        with open(self.class_dir, 'r') as f:
            self.classes = json.load(f)
            self.classes = {int(k): v for k, v in self.classes.items()}

        if not self.small_test:
            if self.mode == 'train':
                self.train_split = np.load(
                    os.path.join(root, 'splits',
                                 f'train_split{self.n_split}_nf{self.num_frames}_ol{self.overlap}_ds{self.ds}.npy'))
            else:
                self.train_split = np.load(
                    os.path.join(root, 'splits',
                                 f'test_split{self.n_split}_nf{self.num_frames}_ol{self.overlap}_ds{self.ds}.npy'))
        else:
            self.train_split = np.load(
                os.path.join(root, 'splits',
                             f'smalltest_split{self.n_split}_nf{self.num_frames}_ol{self.overlap}_ds{self.ds}.npy'))

    def frame_sampler(self, videoname, vlen):
        start_idx = int(videoname[1])
        ds = videoname[2]
        seq_idx = np.arange(self.num_frames) * int(ds) + start_idx
        seq_idx = np.where(seq_idx < vlen, seq_idx, vlen - 1)
        return seq_idx

    def __getitem__(self, index):
        videoname = self.train_split[index]
        vsplt = videoname[0]
        vpath = os.path.join(self.frame_dir, vsplt)
        vlen = len([f for f in os.listdir(vpath) if os.path.isfile(os.path.join(vpath, f))])
        vlabel = np.load(os.path.join(self.label_dir, vsplt + '.npy')).astype(np.int_)
        path_list = os.listdir(vpath)
        path_list.sort(key=lambda x: int(x[4:-4]))
        frame_index = self.frame_sampler(videoname, vlen)
        seq = [Image.open(os.path.join(vpath, path_list[i])).convert('RGB') for i in frame_index]
        vid = vlabel[frame_index]
        if self.pretrain:
            vid = torch.from_numpy(vid)
            vid = torch.unique_consecutive(vid)
            vid = vid.numpy()
            vid = np.ma.masked_equal(vid, 0)
            vid = vid.compressed()
            vid = np.pad(vid, (0, 10 - vid.shape[0]), 'constant', constant_values=(0, -1))

        if self.transform is not None:
            seq = self.transform(seq)
        else:
            convert_tensor = transforms.ToTensor()
            seq = [convert_tensor(img) for img in seq]
            seq = torch.stack(seq)
        # seq = torch.stack(seq, 1)
        # seq = seq.permute(1, 0, 2, 3)
        return seq, vid

    def __len__(self):
        # return 2
        return len(self.train_split)


class GTEA_FRAMES(data.Dataset):
    def __init__(self,
                 root='./data/gtea',
                 small_test=False,
                 frame_dir='./data/gtea/frames/',
                 save_feat_dir='gtea_vit_features',
                 transform=None):
        self.root = root
        self.small_test = small_test
        self.frame_dir = frame_dir
        self.save_feat_dir = save_feat_dir
        self.transform = transform

        all_files = os.walk(self.frame_dir)
        self.convert_tensor = transforms.ToTensor()
        self.data_lst = []
        for path, dir, filelst in all_files:
            if len(filelst) > 0:
                self.data_lst.append((filelst, path))

    def __getitem__(self, index):
        videoname = self.data_lst[index]
        vroot = videoname[1]
        path_list = videoname[0]
        # vlen = len(path_list)
        path_list.sort(key=lambda x: int(x[4:-4]))
        seq = [Image.open(os.path.join(vroot, p)).convert('RGB') for p in path_list]
        if self.transform is not None:
            seq = self.transform(seq)
        else:
            convert_tensor = transforms.ToTensor()
            seq = [convert_tensor(img) for img in seq]
            seq = torch.stack(seq)
        vsplt = vroot.split('/')[-1]
        fname = vsplt + '.npy'
        return seq, fname

    def __len__(self):
        return len(self.data_lst)


class SALADS(data.Dataset):
    def __init__(self,
                 root='./data/50salads',
                 transform=None, mode='val',
                 num_frames=16, ds=None, ol=None,
                 small_test=False,
                 frame_dir="./data/50salads/frames/",
                 label_dir="./data/50salads/action_descriptions_id/",
                 class_dir="./data/50salads/mapping_adj.json",
                 pretrain=True,
                 n_split=1):
        if ds is None:
            ds = [24, 32]
        if ol is None:
            ol = [1, 1]
        self.root = root
        self.transform = transform
        self.mode = mode
        self.num_frames = num_frames
        self.ds = ds
        self.overlap = ol
        self.small_test = small_test
        self.frame_dir = frame_dir
        self.label_dir = label_dir
        self.class_dir = class_dir
        self.pretrain = pretrain
        self.n_split = n_split

        # if self.mode == 'train':
        with open(self.class_dir, 'r') as f:
            self.classes = json.load(f)
            self.classes = {int(k): v for k, v in self.classes.items()}
        # else:
        #     with open(self.ext_class_dir, 'r') as f:
        #         self.classes = json.load(f)
        #         self.classes = {int(k): v for k, v in self.classes.items()}

        if not self.small_test:
            if self.mode == 'train':
                self.train_split = np.load(
                    os.path.join(root, 'splits',
                                 f'train_split{self.n_split}_nf{self.num_frames}_ol{self.overlap}_ds{self.ds}.npy'))
            else:
                self.train_split = np.load(
                    os.path.join(root, 'splits',
                                 f'test_split{self.n_split}_nf{self.num_frames}_ol{self.overlap}_ds{self.ds}.npy'))
        else:
            self.train_split = np.load(
                os.path.join(root, 'splits',
                             f'smalltest_split{self.n_split}_nf{self.num_frames}_ol{self.overlap}_ds{self.ds}.npy'))

    def frame_sampler(self, videoname, vlen):
        start_idx = int(videoname[1])
        ds = videoname[2]
        seq_idx = np.arange(self.num_frames) * int(ds) + start_idx
        seq_idx = np.where(seq_idx < vlen, seq_idx, vlen - 1)
        return seq_idx

    def __getitem__(self, index):
        videoname = self.train_split[index]
        vsplt = videoname[0]
        vpath = os.path.join(self.frame_dir, vsplt)
        vlen = len([f for f in os.listdir(vpath) if os.path.isfile(os.path.join(vpath, f))])
        vlabel = np.load(os.path.join(self.label_dir, vsplt + '.npy')).astype(np.int_)
        # diff = vlabel.size - vlen
        # if diff > 0:
        #     vlabel = vlabel[:-diff]
        # elif diff < 0:
        #     vlabel = np.pad(vlabel, (0, -diff), 'constant', constant_values=(0, vlabel[-1]))
        path_list = os.listdir(vpath)
        path_list.sort(key=lambda x: int(x[4:-4]))
        frame_index = self.frame_sampler(videoname, vlen)
        seq = [Image.open(os.path.join(vpath, path_list[i])).convert('RGB') for i in frame_index]
        vid = vlabel[frame_index]
        if self.pretrain:
            vid = torch.from_numpy(vid)
            vid = torch.unique_consecutive(vid)
            vid = vid.numpy()
            vid = np.ma.masked_equal(vid, 0)
            vid = vid.compressed()
            vid = np.pad(vid, (0, 10 - vid.shape[0]), 'constant', constant_values=(0, -1))

        if self.transform is not None:
            seq = self.transform(seq)
        else:
            convert_tensor = transforms.ToTensor()
            seq = [convert_tensor(img) for img in seq]
            seq = torch.stack(seq)
        # seq = torch.stack(seq, 1)
        # seq = seq.permute(1, 0, 2, 3)
        return seq, vid

    def __len__(self):
        # return 1
        return len(self.train_split)


class SALADS_FRAMES(data.Dataset):
    def __init__(self,
                 root='./data/50salads',
                 transform=None, mode='val',
                 num_frames=32,
                 frame_dir="./data/50salads/frames/",
                 label_dir="./data/50salads/action_descriptions_id/",
                 class_dir="./data/50salads/mapping_adj.json",
                 pretrain=True,
                 n_split=1):
        self.root = root
        self.transform = transform
        self.mode = mode
        self.num_frames = num_frames
        self.frame_dir = frame_dir
        self.label_dir = label_dir
        self.class_dir = class_dir
        self.pretrain = pretrain
        self.n_split = n_split

        self.train_split = np.load(
            os.path.join(root, 'splits', 'exfm_nf32.npy'))

    def frame_sampler(self, videoname, vlen):
        start_idx = int(videoname[1])
        seq_idx = np.arange(self.num_frames) + start_idx
        seq_idx = np.where(seq_idx < vlen, seq_idx, vlen - 1)
        return seq_idx

    def __getitem__(self, index):
        videoname = self.train_split[index]
        vsplt = videoname[0]
        vpath = os.path.join(self.frame_dir, vsplt)
        vlen = len([f for f in os.listdir(vpath) if os.path.isfile(os.path.join(vpath, f))])
        path_list = os.listdir(vpath)
        path_list.sort(key=lambda x: int(x[4:-4]))
        frame_index = self.frame_sampler(videoname, vlen)
        seq = [Image.open(os.path.join(vpath, path_list[i])).convert('RGB') for i in frame_index]

        if self.transform is not None:
            seq = self.transform(seq)
        else:
            convert_tensor = transforms.ToTensor()
            seq = [convert_tensor(img) for img in seq]
            seq = torch.stack(seq)
        fname = vsplt + '_' + videoname[1] + '.npy'
        return seq, fname

    def __len__(self):
        # return 1
        return len(self.train_split)
    

# RARP-50 dataset
class RARP50(data.Dataset):
    def __init__(self,
                 root='./data/rarp50',
                 transform=None, mode='val',
                 num_frames=16, ds=1, ol=0.5,
                 small_test=False,
                 frame_dir='./data/rarp50/',
                 class_dir=None,
                 pretrain=True, n_split=1):
        """
        RARP50 dataset class for Bridge-Prompt
        
        Args:
            root: Root directory for the dataset
            transform: Image transformations
            mode: 'train' or 'test'
            num_frames: Number of frames to use per clip
            ds: Downsampling rate
            ol: Overlap ratio
            small_test: Use a smaller test set for debugging
            frame_dir: Directory containing the frames
            class_dir: Optional path to a JSON file containing class mappings
            pretrain: Whether we're pretraining
            n_split: Split index
        """
        self.root = root
        self.transform = transform
        self.mode = mode
        self.num_frames = num_frames
        self.ds = ds
        self.overlap = ol
        self.small_test = small_test
        self.frame_dir = frame_dir
        self.pretrain = pretrain
        self.n_split = n_split
        self.class_dir = class_dir

        # Define the action class mapping for RARP-50
        # This can be overridden by loading from a JSON file if class_dir is provided
        self.classes = {
            0: "Other",
            1: "Picking-up the needle",
            2: "Positioning the needle tip", 
            3: "Pushing the needle through the tissue",
            4: "Pulling the needle out of the tissue",
            5: "Tying a knot",
            6: "Cutting the suture",
            7: "Returning/dropping the needle"
        }

        # Load class mapping from JSON file if provided
        if self.class_dir and os.path.exists(self.class_dir):
            import json
            with open(self.class_dir, 'r') as f:
                self.classes = json.load(f)
                self.classes = {int(k): v for k, v in self.classes.items()}

        # Load split file based on mode
        if self.mode == 'train':
            splits_dir = os.path.join(self.root, "splits")
            with open(os.path.join(splits_dir, "train_split1.txt"), 'r') as f:
                video_names = [line.strip() for line in f.readlines()]
                self.train_split = self._prepare_splits(video_names)
        else:
            splits_dir = os.path.join(self.root, "splits")
            with open(os.path.join(splits_dir, "test_split1.txt"), 'r') as f:
                video_names = [line.strip() for line in f.readlines()]
                self.train_split = self._prepare_splits(video_names)

    def _prepare_splits(self, video_names):
        """
        Prepare split data for training/testing
        
        Args:
            video_names: List of video directories in split file
            
        Returns:
            List of (videoname, start_idx, ds) tuples for frame sampling
        """
        split_data = []
        
        for video_path in video_names:
            # Get the video directory path
            video_dir = os.path.join(self.frame_dir, video_path)
            rgb_dir = os.path.join(video_dir, 'rgb')
            
            # Count frames in the directory
            if os.path.exists(rgb_dir):
                frame_count = len([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
                
                # Calculate number of clips we can extract
                clip_length = self.num_frames * self.ds
                stride = int(clip_length * (1 - self.overlap))
                
                # Generate clips with different starting points
                for start_idx in range(0, frame_count - clip_length + 1, stride):
                    split_data.append((video_path, start_idx, self.ds))
        
        return np.array(split_data)

    def _load_label_file(self, label_file):
        """
        Load labels from action_discrete.txt file
        
        Args:
            label_file: Path to the label file
            
        Returns:
            Numpy array of phase labels
        """
        labels = []
        with open(label_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:  # Ensure row has timestamp and label
                    timestamp, label = row
                    labels.append(int(label))
        
        return np.array(labels)

    def frame_sampler(self, videoname, frame_list):
        """
        Sample frames based on the specified parameters
        
        Args:
            videoname: Video information tuple (path, start_idx, downsampling)
            frame_list: List of frame filenames
            
        Returns:
            List of frame indices to use
        """
        start_idx = int(videoname[1])
        ds = int(videoname[2])
        
        # Generate sequence of frame indices
        seq_idx = np.arange(self.num_frames) * ds + start_idx
        seq_idx = np.clip(seq_idx, 0, len(frame_list) - 1)
        
        return seq_idx

    def __getitem__(self, index):
        """
        Get a video clip and its corresponding labels
        
        Args:
            index: Index of the video clip
            
        Returns:
            seq: Tensor containing video frames
            vid: Tensor containing phase labels
        """
        videoname = self.train_split[index]
        video_path = videoname[0]
        
        # Get the RGB frames directory
        rgb_dir = os.path.join(self.frame_dir, video_path, 'rgb')
        
        # Get the label file
        label_file = os.path.join(self.frame_dir, video_path, 'action_discrete.txt')
        
        # Load labels
        vlabel = self._load_label_file(label_file)
        
        # Get frame filenames and sort by timestamp
        frame_list = [f for f in os.listdir(rgb_dir) if f.endswith('.jpg')]
        frame_list.sort(key=lambda x: int(os.path.splitext(x)[0]))
        
        # Sample frames
        frame_index = self.frame_sampler(videoname, frame_list)
        
        # Load images
        seq = [Image.open(os.path.join(rgb_dir, frame_list[i])).convert('RGB') for i in frame_index]
        
        # Get labels corresponding to sampled frames
        vid = vlabel[frame_index]
        
        # Process labels for pretraining if required
        if self.pretrain:
            vid = torch.from_numpy(vid)
            vid = torch.unique_consecutive(vid)
            vid = vid.numpy()
            vid = np.pad(vid, (0, 10 - vid.shape[0]), 'constant', constant_values=(0, -1))
        
        # Apply transformations
        if self.transform is not None:
            seq = self.transform(seq)
        else:
            convert_tensor = transforms.ToTensor()
            seq = [convert_tensor(img) for img in seq]
            seq = torch.stack(seq)
        
        return seq, vid

    def __len__(self):
        """
        Get the total number of video clips
        
        Returns:
            Number of clips in the dataset split
        """
        return len(self.train_split)


class RARP50_FRAMES(data.Dataset):
    """
    Dataset class for extracting features from RARP50 frames
    This class is designed for feature extraction rather than training
    """
    def __init__(self,
                 root='./data/rarp50',
                 small_test=False,
                 frame_dir='./data/rarp50/',
                 save_feat_dir='rarp50_vit_features',
                 transform=None):
        self.root = root
        self.small_test = small_test
        self.frame_dir = frame_dir
        self.save_feat_dir = save_feat_dir
        self.transform = transform
        
        # Get all videos from both train and test sets
        self.data_lst = []
        
        # Process train videos
        train_dir = os.path.join(self.frame_dir, "train")
        if os.path.exists(train_dir):
            for video_dir in os.listdir(train_dir):
                if os.path.isdir(os.path.join(train_dir, video_dir)):
                    rgb_dir = os.path.join(train_dir, video_dir, 'rgb')
                    if os.path.exists(rgb_dir):
                        frames = [f for f in os.listdir(rgb_dir) if f.endswith('.jpg')]
                        frames.sort(key=lambda x: int(os.path.splitext(x)[0]))
                        for i, frame in enumerate(frames):
                            self.data_lst.append((os.path.join("train", video_dir, "rgb", frame), i))
        
        # Process test videos
        test_dir = os.path.join(self.frame_dir, "test")
        if os.path.exists(test_dir):
            for video_dir in os.listdir(test_dir):
                if os.path.isdir(os.path.join(test_dir, video_dir)):
                    rgb_dir = os.path.join(test_dir, video_dir, 'rgb')
                    if os.path.exists(rgb_dir):
                        frames = [f for f in os.listdir(rgb_dir) if f.endswith('.jpg')]
                        frames.sort(key=lambda x: int(os.path.splitext(x)[0]))
                        for i, frame in enumerate(frames):
                            self.data_lst.append((os.path.join("test", video_dir, "rgb", frame), i))

    def __getitem__(self, index):
        """
        Get a single frame for feature extraction
        
        Args:
            index: Index of the frame
            
        Returns:
            seq: Tensor containing the frame
            fname: Filename for saving the features
        """
        frame_path, frame_idx = self.data_lst[index]
        
        # Load image
        img = Image.open(os.path.join(self.frame_dir, frame_path)).convert('RGB')
        
        # Apply transformations
        if self.transform is not None:
            seq = self.transform([img])
        else:
            convert_tensor = transforms.ToTensor()
            seq = convert_tensor(img).unsqueeze(0)
        
        # Create a filename for saving the features
        # Format: split/video_name/frame_idx.npy
        parts = frame_path.split(os.sep)
        video_name = parts[1]
        frame_name = os.path.splitext(parts[3])[0]
        fname = f"{parts[0]}_{video_name}_{frame_name}.npy"
        
        return seq, fname

    def __len__(self):
        """
        Get the total number of frames
        
        Returns:
            Number of frames in the dataset
        """
        return len(self.data_lst)

if __name__ == '__main__':
    # train_dataset = Breakfast_FRAMES()
    train_dataset = Breakfast(ds=4, ol=2, n_split=5, mode='train')
    train_dataloader = data.DataLoader(train_dataset, batch_size=12, shuffle=True)
    acts = set()
    from tqdm import tqdm

    for i, (img, id_list) in enumerate(tqdm(train_dataloader)):
        print(img)
