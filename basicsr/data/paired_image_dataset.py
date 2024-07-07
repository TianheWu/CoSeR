from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file, paired_paths_from_meta_info_file_2
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import cv2
import os
import torch.nn.functional as F
import pandas as pd
import random
import numpy as np
import warnings
import pickle


@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file_2([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        h, w = img_gt.shape[0:2]
        # pad
        if h < self.opt['gt_size'] or w < self.opt['gt_size']:
            pad_h = max(0, self.opt['gt_size'] - h)
            pad_w = max(0, self.opt['gt_size'] - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            img_lq = cv2.copyMakeBorder(img_lq, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)

@DATASET_REGISTRY.register()
class PairedImageCapRefDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageCapRefDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = []
        if 'meta_info' in opt:
            with open(self.opt['meta_info']) as fin:
                for line in fin.readlines():
                    self.paths.append(line.rstrip('\n'))
        else:
            self.paths = os.listdir(opt['dataroot_lq'])

        print(f"Total val set size is {len(self.paths)}")

        if 'caption_path' in opt:
            df = pd.read_json(opt['caption_path'])
            df.set_index(["filename"], inplace=True)
            self.caption_df = df
            df = pd.read_table('data/ImageNet/class.txt', sep='\t', header=None)
            df.set_index([1], inplace=True)
            self.class_df = df
            self.use_caption = True
            self.only_class = False
        elif opt['only_class']:
            df = pd.read_table('data/ImageNet/class.txt', sep='\t', header=None)
            df.set_index([1], inplace=True)
            self.class_df = df
            self.use_caption = True
            self.only_class = True
        else:
            self.use_caption = False

        # reference
        with open(opt['reference_path'], 'rb') as f:
            self.reference_sim = pickle.load(f)
        self.reference_select_num = opt['reference_select_num']

        self.drop_rate = opt['drop_rate']
        self.ref_drop_rate = opt['ref_drop_rate'] if 'ref_drop_rate' in opt else 0.0

    def generate_caption(self, image_name, clip_thre=0.28, class_name_score=0.5):
        caption_list = []
        prob_list = []
        caption_data = self.caption_df.loc[image_name]
        class_name = self.class_df.loc[image_name.split('_')[0], 2].replace('_', ' ')
        base_caption = 'a photo of ' + class_name
        caption_list.append(base_caption)
        prob_list.append(class_name_score)

        all_caption = [base_caption]
        all_prob = []

        if caption_data['clip_score1'] >= clip_thre:
            caption_list.append(base_caption + ', ' + caption_data['caption1'])
            all_caption.append(caption_data['caption1'])
            prob_list.append(caption_data['clip_score1'])
            all_prob.append(caption_data['clip_score1'])

        if caption_data['clip_score2'] >= clip_thre:
            caption_list.append(base_caption + ', ' + caption_data['caption2'])
            all_caption.append(caption_data['caption2'])
            prob_list.append(caption_data['clip_score2'])
            all_prob.append(caption_data['clip_score2'])

        if caption_data['clip_score3'] >= clip_thre:
            caption_list.append(base_caption + ', ' + caption_data['caption3'])
            all_caption.append(caption_data['caption3'])
            prob_list.append(caption_data['clip_score3'])
            all_prob.append(caption_data['clip_score3'])

        if all_prob != []:
            caption_list.append(', '.join(all_caption))
            prob_list.append(np.mean(all_prob))

        assert len(caption_list) == len(prob_list)
        prob_list = np.array(prob_list) / np.sum(prob_list)

        return caption_list, prob_list

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = os.path.join(self.gt_folder, self.paths[index])
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = os.path.join(self.lq_folder, self.paths[index])
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # generate caption
        image_name = gt_path.split('/')[-1]
        class_num = image_name.split('_')[0]
        if self.use_caption and (random.random() > self.drop_rate):
            if not self.only_class:
                caption_list, prob_list = self.generate_caption(image_name)
                caption = np.random.choice(caption_list, p=prob_list)
            else:
                class_name = self.class_df.loc[class_num, 2].replace('_', ' ')
                caption = 'a photo of ' + class_name
        else:
            caption = ""

        if random.random() > self.drop_rate:
            ref_filenames = self.reference_sim[class_num]['filename']
            ref_filenames_numpy = np.array(self.reference_sim[class_num]['filename'])
            ref_self_index = ref_filenames.index(image_name)
            ref_sim = self.reference_sim[class_num]['loss'][ref_self_index]

            ref_filenames_numpy = np.delete(ref_filenames_numpy, [ref_self_index])
            ref_sim = np.delete(ref_sim, [ref_self_index])
            
            sortindex = np.argsort(ref_sim)
            ref_filenames_selected = ref_filenames_numpy[sortindex][-self.reference_select_num:]
            ref_sim_selected = ref_sim[sortindex][-self.reference_select_num:]

            ref_filename_selected = np.random.choice(ref_filenames_selected, p=ref_sim_selected / np.sum(ref_sim_selected))

            img_bytes = self.file_client.get(os.path.join(self.gt_folder, ref_filename_selected), 'gt')
            img_ref = imfrombytes(img_bytes, float32=True)
            img_ref = img2tensor([img_ref], bgr2rgb=True, float32=True)[0]
        else:
            img_ref = np.zeros(img_gt.shape)
            img_ref = img2tensor([img_ref], bgr2rgb=True, float32=True)[0]

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, 'caption': caption, 'ref': img_ref}

    def __len__(self):
        return len(self.paths)
