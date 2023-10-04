import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import pickle

import cv2
from PIL import Image
from torchvision import transforms

from config import cfg
import pdb

# import torchaudio


def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL


def load_image_in_cv2_to_Tensor(path, mode='RGB', transform=None):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('test.png', image)
    return image

def load_audio_lm(audio_lm_path):
    with open(audio_lm_path, 'rb') as fr:
        audio_log_mel = pickle.load(fr)
    audio_log_mel = audio_log_mel.detach()# [5, 1, 96, 64]
    return audio_log_mel


def load_image_embedding_to_Tensor(path):
    if os.path.exists(path):
        img_embedding = torch.load(path)
    else:
        raise ValueError("Path {} does not exist".format(path))
    return img_embedding

class MS3Dataset(Dataset):
    """Dataset for multiple sound source segmentation"""
    def __init__(self, split='train'):
        super(MS3Dataset, self).__init__()
        self.split = split
        self.mask_num = 5
        df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
        self.df_split = df_all[df_all['split'] == split]
        print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all), self.split))
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])



    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        video_name = df_one_video[0]
        img_base_path =  os.path.join(cfg.DATA.DIR_IMG, video_name)
        audio_lm_path = os.path.join(cfg.DATA.DIR_AUDIO_LOG_MEL, self.split, video_name + '.pkl')
        mask_base_path = os.path.join(cfg.DATA.DIR_MASK, self.split, video_name)
        audio_log_mel = load_audio_lm(audio_lm_path)
        # audio_lm_tensor = torch.from_numpy(audio_log_mel)
        imgs, masks = [], []
        for img_id in range(1, 6):
            img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s.mp4_%d.png"%(video_name, img_id)), transform=self.img_transform)
            imgs.append(img)
        for mask_id in range(1, self.mask_num + 1):
            mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png"%(video_name, mask_id)), transform=self.mask_transform, mode='P')
            masks.append(mask)
        imgs_tensor = torch.stack(imgs, dim=0)
        masks_tensor = torch.stack(masks, dim=0)

        return imgs_tensor, audio_log_mel, masks_tensor, video_name

    def __len__(self):
        return len(self.df_split)

class MS3Dataset_SAM(Dataset):
    """Dataset for multiple sound source segmentation"""
    def __init__(self, split='train', get_img_embedding=True, audio_type='mel'):
        super(MS3Dataset_SAM, self).__init__()
        self.split = split
        self.mask_num = 5
        df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
        self.df_split = df_all[df_all['split'] == split]
        print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all), self.split))
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            # transforms.Resize((1024, 1024), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        self.get_img_embedding = get_img_embedding
        self.audio_type = audio_type

    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        video_name = df_one_video[0]
        img_base_path =  os.path.join(cfg.DATA.DIR_IMG, video_name)
        img_embed_base_path = os.path.join(cfg.DATA.DIR_IMG_EMBED, video_name)
        audio_lm_path = os.path.join(cfg.DATA.DIR_AUDIO_LOG_MEL, self.split, video_name + '.pkl')
        mask_base_path = os.path.join(cfg.DATA.DIR_MASK, self.split, video_name)
        audio_log_mel = load_audio_lm(audio_lm_path)
        
        # if self.audio_type == 'wav':
            # audio_log_mel = torchaudio.load(os.path.join(cfg.DATA.DIR_AUDIO_WAV, self.split, video_name + '.wav'))[0]
        # audio_lm_tensor = torch.from_numpy(audio_log_mel)
        imgs, img_embeddings, masks = [], [], []
        for img_id in range(1, 6):
            img = load_image_in_cv2_to_Tensor(os.path.join(img_base_path, "%s.mp4_%d.png"%(video_name, img_id)), transform=self.img_transform)
            imgs.append(img)
            if self.get_img_embedding:
                img_embedding = load_image_embedding_to_Tensor(os.path.join(img_embed_base_path, "%s_%d.pt"%(video_name, img_id)))
                img_embeddings.append(img_embedding)
        for mask_id in range(1, self.mask_num + 1):
            mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png"%(video_name, mask_id)), transform=self.mask_transform, mode='P')
            masks.append(mask)
        imgs_tensor = np.stack(imgs, axis=0)
        masks_tensor = torch.stack(masks, dim=0)
        
        if self.get_img_embedding:
            img_embeddings = torch.stack(img_embeddings, dim=0)
            return imgs_tensor, img_embeddings, audio_log_mel, masks_tensor, video_name
        else:
            return imgs_tensor, audio_log_mel, masks_tensor, video_name

    def __len__(self):
        return len(self.df_split)


if __name__ == "__main__":
    train_dataset = MSSSDataset('train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                     batch_size=2,
                                                     shuffle=False,
                                                     num_workers=8,
                                                     pin_memory=True)

    for n_iter, batch_data in enumerate(train_dataloader):
        imgs, audio, mask, video_name = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
        # imgs, audio, mask, video_name = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
        pdb.set_trace()
    print('n_iter', n_iter)
    pdb.set_trace()