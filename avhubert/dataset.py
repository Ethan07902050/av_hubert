import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from python_speech_features import logfbank
from tqdm import tqdm

def stacker(feats, stack_order=4):
    """
    Concatenating consecutive audio frames
    Args:
    feats - numpy.ndarray of shape [T, F]
    stack_order - int (number of neighboring frames to concatenate
    Returns:
    feats - numpy.ndarray of shape [T', F']
    """
    feat_dim = feats.shape[1]
    if len(feats) % stack_order != 0:
        res = stack_order - len(feats) % stack_order
        res = np.zeros([res, feat_dim]).astype(feats.dtype)
        feats = np.concatenate([feats, res], axis=0)
    feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
    return feats


class ClassificationDataset(Dataset):
    def __init__(self, root_dir, mode="train", random_seed=777):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.npz')]
        self.mode = mode # train / test
        if self.mode != 'test':
            random.seed(random_seed)
            random.shuffle(self.files)

    def __len__(self):
        if self.mode == 'train':
            return round(len(self.files) * 0.9)
        elif self.mode == 'val':
            return round(len(self.files) * 0.1)
        else:
            return len(self.files)

    def __getitem__(self, index):
        if self.mode == "val":
            index = index + round(len(self.files) * 0.9)

        filepath = self.files[index]
        data = np.load(os.path.join(self.root_dir, filepath))
        image, audio, is_empty = data['image'], data['audio'], data['is_empty']

        # process image frames
        full_image = np.zeros((len(is_empty), 3, 96, 96)).astype(np.float32)
        k = 0
        for index, empty in enumerate(is_empty):
            if not empty:
                full_image[index] = image[k]
                k += 1

        if audio.shape[0] != 0:
            input_length = audio.shape[0]

            # add noise
            # ns = np.random.choice([0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006]) 
            # if self.mode == 'train' and ns:
            #     wn = np.random.randn(input_length)
            #     audio = audio + ns*wn

            audio_feats = logfbank(audio, samplerate=16000, winstep=0.008, nfft=1024).astype(np.float32) # [T, F]
            audio_feats = stacker(audio_feats) # [T/stack_order_audio, F*stack_order_audio]

            # align the length of video and audio
            diff = len(audio_feats) - len(full_image)
            if diff < 0:
                audio_feats = np.concatenate([audio_feats, np.zeros([-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype)])
            elif diff > 0:
                audio_feats = audio_feats[:-diff]

            # normalization
            audio_feats = torch.from_numpy(audio_feats)
            with torch.no_grad():
                audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])

        else:
            audio_feats = np.zeros((full_image.shape[0], 104)).astype(np.float32)
            audio_feats = torch.from_numpy(audio_feats)

        rgb_video = torch.from_numpy(full_image)
        grayscale_video = transforms.functional.rgb_to_grayscale(rgb_video)
        
        # if self.mode == 'train':
        #     grayscale_video = self.video_transform(grayscale_video)

        if self.mode != 'test':
            ttm = data['ttm']
            return grayscale_video, audio_feats, audio_feats.shape[0], torch.from_numpy(ttm).float()
        else:
            filename = filepath.split('.')[0]
            return grayscale_video, audio_feats, audio_feats.shape[0], filename

    # function to collate data samples into batch tesors
    def collate_fn(self, batch):
        video_batch = [b[0] for b in batch]
        audio_batch = [b[1] for b in batch]
        length_batch = [b[2] for b in batch] 

        video_batch = pad_sequence(video_batch, batch_first=True)
        video_batch = video_batch.permute((0, 2, 1, 3, 4))

        audio_batch = pad_sequence(audio_batch, batch_first=True)
        audio_batch = audio_batch.permute((0, 2, 1))
        
        if self.mode != 'test':
            label_batch = [b[3] for b in batch]
            label_batch = torch.stack(label_batch)
            data = {
                'video': video_batch,
                'audio': audio_batch,
                'length': length_batch,
                'label': label_batch
            }
        else:
            fname_batch = [b[3] for b in batch]
            data = {
                'video': video_batch,
                'audio': audio_batch,
                'length': length_batch,
                'fname': fname_batch
            }

        return data