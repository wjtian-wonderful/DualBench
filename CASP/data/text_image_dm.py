# Originally found in https://github.com/lucidrains/DALLE-pytorch

import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torch_tools
import os
import json
import librosa
import torch
import torchaudio
import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


class TextImageDataset(Dataset):
    def __init__(self, ):

        super().__init__()

        json_files = [

            '/root/v2cdataset/0409-v2c-animation-train-filter3.jsonl'

        ]
        self.audiospeech_data = []
        for file in json_files:
            if os.path.exists(file):
                print('file: ', file)
                data = load_jsonl(file)
                self.audiospeech_data.extend(data)
            else:
                print(f"文件 {file} 不存在，跳过。")
        
        import random
        random.shuffle(self.audiospeech_data)
        
        
        self.window_length = 1
        

    def __len__(self):
        return len(self.audiospeech_data)
    
    def __getitem__(self, ind):

        audio_speech = self.audiospeech_data[ind]
        
        audio_path = audio_speech['audiopath']
        speech_path = audio_speech['speechpath']

        audio_wav, audio_sr = librosa.load(audio_path, sr=None, mono=False)# sr=None 保持原始采样率 默认是单通道
        speech_wav, speech_sr = librosa.load(speech_path, sr=None, mono=False)# sr=None 保持原始采样率 默认是单通道

        
        if  audio_wav.shape[1] != speech_wav.shape[1]:
            audio_wav, audio_sr = librosa.load(audio_path, sr=None, mono=False, duration=self.window_length)# sr=None 保持原始采样率 默认是单通道
            speech_wav, speech_sr = librosa.load(speech_path, sr=None, mono=False, duration=self.window_length)# sr=None 保持原始采样率 默认是单通道

            
        total_samples = audio_wav.shape[1]
        desired_samples = audio_sr * self.window_length
        # 如果音频短于所需的段长度，则填充它
        if total_samples > desired_samples:
            # 随机选取起始点
            start_sample = random.randint(0, total_samples - desired_samples)
            audio_wav = audio_wav[:, start_sample:start_sample + desired_samples]
            speech_wav = speech_wav[:, start_sample:start_sample + desired_samples]
        
        audio_wav = torch_tools.read_wav_file_loaded(audio_wav, audio_sr, int(self.window_length * 102.4) * 160) # hop size is 160 并不是1k进制，而是1024进制
        speech_wav = torch_tools.read_wav_file_loaded(speech_wav, speech_sr, int(self.window_length * 102.4) * 160) # hop size is 160 并不是1k进制，而是1024进制

        return audio_wav, speech_wav

class TextImageDataModule(LightningDataModule):
    def __init__(self,
                 batch_size = 48,
                 num_workers = 16,
                 shuffle=False,
                 ):

        super().__init__()
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.setup()
    
    
    def setup(self):
        self.dataset = TextImageDataset()
    
    def train_dataloader(self):
        return DataLoader(
                            self.dataset,
                            batch_size=self.batch_size,
                            shuffle=self.shuffle,
                            num_workers=self.num_workers,
                            drop_last=True,
                            collate_fn=self.dl_collate_fn,
                            pin_memory=True,
                            )
    
    def dl_collate_fn(self, batch):

        return torch.cat([row[0] for row in batch], dim=0), torch.cat([row[1] for row in batch], dim=0)
    
    
