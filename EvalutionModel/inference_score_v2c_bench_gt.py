import torch
import torch.nn.functional as F
from models.wrapper import CASPWrapper, CASP
import torch_tools
import os
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]
def plot_heatmap_topk(matrix, title="Heatmap", xlabel="Speech", ylabel="Audio", topk=5):
    # 将矩阵转换为 numpy 数组
    matrix_np = matrix.cpu().detach().numpy()
    
    # 创建一个与 matrix_np 形状相同的全零矩阵
    topk_matrix = np.zeros_like(matrix_np)
    
    # 找到每一行的 topk 值的索引
    topk_indices = np.argsort(matrix_np, axis=1)[:, -topk:]
    
    # 将 topk 值保留，其余值设为 NaN
    for i, indices in enumerate(topk_indices):
        topk_matrix[i, indices] = matrix_np[i, indices]
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(topk_matrix, cmap="YlGnBu", annot=False, fmt=".2f", linewidths=0.5, cbar=True, mask=(topk_matrix == 0))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig('output/'+title+f'topk-{topk}', dpi=300, bbox_inches='tight')  # 保存图片为300DPI的高质量图像

    
# 加载模型的检查点并恢复模型权重
def load_model_from_ckpt(model, ckpt_path, device):
    # 加载检查点
    checkpoint = torch.load(ckpt_path, map_location=device)
    print('checkpoint keys ', checkpoint.keys())
    model.load_state_dict(checkpoint['state_dict'])
    # ['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops']
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    model.to(device)
    model.eval() 
    return model

class Inference:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval() 

    def preprocess_input(self, audio_path, speech_path):

        audio_wav = torch_tools.read_wav_file(audio_path, int(5 * 102.4) * 160)  # 例如采样5秒
        speech_wav = torch_tools.read_wav_file(speech_path, int(5 * 102.4) * 160)  # 例如采样5秒
        
        audio_wav = audio_wav.to(self.device)
        speech_wav = speech_wav.to(self.device)
        
        return audio_wav, speech_wav

    def inference(self, audio_paths, speech_paths):
        
        speech_wavs = []
        audio_wavs = []
        
        for audio_path, speech_path in zip(audio_paths, speech_paths):
            audio_wav, speech_wav = self.preprocess_input(audio_path, speech_path)
            audio_wavs.append(audio_wav)
            speech_wavs.append(speech_wav)

        audio_wavs = torch.cat(audio_wavs, dim=0)
        speech_wavs = torch.cat(speech_wavs, dim=0)

        with torch.no_grad():
            logits_per_speech, logits_per_audio = self.model.model.inference_noscale(audio_wavs, speech_wavs)

        return logits_per_speech, logits_per_audio


model = CASPWrapper(d_model=768)

ckpt_path = '/root/logs/CASP/version_36/checkpoints/epoch=31-step=115000.ckpt'
print('ckpt_path', ckpt_path)
json_files = [
    '/root/v2cdataset/v2c-animation-test-filter2.jsonl'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载模型
model = load_model_from_ckpt(model, ckpt_path, device)

# 创建推理实例
inference_engine = Inference(model, device)
filenames = []

merged_data = []
for file in json_files:
    if os.path.exists(file):
        print('file: ', file)
        data = load_jsonl(file)
        merged_data.extend(data)
    else:
        print(f"文件 {file} 不存在，跳过。")

score = []


audio_paths, speech_paths = [], []
for item in tqdm(merged_data):
    logits_per_speech, logits_per_audio = inference_engine.inference([item['audio']], [item['speech']])
    score.append(logits_per_speech)
print('score: ', sum(score)/len(score))

