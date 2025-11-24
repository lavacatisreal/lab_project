# model_loader.py
import torch
import torch.nn as nn
import torch.optim as optim
from model.transformer import Transformer
from model.clip import CLIP
from config import config


def build_main_model():
    print('\033[32m[info] Moving model to cuda...\033[0m')
    model = Transformer().cuda()
    model = nn.DataParallel(model)
    if config['resume_model'].strip() != '':
        print('\033[34m[info] loading recorded model ...\033[0m')
        # model.load_state_dict(torch.load(config['resume_model'], weights_only=True))
        #---
        pretrained_dict = torch.load(config['resume_model'])
        model_dict = model.state_dict()

        # 過濾只載入尺寸匹配的參數
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        #---
    return model

# def build_clip_model(vocab_size):
#     print('\033[32m[info] loading CLIP-model (radical model or pre-train model)...\033[0m')
#     clip_model = CLIP(
#         embed_dim=2048,
#         context_length=30,
#         # vocab_size=979,
#         vocab_size=vocab_size,
#         transformer_width=512,
#         transformer_heads=8,
#         transformer_layers=12
#     ).cuda()
#     clip_model = nn.DataParallel(clip_model)
#     clip_model.load_state_dict(torch.load(config['pre-train_model'], weights_only=True), strict=False)
#     return clip_model

def build_clip_model(vocab_size):
    print('\033[32minfo loading CLIP-model (radical model or pre-train model)...\033[0m')
    clip_model = CLIP(embeddim=2048, context_length=30, vocab_size=vocab_size, transformer_width=512, transformer_heads=8, transformer_layers=12).cuda()
    clip_model = nn.DataParallel(clip_model)
    
    if config['pre-train_model'].strip():
        pretrained_dict = torch.load(config['pre-train_model'], map_location='cuda')
        model_dict = clip_model.state_dict()
        
        # 過濾尺寸匹配的權重
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        
        # 更新權重字典
        model_dict.update(pretrained_dict)
        clip_model.load_state_dict(model_dict, strict=False)  # strict=False 忽略缺失權重
    
    return clip_model


def build_optimizer_scheduler(model):
    print('\033[32m[info] Determining the optimizer and scheduler...\033[0m')
    optimizer = optim.Adadelta(model.parameters(), lr=config['lr'], rho=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1
    )
    return optimizer, scheduler
