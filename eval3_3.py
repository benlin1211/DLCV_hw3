# Ref: https://github.com/zarzouram/image_captioning_with_transformers/tree/759476452229f9829be6576e5e6934296e4febe6/code/models/IC_encoder_decoder
# greedy

import timm
from copy import deepcopy

import urllib
import os
import glob
from PIL import Image
from torch.autograd import Variable

import torch
import matplotlib.pyplot as plt
from torch import Tensor
from typing import Tuple
import torch.nn as nn
from torch.nn import MultiheadAttention
from tokenizers import Tokenizer
# https://huggingface.co/docs/tokenizers/api/tokenizer
import json
import torch.nn.functional as F
import math,copy,re
import pandas as pd
import numpy as np
import random

from torch.utils.data import Dataset, DataLoader
import torchvision 
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse



"""
reference:
https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
"""
# seed setting
def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ImageVal(Dataset):
    def __init__(self, image_dir, transform):
        super().__init__()
        self.file_names = glob.glob(os.path.join(image_dir, "*.jpg"))  
        self.transform = transform

        # "[PAD]": 0,
        # "[UNK]": 1,
        # "[BOS]": 2,
        # "[EOS]": 3,

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # read image according to data["images"] list
        file_name = self.file_names[idx]
        #print(file_name)
        image = Image.open(file_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, file_name.split("/")[-1].split(".")[0]


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        param:
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        self.d_model = d_model

        # create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # not a parameter, but should be part of the modules state.
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int,
                 dropout: float):
        super(DecoderLayer, self).__init__()

        self.dec_self_attn = MultiheadAttention(d_model,
                                                num_heads,
                                                dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model,
                                                 num_heads,
                                                 dropout=dropout)

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.multihead_norm = nn.LayerNorm(d_model)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.multihead_dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(nn.Linear(d_model, feedforward_dim),
                                nn.ReLU(inplace=True), nn.Dropout(p=dropout),
                                nn.Linear(feedforward_dim, d_model))

        self.ff_norm = nn.LayerNorm(d_model)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, dec_inputs: Tensor, enc_outputs: Tensor,
                tgt_mask: Tensor,
                tgt_pad_mask: Tensor) -> Tuple[Tensor, Tensor]:

        # self attention + resedual summation + norm
        output, _ = self.dec_self_attn(dec_inputs,
                                       dec_inputs,
                                       dec_inputs,
                                       attn_mask=tgt_mask,
                                       key_padding_mask=tgt_pad_mask,
                                       average_attn_weights=False)
        output = dec_inputs + self.self_attn_dropout(output)
        output = self.self_attn_norm(output)  # type: Tensor

        # # self attention + residual + norm + FF
        output2, attns = self.multihead_attn(output, enc_outputs, enc_outputs, average_attn_weights=False)
        output = output + self.multihead_dropout(output2)
        output = self.multihead_norm(output)

        output2 = self.ff(output)  # type: Tensor
        output = self.ff_norm(output + self.ff_dropout(output2))

        return output, attns



class Decoder(nn.Module):

    def __init__(self,
                 layer: DecoderLayer,
                 vocab_size: int,
                 d_model: int,
                 num_layers: int,
                 max_len: int,
                 dropout: float,
                 pad_id: int):
        super().__init__()

        self.pad_id = pad_id

        # Embedding layer + pos encoding
        self.cptn_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = PositionalEncoding(d_model, max_len)

        # Make copies of the decoder layer
        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_layers)])

        self.dropout = nn.Dropout(p=dropout)

    def get_attn_subsequent_mask(self, sz: int) -> Tensor:

        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, tgt_cptn: Tensor,
                src_img: Tensor) -> Tuple[Tensor, Tensor]:

        # create masks, then pass to decoder
        tgt_pad_mask = (tgt_cptn == self.pad_id)
        tgt_mask = self.get_attn_subsequent_mask(tgt_cptn.size()[1])
        tgt_mask = tgt_mask.to(tgt_cptn.device)

        # encode captions + pos enc
        # (B, max_len) -> (B, max_len, d_model) -> (max_len, B, d_model)
        tgt_cptn = self.cptn_emb(tgt_cptn)  # type: Tensor
        tgt_cptn = self.dropout(self.pos_emb(tgt_cptn.permute(1, 0, 2)))

        attns_all = []
        for layer in self.layers:
            tgt_cptn, attns = layer(tgt_cptn, src_img, tgt_mask, tgt_pad_mask)
            attns_all.append(attns)
        # [layer_num, batch_size, head_num, max_len, encode_size**2]
        attns_all = torch.stack(attns_all)

        return tgt_cptn, attns_all


class Transformer(nn.Module):
    def __init__(self,
                 vocab_size,
                 d_model,
                 encoder_model_name,
                 dec_ff_dim,
                 dec_n_layers,
                 dec_n_heads,
                 max_len,
                 dropout = 0.1,
                 pad_id = 0):
        super(Transformer, self).__init__()
        decoder_layer = DecoderLayer(d_model=d_model,
                                     num_heads=dec_n_heads,
                                     feedforward_dim=dec_ff_dim,
                                     dropout=dropout)
        #print(timm.list_models("*vit*")) 
        #self.encoder = torch.nn.Sequential(*(list(timm.create_model(encoder_model_name, pretrained=True).children())[:-1]))
        self.encoder = timm.create_model(encoder_model_name, pretrained=True)
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.decoder = Decoder(layer=decoder_layer,
                               vocab_size=vocab_size,
                               d_model=d_model,
                               num_layers=dec_n_layers,
                               max_len=max_len,
                               dropout=dropout,
                               pad_id=pad_id)

        self.predictor = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, images: Tensor,
                captions: Tensor) -> Tuple[Tensor, Tensor]:

        # encode, decode, predict
        images_encoded = self.encoder.forward_features(images)  # type: Tensor
        tgt_cptn, attns = self.decoder(captions, images_encoded.permute(1,0,2))
        predictions = self.predictor(tgt_cptn).permute(1, 0, 2)  # type: Tensor

        return predictions.contiguous(), attns.contiguous()

def show_n_param(model):
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    unloader = transforms.ToPILImage()
    image = unloader(image)
    return image

# https://oldpan.me/archives/pytorch-tensor-image-transform

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hw 3-2 train",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--src_path", help="src_path", default="hw3_data/p3_data/images/") 
    parser.add_argument("--des_root", help="des_root", default="hw3/output_p3/") 
    parser.add_argument("--tokenizer_path", help="tokenizer location", default= "./hw3_data/caption_tokenizer.json")
    parser.add_argument("--dropout", help="dropout in encoder", type=int, default= 0.1)
    # ================================ EVAL ======================================    
    parser.add_argument("--ckpt_path", help="Checkpoint location", default= "./ckpt_encoder_continue")
    parser.add_argument("--resume_name", help="Checkpoint resume name", default= "epoch_3_best.pth")

    parser.add_argument("--model_option",  default= "vit_large_patch14_224_clip_laion2b") #"vit_base_resnet50_384"  "vit_base_patch14_224_clip_laion2b"
    parser.add_argument("--resize", help="resize", type=int, default=224)
    parser.add_argument("--n_heads", help="n_heads", type=int, default=16)
    parser.add_argument("--embed_dim", help="embed_dim", type=int, default=1024)
    parser.add_argument("--num_layers", help="num_layers", type=int, default=6) # actually 6
    parser.add_argument("--num_freeze_layer", help="num_freeze_layer in encoder", type=int, default=12)
    # ================================ EVAL ====================================== 
    args = parser.parse_args()
    print(vars(args))

    # model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)  # you can choose between v1, v2 and v3
    # print(model)
    same_seeds(1211)
    if torch.cuda.is_available():
        if torch.cuda.device_count()==2:
            device = torch.device("cuda:1")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # device = torch.device("cuda")

    print("Using", device)
    des_root = args.des_root
    os.makedirs(des_root, exist_ok=True)
    src_path = args.src_path
    
    tokenizer_path = args.tokenizer_path
    ckpt_path = args.ckpt_path
    resume_name = args.resume_name 

    encoder_model_name= args.model_option 
    num_layers = args.num_layers 
    num_freeze_layer = args.num_freeze_layer
    resize = args.resize
    batch_size = 1 # args.batch_size
    embed_dim = args.embed_dim
    n_heads = args.n_heads
    dropout = args.dropout

    # Tokenizer setting
    tokenizer = Tokenizer.from_file(tokenizer_path)
    target_vocab_size = len(tokenizer.get_vocab()) # vocab_size 18022
    seq_length = 196 # I don't know why...


    # Dataset
    val_transform = transforms.Compose([
        transforms.Resize((resize,resize)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # visualization
    attn_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    dataset_val = ImageVal(image_dir=src_path, 
                            transform=val_transform)

    data_loader_val = DataLoader(dataset_val, batch_size, shuffle=False, drop_last=False, num_workers=0)

    # Debugger
    len_dataloader_val = len(data_loader_val)
    #data_iter_val = iter(data_loader_val)
    
    print("val:", len_dataloader_val)

    # model
    model = Transformer(                 
                vocab_size=target_vocab_size,
                d_model=embed_dim,
                encoder_model_name=encoder_model_name,
                dec_ff_dim=embed_dim*4,
                dec_n_layers=num_layers,
                dec_n_heads=n_heads,
                max_len=seq_length,
                dropout = dropout,
                pad_id = 0)

    show_n_param(model)
    model = model.to(device)

    # Load 
    resume  = os.path.join(ckpt_path, resume_name)
    print(f"load from {resume}")
    checkpoint = torch.load(resume, map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    max_len = 52
    result = {}
    BOS=2
    EOS=3

    for data in tqdm(data_loader_val):
        
        # preprocessing 
        image, file_name = data 
        image = image.to(device)
        gen_seq = np.zeros((1, max_len)) # [bz, max_seq_len]
        gen_seq[:,0] = BOS  #[BOS]
        gen_seq = torch.tensor(gen_seq, dtype=torch.int64).to(device) 
        id_end = 1
        attn_list = []
        with torch.no_grad():
            for step in range(1, max_len):

                logits, attns = model(image, gen_seq[:, :step])
                # attns: [layer, 1, head, word, attn_map(16*16)+1]
                # get mean weight value of all heads
                attns = attns.mean(dim=2)
                # attns: [layer, 1, head, word, attn_map(16*16)+1]
                word_ids = torch.argmax(logits, 2)                
                next_word_id = word_ids[:,-1]
        
                # print("next_word_id", next_word_id)
                gen_seq[:, step] = next_word_id
                id_end = step+1
                # print(attns.shape)
                # print(attns[-1, 0, -1, :-1].shape)
                attn_list.append(attns[-1, 0, -1, :-1].reshape(16, 16).cpu())
                # print(tokenizer.decode(list(next_word_id.cpu())))
                if next_word_id==EOS:
                    break
        pred_ids = list(gen_seq.squeeze().cpu())
        reconstruct_caption = tokenizer.decode(pred_ids)
        # print("pred_ids",pred_ids)
        # reconstruct_caption = tokenizer.decode(pred_ids[:id_end])

        result[file_name[0]] = reconstruct_caption
        # print(file_name)
        # print(reconstruct_caption)
        # print(len(attn_list))
        # print(len(pred_ids))
        # print(attn_list[0].shape)

        ori_img = image[0].permute(1,2,0).clone().cpu()
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle(f"Attention Visualization of {file_name[0]}.jpg", fontsize=24)

        for i in range(0, len(pred_ids), 1):
            token = pred_ids[i]
            if token == 2:
                caption = "[BOS]"
            elif token == 3:
                caption = "[EOS]"
            else:
                caption = tokenizer.decode(list([token]))

            
            if i==0:
                # show origin
                ax = fig.add_subplot(3, 5, i+1)
                ax.imshow(ori_img)
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                ax.set_title(caption)
                pass
            else:
                ax = fig.add_subplot(3, 5, i+1)
                attn = attn_list[i-1]
                attn_map = attn_transform(tensor_to_PIL(attn)).permute(1,2,0)
                # print(caption)
                ax.imshow(attn_map, cmap='jet')
                ax.imshow(ori_img, alpha = 0.2)
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                ax.set_title(caption)

            if token == 3: #[EOS]
                break
        save_path = os.path.join(des_root, f"{file_name[0]}_vis.png")
        plt.savefig(save_path)