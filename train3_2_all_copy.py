# Ref: https://github.com/zarzouram/image_captioning_with_transformers/tree/759476452229f9829be6576e5e6934296e4febe6/code/models/IC_encoder_decoder


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

class ImageCaption(Dataset):
    def __init__(self, image_dir, annotation_json_dir, max_length, transform, tokenizer, mode='train'):
        super().__init__()
        self.image_dir = image_dir # read image according to caption list
        self.transform = transform
        with open(os.path.join(annotation_json_dir), newline='') as jsonfile:
            data = json.load(jsonfile)
        self.annotations = data["annotations"]
        self.images = data["images"]

        self.max_length = max_length + 1
        self.tokenizer = tokenizer
        self.mode = mode
        self.pad_token_id = 0
        # "[PAD]": 0,
        # "[UNK]": 1,
        # "[BOS]": 2,
        # "[EOS]": 3,

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # read image according to caption list
        # Get image_id
        image_id = self.annotations[idx]["image_id"]

        # Get caption
        caption = self.annotations[idx]["caption"]
    
        # Get image with corresponding caption id
        file_name = [item["file_name"] for item in self.images if item["id"] == image_id] 
        assert len(file_name) == 1
        image = Image.open(os.path.join(self.image_dir, file_name[0])).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, caption


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
        self.encoder = torch.nn.Sequential(*(list(timm.create_model(encoder_model_name, pretrained=True).children())[:-1]))
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
        images_encoded = self.encoder(images)  # type: Tensor
        tgt_cptn, attns = self.decoder(captions, images_encoded.permute(1,0,2))
        predictions = self.predictor(tgt_cptn).permute(1, 0, 2)  # type: Tensor

        return predictions.contiguous(), attns.contiguous()

def show_n_param(model):
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def loss_compute(criterion, pred, gth):
    # https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/train.py#L40 
    PAD = 0
    #loss = F.cross_entropy(pred, gth, ignore_index=PAD, label_smoothing=smoothing)# ,reduction='sum')

    v_sz = pred.size()[-1]
    gth = gth.contiguous()
    
    # print(pred.view(-1, v_sz))
    # print(gth.view(-1))

    loss = criterion(pred.view(-1, v_sz), gth.view(-1))
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hw 3-2 train",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", help="batch size", type=int, default=32)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", help="weight decay", type=float, default=0)
    parser.add_argument("--scheduler_warmup_steps", help="scheduler learning rate warmup step ", type=int, default=500)
    parser.add_argument("--gamma", help="learning rate decay factor.",type=float, default=0.9)
    parser.add_argument("--n_epochs", help="n_epochs", type=int, default=20) #6
    parser.add_argument("--smoothing", help="label smoothing factor", type=float, default=0.0)
    parser.add_argument("--dropout", help="dropout in encoder", type=int, default=0.1)
    # ================================= TRAIN =====================================                             
    parser.add_argument("--ckpt_path", help="Checkpoint location", default= "./ckpt_all_copy_no_aug") 
    # patch 越小越強
    parser.add_argument("--model_option",  default= "vit_large_patch14_224_clip_laion2b") #"vit_base_resnet50_384"  "vit_large_patch14_224_clip_laion2b" "vit_base_patch8_224"
    parser.add_argument("--resize", help="resize", type=int, default=224)
    parser.add_argument("--n_heads", help="n_heads", type=int, default=8)
    parser.add_argument("--embed_dim", help="embed_dim", type=int, default=1024) # 16*96
    parser.add_argument("--num_layers", help="num_layers", type=int, default=8)
    parser.add_argument("--num_freeze_layer", help="num_freeze_layer in encoder", type=int, default=12)
    # ================================= TRAIN ===================================== 

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

    print("Using", device)

    encoder_model_name= args.model_option 
    num_layers = args.num_layers 
    num_freeze_layer = args.num_freeze_layer
    resize = args.resize
    batch_size = args.batch_size
    embed_dim = args.embed_dim
    n_heads = args.n_heads
    dropout = args.dropout
    # Leaning rate
    lr = args.learning_rate
    weight_decay = args.weight_decay

    # loss smootthing
    smoothing = args.smoothing
    
    # Epoch
    epochs = args.n_epochs
    gradient_clip = 0.1
    # lr scheduler
    num_warmup_steps = args.scheduler_warmup_steps
    gamma = args.gamma

    root_dir = "./hw3_data"
    ckpt_path= args.ckpt_path
    os.makedirs(ckpt_path, exist_ok=True)


    # Tokenizer setting
    tokenizer = Tokenizer.from_file(os.path.join(root_dir, "caption_tokenizer.json"))
    
    target_vocab_size = len(tokenizer.get_vocab()) # vocab_size 18022
    seq_length = 196 # I don't know why...

    # Dataset
    train_transform = transforms.Compose([
        # transforms.Lambda(under_max),

        transforms.Resize((resize,resize)),
        #transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[0.8, 1.5], saturation=[0.2, 1.5]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((resize,resize)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data_dir = os.path.join(root_dir,"p2_data") # "./hw3_data/p2_data"
    # TODO: set the following into Dataset.
    image_dir_train = os.path.join(data_dir,"images/train") # "./hw3_data/p2_data/images/train"
    annotation_dir_train = os.path.join(data_dir,"train.json") # "./hw3_data/p2_data/train.json"
    image_dir_val = os.path.join(data_dir,"images/val")     # "./hw3_data/p2_data/images/val"
    annotation_dir_val = os.path.join(data_dir,"val.json")     # "./hw3_data/p2_data/val.json"

    max_pos_emb = 128
    dataset_train = ImageCaption(image_dir=image_dir_train, 
                                annotation_json_dir=annotation_dir_train, 
                                max_length=max_pos_emb, 
                                transform=train_transform, 
                                tokenizer=tokenizer,
                                mode="train")
    dataset_val = ImageCaption(image_dir=image_dir_val, 
                                annotation_json_dir=annotation_dir_val, 
                                max_length=max_pos_emb, 
                                transform=val_transform,
                                tokenizer=tokenizer,
                                mode="validate")


    data_loader_train = DataLoader(dataset_train, batch_size, shuffle=True, num_workers=0)
    data_loader_val = DataLoader(dataset_val, batch_size, shuffle=False, drop_last=False, num_workers=0)

    # Debugger
    len_dataloader_train = len(data_loader_train)
    len_dataloader_val = len(data_loader_val)

    print("train:", len_dataloader_train)
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

    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=smoothing, reduction='mean')
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, len(data_loader_train)*epochs)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=gamma)
    
    start_epoch = 0
    loss_curve_train = []
    loss_curve_val = []
    # # Load 
    # resume  = os.path.join(ckpt_path, f"epoch_0_best.pth")
    # checkpoint = torch.load(resume, map_location = device)
    # print(f"Load from {resume}")

    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # start_epoch = checkpoint['epoch'] + 1 


    for epoch in range(start_epoch, epochs):
        # ========================= Train ==========================
        model.train()
        print("Train")
        pbar = tqdm(data_loader_train)
        pbar.set_description(f"Epoch {epoch}")
        for data in pbar:
            #data = data_iter_train.next()
            image, captions = data    
            image = image.to(device)
            # print(image.shape)
            # print(captions)

            # preprocessing
            tokenized_captions = tokenizer.encode_batch(captions)
            tokenized_ids = torch.tensor([c.ids for c in tokenized_captions])
            tokenized_ids = tokenized_ids.to(device)
            # Shift
            # gth_ids = torch.zeros(tokenized_ids.shape, dtype=torch.int64).to(device)
            # gth_ids[:,:-1] = tokenized_ids[:,1:]

            
            logits, attns  = model(image, tokenized_ids[:, :-1])

            # print("logits", logits.shape)
            # print("logits", logits)
            # print("in", tokenized_ids)
            # print("gth",gth_ids)

            # print("tokenized_ids",  tokenized_ids[:, 1:].shape)
            # print("tokenized_ids[]",  tokenized_ids.shape)
            # print("logits", logits.shape)
            loss = loss_compute(criterion, logits, tokenized_ids[:, 1:])
            
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            #optimizer = clip_gradient(optimizer)
            optimizer.step()
            lr_scheduler.step()

            # Record
            pbar.set_postfix(loss=loss.item(), lr = optimizer.param_groups[0]['lr'])
            loss_curve_train.append(loss.item())

        # ========================= Eval: how to do ? ==========================
        model.eval()
        print("Eval")
        pbar_val = tqdm(data_loader_val)
        pbar_val.set_description(f"Epoch {epoch}")
        val_loss = 0
        with torch.no_grad():
            for data in pbar_val:
                image, captions = data 
                image = image.to(device)
                # preprocessing 
                tokenized_captions = tokenizer.encode_batch(captions)
                tokenized_ids = torch.tensor([c.ids for c in tokenized_captions])
                tokenized_ids = tokenized_ids.to(device)
                
                logits, attn = model(image,  tokenized_ids[:, :-1]) # trg_mask, padding_masks)
                loss = loss_compute(criterion, logits, tokenized_ids[:, 1:])

                val_loss += loss.item()
                pbar_val.set_postfix(loss=loss.item())
                loss_curve_val.append(loss.item())
                

        if True: #best_clip_score <= clip_score:
            #best_clip_score = clip_score
            save_as = os.path.join(ckpt_path, f"epoch_{epoch}_best.pth")
            print(f"Saving emodel at {save_as}")
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    }, save_as)
        


    print("OK")

    df = pd.DataFrame() # apply pd.DataFrame format 
    df["loss_train"] = loss_curve_train
    csv_name = os.path.join(ckpt_path, f"loss_train.csv")
    df.to_csv(csv_name, index = False)


    df = pd.DataFrame() # apply pd.DataFrame format 
    df["loss_val"] = loss_curve_val
    csv_name = os.path.join(ckpt_path, f"loss_val.csv")
    df.to_csv(csv_name, index = False)