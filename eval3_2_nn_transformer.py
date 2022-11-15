# Ref: 
# https://rwightman.github.io/pytorch-image-models/models/vision-transformer/
# https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch/notebook

import timm
import urllib
import os
import glob
from PIL import Image
from torch.autograd import Variable

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
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

        # # Tokenize the caption
        # tokenized_caption = self.tokenizer.encode(caption, padding=True)
        # tokenized_caption_ids = torch.tensor(tokenized_caption.ids)

        # cap_mask = (
        #     1 - np.array(tokenized_caption.attention_mask)).astype(bool)
        # print(file_name)
        # print(caption)
        return image, caption


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
    def forward(self, x):
        out = self.embed(x)
        return out


class PositionalEmbedding(nn.Module):
    def __init__(self,max_seq_len, embed_model_dim, dropout = 0):
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i        
        pe = torch.zeros(max_seq_len,self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0,self.embed_dim,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        #add constant to embedding
        seq_length = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_length], requires_grad=False)
        x = self.dropout(x)
        return x


class VisualTransformer(nn.Module):
    def __init__(self, embed_dim, encoder_model_name, target_vocab_size, seq_length ,num_layers=12, expansion_factor=4, n_heads=8):
        super(VisualTransformer, self).__init__()
        
        """  
        Args:
           embed_dim:  dimension of embedding 
           # src_vocab_size: vocabulary size of source (we don't have it in ViT.)
           encoder_model_name: timm pretrain model
           target_vocab_size: vocabulary size of target
           seq_length : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention
        
        """

        self.word_embedding = nn.Embedding(target_vocab_size, embed_dim)
        self.position_embedding = PositionalEmbedding(seq_length, embed_dim)
        self.target_vocab_size = target_vocab_size

        #self.encoder = TransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads)
        self.encoder = torch.nn.Sequential(*(list(timm.create_model(encoder_model_name, pretrained=True).children())[:-1]))
        #self.encoder = timm.create_model('vit_base_patch16_224', pretrained=True)
        
        #print(self.encoder)
        dim_feedforward = embed_dim*expansion_factor

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=n_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.linear = nn.Linear(embed_dim ,self.target_vocab_size)

    def decode(self, img, max_seq_len, device):
        BOS = 2
        EOS = 3
        bz = img.size(0)
        # ref: https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/Translator.py#L9
        
        # init:
        enc_out = self.encoder(img)
        ans_idx = 0   # default
        gen_seq = np.zeros((bz, max_seq_len)) # [bz, max_seq_len]
        gen_seq[:,0] = BOS  #[BOS]
        gen_seq = torch.tensor(gen_seq, dtype=torch.int64).to(device) 
        
        # Greedy
        # start: [BOS] 
        for step in range(1, max_seq_len):    # decode up to max length 
            # print("step",step)
            # print(gen_seq[:, :step])
            tgt = self.word_embedding(gen_seq[:, :step]) 
            tgt = self.position_embedding(tgt)

            outputs = self.decoder(tgt=tgt.permute(1,0,2), memory=enc_out.permute(1,0,2), tgt_mask=None, tgt_key_padding_mask=None)
            outputs = self.linear(outputs)
            outputs = outputs.permute(1,2,0) # torch.Size([bz, target_vocab_size, tgt_len])
            # print(outputs.shape)
            # print(outputs)
            # print(torch.amax(outputs,1))
            best_k_idx = torch.argmax(outputs,1)
            #print("best_k_idx", best_k_idx)
            gen_seq[:, 1:step+1] = best_k_idx
            #print(outputs.shape)

            # Locate the position of [EOS]
            # eos_locs = gen_seq == EOS #[EOS] 

            # assert bz = 1
            if gen_seq[:, step]==EOS:
                break
            # if step > 4:
            #     break
        return gen_seq
        
    def forward(self, img, tgt, trg_mask, padding_masks):
        tgt = self.word_embedding(tgt) 
        tgt = self.position_embedding(tgt)

        enc_out = self.encoder(img)
        # print("enc_out",enc_out.shape)   # torch.Size([1, 196, 768])
        # print("tgt",tgt.shape)           # torch.Size([1, tgt.shape[1]])

        # print("trg_mask",trg_mask.shape) # torch.Size([tgt.shape[1], tgt.shape[1]])
        # print("trg_mask",trg_mask)
        # print("padding_masks", padding_masks.shape)
        # print("padding_masks", padding_masks)
        
        # (tgt_len, batch, embed_dim)
        outputs = self.decoder(tgt=tgt.permute(1,0,2), memory=enc_out.permute(1,0,2), tgt_mask=trg_mask, tgt_key_padding_mask=padding_masks)
        outputs = self.linear(outputs)
        outputs = outputs.permute(1,2,0) # torch.Size([bz, tgt_len, target_vocab_size])
        return outputs


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
# 版权声明：本文为CSDN博主「旺旺棒棒冰」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/ltochange/article/details/116524264

def show_n_param(model):
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hw 3-2 train",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckpt_path", help="Checkpoint location", default= "./ckpt3_2")
    parser.add_argument("--batch_size", help="batch size", type=int, default=1)
    parser.add_argument("--model_option", default="vit_base_patch16_224") #"vit_base_resnet50_384" 
    parser.add_argument("--resize", help="resize", type=int, default=224)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", help="weight decay", type=float, default=0)
    parser.add_argument("--scheduler_warmup_steps", help="scheduler learning rate warmup step ", type=int, default=2000)
    parser.add_argument("--gamma", help="learning rate decay factor.",type=float, default=0.9)
    parser.add_argument("--n_epochs", help="n_epochs", type=int, default=250)
    parser.add_argument("--num_layers", help="num_layers", type=int, default=6)
    parser.add_argument("--smoothing", help="label smoothing factor", type=float, default=0.1)


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
    device = torch.device("cuda")

    print("Using", device)

    encoder_model_name= args.model_option 
    num_layers = args.num_layers 
    resize = args.resize
    batch_size = args.batch_size
    # Leaning rate
    lr = args.learning_rate
    weight_decay = args.weight_decay

    # loss smootthing
    smoothing = args.smoothing
    
    # Epoch
    epochs = args.n_epochs
    weight_clip = 1
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

    # with open("./hw3_data/p2_data/train.json", newline='') as jsonfile:
    #     data = json.load(jsonfile)

    # example_caption = data["annotations"][0]["caption"]
    # print("example_caption", example_caption)
    # tokenized_caption = tokenizer.encode(example_caption)
    # tokenized_id = list(torch.tensor(tokenized_caption.ids))
    # print("id", tokenized_id)
    # reconstruct_caption = tokenizer.decode(tokenized_id)
    # print("reconstruct_caption", reconstruct_caption)


    # Dataset
    train_transform = transforms.Compose([
        # transforms.Lambda(under_max),

        transforms.Resize((resize,resize)),
        transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[0.8, 1.5], saturation=[0.2, 1.5]),
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
    # sampler_train = torch.utils.data.RandomSampler(dataset_train)
    # batch_sampler_train = torch.utils.data.BatchSampler(
    #     sampler_train, batch_size, drop_last=True
    # )
    # sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = DataLoader(dataset_train, batch_size, shuffle=True, num_workers=0)
    data_loader_val = DataLoader(dataset_val, batch_size, shuffle=False, drop_last=False, num_workers=0)

    # Debugger
    len_dataloader_train = len(data_loader_train)
    #data_iter_train = iter(data_loader_train)
    len_dataloader_val = len(data_loader_val)
    #data_iter_val = iter(data_loader_val)
    
    pred_list = []
    filename_list = []
    print("train:", len_dataloader_train)
    print("val:", len_dataloader_val)

    # model
    model = VisualTransformer(embed_dim=768, encoder_model_name=encoder_model_name, 
                              target_vocab_size=target_vocab_size, seq_length=seq_length)
    show_n_param(model)
    model = model.to(device)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, len(data_loader_train)*epochs)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, drop_step, gamma)
    criterion = nn.CrossEntropyLoss() 
    
    #model.eval()
    best_loss = 20
    loss_curve_train = []
    loss_curve_val = []

    # Load 

    resume  = os.path.join(ckpt_path, f"epoch_0.pth")
    checkpoint = torch.load(resume, map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    val_loss = 0

    max_seq_len = 100
    with torch.no_grad():
        for i, data in enumerate(data_loader_val):
            #data = data_iter_val.next()
            image, _caption = data 
            print(_caption)
            image = image.to(device)
            # preprocessing 

            pred_ids = model.decode(image, max_seq_len, device)

            for p in pred_ids:
                p = list(p.cpu())
                reconstruct_caption = tokenizer.decode(p)
                #print(p) 
                #print("reconstruct_caption:")
                print(reconstruct_caption)

            # if i == 10:
            #     break




    




