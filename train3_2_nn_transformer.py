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
    def __init__(self, embed_dim, encoder_model_name, target_vocab_size, seq_length ,num_layers=2, expansion_factor=4, n_heads=8):
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

        #self.decoder = 
  
    
    def forward(self, img, tgt, trg_mask, padding_masks):

        

        tgt = self.word_embedding(tgt)  #32x10x512
        tgt = self.position_embedding(tgt) #32x10x512

        enc_out = self.encoder(img)
        print("enc_out",enc_out.shape)   # torch.Size([1, 196, 768])
        print("tgt",tgt.shape)           # torch.Size([1, tgt.shape[1]])

        print("trg_mask",trg_mask.shape) # torch.Size([1, 1, tgt.shape[1], tgt.shape[1]])
        print("trg_mask",trg_mask)
        
        outputs = self.decoder(tgt=tgt.permute(1,0,2), memory=enc_out.permute(1,0,2), tgt_mask=trg_mask, tgt_key_padding_mask=None)
        print(outputs.shape)
        return outputs


def show_n_param(model):
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")


if __name__ == "__main__":

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

    encoder_model_name="vit_base_resnet50_384"  #'vit_base_patch16_224' #
    resize = 384
    batch_size = 8
    # Leaning rate
    lr = 1e-4
    weight_decay = 1e-4
    
    # Epoch
    epochs = 30
    drop_step = 20
    root_dir = "./hw3_data"

    # Tokenizer setting
    tokenizer = Tokenizer.from_file(os.path.join(root_dir, "caption_tokenizer.json"))
    
    target_vocab_size = len(tokenizer.get_vocab()) # vocab_size 18022
    seq_length = 196 # I don't know why...

    # model
    model = VisualTransformer(embed_dim=768, encoder_model_name=encoder_model_name, 
                              target_vocab_size=target_vocab_size, seq_length=seq_length)
    show_n_param(model)
    model = model.to(device)

    # optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, drop_step)

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
    len_dataloader = len(data_loader_train)
    data_iter = iter(data_loader_train)
    
    pred_list = []
    filename_list = []
    print(len_dataloader)
    model.eval()
    for i in tqdm(range(len_dataloader)):
        data = data_iter.next()
        image, captions = data    
        image = image.to(device)
        # print(image.shape)
        # print(captions)

        # preprocessing
        tokenized_captions = tokenizer.encode_batch(captions)
        tokenized_ids = torch.tensor([c.ids for c in tokenized_captions])
        tokenized_ids = tokenized_ids.to(device)
        # print(tokenized_ids)
        padding_masks = torch.tensor([c.attention_mask for c in tokenized_captions]).to(device)
        print(padding_masks)
        # tokens = [c.tokens for c in tokenized_captions]
        # n_sequences = [c.n_sequences for c in tokenized_captions]
        trg_mask = nn.Transformer.generate_square_subsequent_mask(tokenized_ids.size(-1)).to(device)
        out = model(image, tokenized_ids, trg_mask, padding_masks)
        print(out.shape)

 
    
    



