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
import clip

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


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        """
        Args:
            embed_dim: dimension of embeding vector output
            n_heads: number of self attention heads
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim    #512 dim
        self.n_heads = n_heads   #8
        self.single_head_dim = int(self.embed_dim / self.n_heads)   #512/8 = 64  . each key,query, value will be of 64d
       
        #key,query and value matrixes    #64 x 64   
        self.query_matrix = nn.Linear(self.single_head_dim , self.single_head_dim ,bias=False)  # single key matrix for all 8 keys #512x512
        self.key_matrix = nn.Linear(self.single_head_dim  , self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim ,self.single_head_dim , bias=False)
        self.out = nn.Linear(self.n_heads*self.single_head_dim ,self.embed_dim) 

    def forward(self,key,query,value,mask=None):    #batch_size x sequence_length x embedding_dim    # 32 x 10 x 512
        
        """
        Args:
           key : key vector
           query : query vector
           value : value vector
           mask: mask (only for decoder)
        
        Returns:
           output vector from multihead attention
        """
        batch_size = key.size(0)
        #seq_length = key.size(1)
        
        # query dimension can change in decoder during inference. 
        # so we cant take general seq_length
        # seq_length_query = query.size(1)
        
        # Reshape for matrix computation
        key = key.view(batch_size, -1, self.n_heads, self.single_head_dim)  #batch_size x sequence_length x n_heads x single_head_dim = (32x10x8x64)
        query = query.view(batch_size, -1, self.n_heads, self.single_head_dim) #(32x10x8x64)
        value = value.view(batch_size, -1, self.n_heads, self.single_head_dim) #(32x10x8x64)
       
        # matrix computation
        k = self.key_matrix(key)       # (32x10x8x64)
        q = self.query_matrix(query)   
        v = self.value_matrix(value)

        # Reshape back
        q = q.transpose(1,2)  # (batch_size, n_heads, seq_length, single_head_dim)    # (32 x 8 x 10 x 64)
        k = k.transpose(1,2)  # (batch_size, n_heads, seq_length, single_head_dim)
        v = v.transpose(1,2)  # (batch_size, n_heads, seq_length, single_head_dim)
       
        # computes attention
        # adjust key for matrix multiplication
        """"""
        # k_adjusted = k.transpose(-1,-2)  #(batch_size, n_heads, single_head_dim, seq_ken)  #(32 x 8 x 64 x 10)
        # # calculate attention using function we will define next
        # product = torch.matmul(q, k_adjusted)  #(32 x 8 x 10 x 64) x (32 x 8 x 64 x 10) = #(32x8x10x10)
        # # fill those positions of product matrix as (-1e20) where mask positions are 0
        # if mask is not None:
        #     print("AAA")
        #     print(mask.shape)
        #     print(product.shape)
        #     product = product.masked_fill(mask == 0, float("-1e20"))
        # #divising by square root of key dimension
        # product = product / math.sqrt(self.single_head_dim) # / sqrt(64)

        def attention(q, k, v, d_k, mask=None):
        
            scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
            # it is src_mask
            if mask is not None:
                print(mask.shape)
                print(scores.shape)
                mask = mask.unsqueeze(1)
                scores = scores.masked_fill(mask == 0, -1e9)
            
            scores = F.softmax(scores, dim=-1)
            output = torch.matmul(scores, v)
            return output  

        scores = attention(q, k, v, self.single_head_dim, mask)

        #concatenated output
        concat = scores.transpose(1,2).contiguous().view(batch_size, -1, self.single_head_dim*self.n_heads)  # (32x8x10x64) -> (32x10x8x64)  -> (32,10,512)
        
        output = self.out(concat) #(32,10,512) -> (32,10,512)
       
        return output

# Elementary Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(TransformerBlock, self).__init__()
        
        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads
        
        """
        # We don't use mask in encoder. (For simplicity)
        self.attention = MultiHeadAttention(embed_dim, n_heads) 

        self.norm1 = nn.LayerNorm(embed_dim) 
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
                          nn.Linear(embed_dim, expansion_factor*embed_dim),
                          nn.ReLU(),
                          nn.Linear(expansion_factor*embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self,key,query,value):
        
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           norm2_out: output of transformer block
        
        """
        
        attention_out = self.attention(key,query,value)  #32x10x512
        attention_residual_out = attention_out + value  #32x10x512
        norm1_out = self.dropout1(self.norm1(attention_residual_out)) #32x10x512

        feed_fwd_out = self.feed_forward(norm1_out) #32x10x512 -> #32x10x2048 -> 32x10x512
        feed_fwd_residual_out = feed_fwd_out + norm1_out #32x10x512
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out)) #32x10x512

        return norm2_out


# Encoder
# class TransformerEncoder(nn.Module):
#     def __init__(self, seq_length, vocab_size, embed_dim, num_layers=2, expansion_factor=4, n_heads=8):
#         super(TransformerEncoder, self).__init__()
#         """
#         Args:
#             seq_length : length of input sequence
#             embed_dim: dimension of embedding
#             num_layers: number of encoder layers
#             expansion_factor: factor which determines number of linear layers in feed forward layer
#             n_heads: number of heads in multihead attention
            
#         Returns:
#             out: output of the encoder
#         """        
#         self.embedding_layer = Embedding(vocab_size, embed_dim)
#         self.positional_encoder = PositionalEmbedding(seq_length, embed_dim)

#         self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])
    
#     def forward(self, x):
#         embed_out = self.embedding_layer(x)
#         out = self.positional_encoder(embed_out)
#         for layer in self.layers:
#             out = layer(out,out,out)

#         return out  #32x10x512

# Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(DecoderBlock, self).__init__()

        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads
        
        """
        self.attention = MultiHeadAttention(embed_dim, n_heads=n_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.transformer_block = TransformerBlock(embed_dim, expansion_factor, n_heads)
        
    
    def forward(self, key, query, value, mask):
        
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           mask: mask to be given for multi head attention 
        Returns:
           out: output of transformer block
    
        """
        
        #we need to pass mask mask only to fst attention
        attention = self.attention(value,value,value,mask=mask) #32x10x512
        value = self.dropout(self.norm(attention + value))
        
        out = self.transformer_block(key, query, value)

        
        return out

# Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, seq_length, num_layers=2, expansion_factor=4, n_heads=8):
        super(TransformerDecoder, self).__init__()
        """  
        Args:
           target_vocab_size: vocabulary size of taget
           embed_dim: dimension of embedding
           seq_length : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention
        
        """
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_dim, expansion_factor=expansion_factor, n_heads=n_heads) 
                for _ in range(num_layers)
            ]

        )
        self.fc_out = nn.Linear(embed_dim, target_vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, tgt, memory, tgt_mask, tgt_key_padding_mask):
        enc_out = memory
        x = tgt  
        """
        Args:
            x: input vector from target
            enc_out : output from encoder layer
            trg_mask: mask for decoder self attention
        Returns:
            out: output vector
        """
        attn_mask = tgt_mask
        print("attn_mask",attn_mask)
        print("tgt_mask",tgt_mask.shape)
        print("tgt_key_padding_mask",tgt_key_padding_mask.shape)
        print("tgt_key_padding_mask",tgt_key_padding_mask)
        attn_mask = attn_mask.logical_or(tgt_key_padding_mask)
        print("attn_mask",attn_mask)

        # create mask
     
        for layer in self.layers:
            x = layer(key=enc_out, query=x, value=enc_out, mask=mask) 

        out = F.softmax(self.fc_out(x))

        return out


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
        self.decoder = TransformerDecoder(target_vocab_size, embed_dim, seq_length, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads)
        self.linear = nn.Linear(embed_dim ,target_vocab_size)
    
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
        outputs = outputs.permute(1,2,0) # torch.Size([bz, target_vocab_size, tgt_len])
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

def loss_compute(pred, gth, smoothing):
    # https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/train.py#L40 
    PAD = 0
    loss = F.cross_entropy(pred, gth, ignore_index=PAD, label_smoothing=smoothing)# ,reduction='sum')
    #loss = nn.CrossEntropyLoss(pred, gth, ignore_index=PAD)
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    return loss

def show_n_param(model):
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hw 3-2 train",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckpt_path", help="Checkpoint location", default= "./ckpt3_2")
    parser.add_argument("--batch_size", help="batch size", type=int, default=16)
    parser.add_argument("--model_option", default="vit_base_patch16_224") #"vit_base_resnet50_384" 
    parser.add_argument("--resize", help="resize", type=int, default=224)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", help="weight decay", type=float, default=1e-4)
    parser.add_argument("--scheduler_warmup_steps", help="scheduler learning rate warmup step ", type=int, default=2000)
    parser.add_argument("--gamma", help="learning rate decay factor.",type=float, default=0.9)
    parser.add_argument("--n_epochs", help="n_epochs", type=int, default=250)
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
    resize = args.resize
    batch_size = args.batch_size
    # Leaning rate
    lr = args.learning_rate
    weight_decay = args.weight_decay

    # loss smootthing
    smoothing = 0.1
    
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
    len_dataloader_val = len(data_loader_val)

    
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
    
    # clip computer
    clip_model, _preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    cos = torch.nn.CosineSimilarity(dim=1)
    w=2.5

    #model.eval()
    #best_loss = 1000
    best_clip_score = -1
    loss_curve_train = []
    loss_curve_val = []

    # # Load 
    # resume  = os.path.join(ckpt_path, f"epoch_3.pth")
    # checkpoint = torch.load(resume, map_location = device)

    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # epoch = checkpoint['epoch']

    for epoch in range(epochs):
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
            gth_ids = torch.zeros(tokenized_ids.shape, dtype=torch.int64).to(device)
            gth_ids[:,:-1] = tokenized_ids[:,1:]

            padding_masks = torch.tensor([c.attention_mask for c in tokenized_captions], dtype=torch.bool).to(device) #1 not seen
            padding_masks = ~padding_masks

            #padding_masks[padding_masks==1] = float('-inf')
            # print(tokenized_ids)
            # print(padding_masks)
            # tokens = [c.tokens for c in tokenized_captions]
            # n_sequences = [c.n_sequences for c in tokenized_captions]
            trg_mask = nn.Transformer.generate_square_subsequent_mask(tokenized_ids.size(-1)).to(device) #1 not seen
            # https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
            # print(trg_mask)
            out = model(image, tokenized_ids, trg_mask, padding_masks)

            # print("out", out)
            # print("in", tokenized_ids)
            # print("gth",gth_ids)
            # print("out_argmax", torch.argmax(out,1))
            # print("Shift", tokenized_ids)
            loss = loss_compute(out, gth_ids, smoothing)
            
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), weight_clip)
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
        # val_loss = 0
        # with torch.no_grad():
        #     for data in pbar_val:
        #         image, captions = data 
        #         image = image.to(device)
        #         # preprocessing 
        #         tokenized_captions = tokenizer.encode_batch(captions)
        #         tokenized_ids = torch.tensor([c.ids for c in tokenized_captions])
        #         tokenized_ids = tokenized_ids.to(device)
        #         # Shift
        #         gth_ids = torch.zeros(tokenized_ids.shape, dtype=torch.int64).to(device)
        #         gth_ids[:,:-1] = tokenized_ids[:,1:]

        #         padding_masks = torch.tensor([c.attention_mask for c in tokenized_captions], dtype=torch.bool).to(device) #1 not seen
        #         padding_masks = ~padding_masks  
        #         trg_mask =  nn.Transformer.generate_square_subsequent_mask(tokenized_ids.size(-1)).to(device) #1 not seen
        #         out = model(image, tokenized_ids, None,None) # trg_mask, padding_masks)

        #         loss = loss_compute(out, tokenized_ids, smoothing)
        #         val_loss += loss.item()
        #         pbar_val.set_postfix(loss=loss.item())
        #         loss_curve_val.append(loss.item())
        with torch.no_grad():
            for data in pbar_val:
                image, captions = data 
                image = image.to(device)
                # preprocessing 
                tokenized_captions = tokenizer.encode_batch(captions)
                tokenized_ids = torch.tensor([c.ids for c in tokenized_captions])
                tokenized_ids = tokenized_ids.to(device)
                # Shift
                gth_ids = torch.zeros(tokenized_ids.shape, dtype=torch.int64).to(device)
                gth_ids[:,:-1] = tokenized_ids[:,1:]

                padding_masks = torch.tensor([c.attention_mask for c in tokenized_captions], dtype=torch.bool).to(device) #1 not seen
                padding_masks = ~padding_masks  
                trg_mask =  nn.Transformer.generate_square_subsequent_mask(tokenized_ids.size(-1)).to(device) #1 not seen
                
                out = model(image, tokenized_ids, None,None) # trg_mask, padding_masks)
                #text = tokenizer.decode_batch(torch.argmax(out,1).cpu().numpy())
                text_ids = torch.zeros(tokenized_ids.shape, dtype=torch.int64).to(device)
                text_ids[:,1:] = torch.argmax(out,1)[:,:-1]
                text_ids[:,0] = 2 # BOS
                text = tokenizer.decode_batch(text_ids.cpu().numpy())
                #print("text", text)
                
                # compute clip
                image_embedding = clip_model.encode_image(image)
                text_ids = clip.tokenize(text).to(device)
                text_embedding = clip_model.encode_text(text_ids)
                # cos_similarity =  a*b/|a|/|b|
                score = cos(image_embedding, text_embedding)

                mean_score = score.mean()
                clip_score = w*max(mean_score, 0).cpu().numpy()

                pbar_val.set_postfix(clip=clip_score, lr = optimizer.param_groups[0]['lr'])
                
        if best_clip_score < clip_score:
            best_clip_score = clip_score
            save_as = os.path.join(ckpt_path, f"epoch_{epoch}.pth")
            print(f"New best clip score: {best_clip_score}. Saving emodel at {save_as}")
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    }, save_as)

        
    # Load 
    resume  = os.path.join(ckpt_path, f"epoch_0.pth")
    checkpoint = torch.load(resume, map_location = device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']


    print("OK")

    df = pd.DataFrame() # apply pd.DataFrame format 
    df["loss_train"] = loss_curve_train
    csv_name = os.path.join(ckpt_path, f"loss_train.csv")
    df.to_csv(csv_name, index = False)


    df = pd.DataFrame() # apply pd.DataFrame format 
    df["loss_val"] = loss_curve_val
    csv_name = os.path.join(ckpt_path, f"loss_val.csv")
    df.to_csv(csv_name, index = False)
