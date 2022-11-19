# ref: https://github.com/zarzouram/image_captioning_with_transformers/tree/main/code/models/IC_encoder_decoder
# Ref: 
# https://rwightman.github.io/pytorch-image-models/models/vision-transformer/
# https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch/notebook

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


class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int,
                 dropout: float):
        super(DecoderLayer, self).__init__()
        """
        param:
        d_model:    features size.
                    int
        num_heads:  number of heads in the multiheadattention model.
                    int
        dropout:    dropout value
                    float
        """

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
        """
        param:
        dec_inputs:     Captions to decode
                        Tensor
                        [max_len, batch_size, embed_dim]
        enc_outputs:    Encoded image to decode
                        Tensor
                        [encode_size^2=196, batch_size, embed_dim]
        tgt_mask:       Mask to ensure that decoder doesn't look at future
                        tokens from a given subsequence
                        [max_len , max_len]
        tgt_pad_mask:   Mask to ensure that decoder doesn't attend pad tokens
                        [batch_size , max_len]
        outputs:
        output:         Decoder output
                        Tensor
                        [max_len, batch_size, embed_dim]
        attn:           Attension weights
                        Tensor
                        [layer_num, batch_size, head_num, max_len,
                        encode_size^2]
                        To be able to do so, I have changed the code at
                        /.virtualenvs/<env_name>/lib/python3.8/site-packages/torch/nn/functional.py
                        line 4818 and changed
                        `return attn_output, attn_output_weights.sum(dim=1) /
                        num_heads` to be
                        `return attn_output, attn_output_weights`
        """
        # self attention + resedual summation + norm
        output, _ = self.dec_self_attn(dec_inputs,
                                       dec_inputs,
                                       dec_inputs,
                                       attn_mask=tgt_mask,
                                       key_padding_mask=tgt_pad_mask)
        output = dec_inputs + self.self_attn_dropout(output)
        output = self.self_attn_norm(output)  # type: Tensor

        # # self attention + residual + norm + FF
        output2, attns = self.multihead_attn(output, enc_outputs, enc_outputs)
        output = output + self.multihead_dropout(output2)
        output = self.multihead_norm(output)

        output2 = self.ff(output)  # type: Tensor
        output = self.ff_norm(output + self.ff_dropout(output2))

        return output, attns


class Decoder(nn.Module):
    """
    param:
    layer:          an instance of the EecoderLayer() class
    vocab_size:     the number of vocabulary
                    int
    d_model:        size of features in the transformer inputs
                    int
    num_layers:     the number of decoder-layers
                    int
    max_len:        maximum len pf target captions
                    int
    dropout:        dropout value
                    float
    pad_id:         padding token id
                    float
    """

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
        # self.cptn_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        # self.pos_emb = PositionalEmbedding(d_model, max_len)

        # Make copies of the decoder layer
        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_layers)])

        self.dropout = nn.Dropout(p=dropout)

    def get_attn_subsequent_mask(self, sz: int) -> Tensor:
        """
        Generates an upper-triangular matrix of -inf, with zeros on diag.
        """
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, tgt_cptn, src_img, tgt_mask, tgt_pad_mask):
        """
        param:
        tgt_cptn:   Captions (Transformer target sequence)
                    Tensor
                    [batch_size, max_len-1]
        src_img:    Encoded images (Transformer source sequence)
                    Tensor
                    [encode_size^2, batch_size, image_embed_dim]
        outputs:
        output:     Decoder output
                    Tensor
                    [max_len, batch_size, model_embed_dim]
        attn_all:   Attension weights
                    Tensor
                    [layer_num, batch_size, head_num, max_len-1,
                    encode_size^2]
                    See comments in decoder_layers.DecoderLayer
        """

        # create masks, then pass to decoder
        # tgt_pad_mask = (tgt_cptn == self.pad_id)
        # tgt_mask = self.get_attn_subsequent_mask(tgt_cptn.size()[1])
        # tgt_mask = tgt_mask.to(tgt_cptn.device)

        # encode captions + pos enc
        # (B, max_len) -> (B, max_len, d_model) -> (max_len, B, d_model)
        # tgt_cptn = self.cptn_emb(tgt_cptn)  # type: Tensor
        # tgt_cptn = self.dropout(self.pos_emb(tgt_cptn.permute(1, 0, 2)))

        attns_all = []
        for layer in self.layers:
            tgt_cptn, attns = layer(tgt_cptn, src_img, tgt_mask, tgt_pad_mask)
            attns_all.append(attns)
        # [layer_num, batch_size, head_num, max_len, encode_size**2]
        attns_all = torch.stack(attns_all)

        return tgt_cptn, attns_all


class VisualTransformer(nn.Module):
    def __init__(self, embed_dim, encoder_model_name, target_vocab_size, seq_length ,num_layers, num_freeze_layer, n_heads, expansion_factor=4):
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
        self.position_embedding = PositionalEmbedding(seq_length, embed_dim) #nn.Embedding
        self.target_vocab_size = target_vocab_size

        #self.encoder = TransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads)
        #print(timm.list_models("*vit*"))
        self.encoder = torch.nn.Sequential(*(list(timm.create_model(encoder_model_name, pretrained=True).children())[:-1]))
        
        # Freeze layers
        #print(self.encoder)
        for i, module in enumerate(self.encoder): 
            if i == 3:
                for j, layer in enumerate(module):
                    if j < num_freeze_layer:
                        for param in layer.parameters():
                            param.requires_grad = False
                    else:
                        print(j, layer)

        #print(timm.list_models("*vit*"))
        dim_feedforward = embed_dim*expansion_factor

        decoder_layer = DecoderLayer(d_model=embed_dim, num_heads=n_heads, feedforward_dim=dim_feedforward, dropout=0.1)
        self.decoder = Decoder(layer=decoder_layer,
                               vocab_size=self.target_vocab_size,
                               d_model=embed_dim,
                               num_layers=num_layers,
                               max_len=seq_length,
                               dropout=0.1,
                               pad_id=0)


        self.linear = nn.Linear(embed_dim ,self.target_vocab_size)


    def decode(self, img, max_seq_len, device):
        BOS = 2
        EOS = 3
        bz = img.size(0)
        # ref: https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/Translator.py#L9
        
        # init:
        enc_out = self.encoder(img)
        ans_idx = 0   # default
        assert bz==1
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
            outputs, _ = self.decoder(tgt_cptn=tgt.permute(1,0,2), src_img=enc_out.permute(1,0,2), 
                                    tgt_mask=None, tgt_pad_mask=None)
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
        outputs, attn_all = self.decoder(tgt_cptn=tgt.permute(1,0,2), src_img=enc_out.permute(1,0,2), 
                               tgt_mask=trg_mask, tgt_pad_mask=padding_masks)
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
    #loss = F.cross_entropy(pred, gth, ignore_index=PAD, label_smoothing=smoothing)# ,reduction='sum')
    # print(pred)
    # print(gth)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=smoothing)
    loss = criterion(pred, gth)
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    return loss

def show_n_param(model):
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hw 3-2 train",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", help="batch size", type=int, default=8)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", help="weight decay", type=float, default=1e-6)
    parser.add_argument("--scheduler_warmup_steps", help="scheduler learning rate warmup step ", type=int, default=500)
    parser.add_argument("--gamma", help="learning rate decay factor.",type=float, default=0.9)
    parser.add_argument("--n_epochs", help="n_epochs", type=int, default=50)
    parser.add_argument("--smoothing", help="label smoothing factor", type=float, default=0.0)
    # ======================================================================                             
    parser.add_argument("--ckpt_path", help="Checkpoint location", default= "./ckpt_laion2b_L8")
    
    parser.add_argument("--model_option",  default= "vit_base_patch32_224_clip_laion2b") #"vit_base_resnet50_384"  "vit_large_patch14_224_clip_laion2b"
    parser.add_argument("--resize", help="resize", type=int, default=224)
    parser.add_argument("--n_heads", help="n_heads. paper=12", type=int, default=16)
    parser.add_argument("--embed_dim", help="embed_dim", type=int, default=768) # 16*96
    parser.add_argument("--num_layers", help="num_layers", type=int, default=8)
    parser.add_argument("--num_freeze_layer", help="num_freeze_layer in encoder", type=int, default=10)
    # ====================================================================== 

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
    # Leaning rate
    lr = args.learning_rate
    weight_decay = args.weight_decay

    # loss smootthing
    smoothing = args.smoothing
    
    # Epoch
    epochs = args.n_epochs
    gradient_clip = 5.0
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
    model = VisualTransformer(embed_dim=embed_dim, encoder_model_name=encoder_model_name, 
                              target_vocab_size=target_vocab_size, seq_length=seq_length,
                              num_layers=num_layers,
                              num_freeze_layer=num_freeze_layer,
                              n_heads=n_heads)
    model = model.to(device)
    #print(model)
    show_n_param(model)
    
    
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, len(data_loader_train)*epochs)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, drop_step, gamma)
    
    # clip computer
    # clip_model, _preprocess = clip.load("ViT-B/32", device=device)
    # clip_model.eval()
    cos = torch.nn.CosineSimilarity(dim=1)
    w=2.5

    #best_loss = 1000
    best_clip_score = -1
    loss_curve_train = []
    loss_curve_val = []
    start_epoch = 0

    # # Load 
    # resume  = os.path.join(ckpt_path, f"epoch_2_best.pth")
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

            # print("out", out.shape)
            # print("out", out)
            # print("in", tokenized_ids)
            # print("gth",gth_ids)
            # print("out_argmax", torch.argmax(out,1))
            #print("out_amax", torch.amax(out,1))
            loss = loss_compute(out, gth_ids, smoothing)
            
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            lr_scheduler.step()

            # Record
            pbar.set_postfix(loss=loss.item(), lr = optimizer.param_groups[0]['lr'])
            loss_curve_train.append(loss.item())

        # ========================= Eval: how to do ? ==========================
        model.eval()
        print("Eval")
        pbar_val = tqdm(enumerate(data_loader_val))
        pbar_val.set_description(f"Epoch {epoch}")
        val_loss = 0
        with torch.no_grad():
            for i, data in pbar_val:
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
                out = model(image, tokenized_ids, trg_mask,padding_masks) # trg_mask, padding_masks)
                loss = loss_compute(out, tokenized_ids, smoothing)
                if i<=20:
                    pred_ids = model.decode(image[0].unsqueeze(0), 100, device)

                    for p in pred_ids:
                        p = list(p.cpu())
                        reconstruct_caption = tokenizer.decode(p)
                        #print(p) 
                        #print("reconstruct_caption:")
                        print(reconstruct_caption)

                    # print("pred:",torch.argmax(out, 1)[0])
                    print("reconstruct_caption", reconstruct_caption)
                    # print("gth:",gth_ids[0])
                    print("gth captions", captions)
                    #print("loss",loss)
                else:
                    break
                val_loss += loss.item()
                pbar_val.set_postfix(loss=loss.item())
                loss_curve_val.append(loss.item())
                
        clip_scores = []
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
        #         #text = tokenizer.decode_batch(torch.argmax(out,1).cpu().numpy())
        #         text_ids = torch.zeros(tokenized_ids.shape, dtype=torch.int64).to(device)
        #         text_ids[:,1:] = torch.argmax(out,1)[:,:-1]
        #         text_ids[:,0] = 2 # BOS
        #         text = tokenizer.decode_batch(text_ids.cpu().numpy())
        #         #print("text", text)
                
        #         # compute clip
        #         image_embedding = clip_model.encode_image(image)
        #         text_ids = clip.tokenize(text).to(device)
        #         text_embedding = clip_model.encode_text(text_ids)
        #         # cos_similarity =  a*b/|a|/|b|
        #         score = cos(image_embedding, text_embedding)

        #         mean_score = score.mean()
        #         _clip = w*max(mean_score, 0).cpu().numpy()

        #         pbar_val.set_postfix(clip=_clip, lr = optimizer.param_groups[0]['lr'])
        #         clip_scores.append(_clip)
        
        # clip_score = sum(clip_scores)/len(clip_scores)
        if True: #best_clip_score <= clip_score:
            #best_clip_score = clip_score
            save_as = os.path.join(ckpt_path, f"epoch_{epoch}_best.pth")
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