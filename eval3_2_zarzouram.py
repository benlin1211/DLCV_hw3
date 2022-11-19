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
import glob


from torch.utils.data import Dataset, DataLoader
import torchvision 
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
import json

from torch import Tensor
from typing import Tuple
from torch.nn import MultiheadAttention
from copy import deepcopy

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
        for param in self.encoder.parameters():
            param.requires_grad = False

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

def show_n_param(model):
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hw 3-2 train",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("src_path", help="src_path. ex: hw3_data/p2_data/images/val") 
    parser.add_argument("des_path", help="des_path. ex: hw3/output_p2/pred.json") 
    parser.add_argument("--tokenizer_path", help="tokenizer location", default= "./hw3_data/caption_tokenizer.json")
    # ================================ EVAL ======================================    
    parser.add_argument("--ckpt_path", help="Checkpoint location", default= "./ckpt_vit_large_patch14_224_clip_laion2b_L8")
    parser.add_argument("--resume_name", help="Checkpoint resume name", default= "epoch_0_best.pth")

    parser.add_argument("--model_option",  default= "vit_large_patch14_224_clip_laion2b") #"vit_base_resnet50_384"  "vit_base_patch14_224_clip_laion2b"
    parser.add_argument("--resize", help="resize", type=int, default=224)
    parser.add_argument("--n_heads", help="n_heads", type=int, default=16)
    parser.add_argument("--embed_dim", help="embed_dim", type=int, default=1024)
    parser.add_argument("--num_layers", help="num_layers", type=int, default=8) # actually 8
    parser.add_argument("--num_freeze_layer", help="num_freeze_layer in encoder", type=int, default=10)
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
    device = torch.device("cuda")

    print("Using", device)
    des_path = args.des_path
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


    # Tokenizer setting
    tokenizer = Tokenizer.from_file(tokenizer_path)
    target_vocab_size = len(tokenizer.get_vocab()) # vocab_size 18022
    seq_length = 196 # I don't know why...


    val_transform = transforms.Compose([
        transforms.Resize((resize,resize)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_val = ImageVal(image_dir=src_path, 
                            transform=val_transform)

    data_loader_val = DataLoader(dataset_val, batch_size, shuffle=False, drop_last=False, num_workers=0)

    # Debugger
    len_dataloader_val = len(data_loader_val)
    #data_iter_val = iter(data_loader_val)
    
    print("val:", len_dataloader_val)

    # model
    model = VisualTransformer(embed_dim=embed_dim, encoder_model_name=encoder_model_name, 
                              target_vocab_size=target_vocab_size, seq_length=seq_length,
                              num_layers=num_layers,
                              num_freeze_layer=num_freeze_layer,
                              n_heads=n_heads)
    show_n_param(model)
    model = model.to(device)


    # Load 
    
    resume  = os.path.join(ckpt_path, resume_name)
    print(f"load from {resume}")
    checkpoint = torch.load(resume, map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    max_seq_len = 50
    result = {}
    with torch.no_grad():
        #for i, data in enumerate(data_loader_val): 
        for data in tqdm(data_loader_val):
            #data = data_iter_val.next()
            image, file_name = data 
            image = image.to(device)
            # preprocessing 

            pred_ids = model.decode(image, max_seq_len, device)

            for _, p in enumerate(pred_ids):
                p = list(p.cpu())
                reconstruct_caption = tokenizer.decode(p)
                #print(p) 
                #print("reconstruct_caption:")
                #print(reconstruct_caption)
                # print(_)
            # print("file_name", file_name)
            # print("caption",_caption)
            # print("reconstruct_caption:", reconstruct_caption)
            result[file_name[0]] = reconstruct_caption

    #print(result)

    # save as json
    json_name = des_path.split('/')[-1]
    #print(json_name)
    dest_folder = des_path.replace(json_name,'') 
    #print(dest_folder)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    with open(des_path, "w") as f:
        json.dump(result, f)
    print(f"Done. json file is saved at {des_path}")



    




