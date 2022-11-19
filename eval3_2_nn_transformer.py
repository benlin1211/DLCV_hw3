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
    def __init__(self, embed_dim, encoder_model_name, target_vocab_size, seq_length ,num_layers, num_freeze_layer, expansion_factor=4, n_heads=8):
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

            outputs = self.decoder(tgt=tgt.permute(1,0,2), memory=enc_out.permute(1,0,2), tgt_mask=None, tgt_key_padding_mask=None)
            outputs = self.linear(outputs)
            outputs = outputs.permute(1,2,0) # torch.Size([bz, target_vocab_size, tgt_len])
            # print(outputs.shape)
            # print(outputs)
            # print(torch.amax(outputs,1))
            best_k_idx = torch.argmax(outputs,1)
            next_token = best_k_idx[:,-1]
            #print("best_k_idx", best_k_idx)
            #print("next_token", next_token)
            gen_seq[:, step] = next_token 
            #print(outputs.shape)

            # Locate the position of [EOS]
            # eos_locs = gen_seq == EOS #[EOS] 
            #print("gen_seq",gen_seq[:, :step], "new", gen_seq[:, step])
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
    parser.add_argument("src_path", help="src_path. ex: hw3_data/p2_data/images/val") 
    parser.add_argument("des_path", help="des_path. ex: hw3/output_p2/pred.json") 
    parser.add_argument("--tokenizer_path", help="tokenizer location", default= "./hw3_data/caption_tokenizer.json")
    # ======================================================================    
    parser.add_argument("--ckpt_path", help="Checkpoint location", default= "./ckpt3_2_all_freeze_layer_4_nhead_16")
    parser.add_argument("--resume_name", help="Checkpoint resume name", default= "epoch_17_best.pth")

    parser.add_argument("--model_option",  default= "vit_large_patch14_224_clip_laion2b") #"vit_base_resnet50_384"  "vit_base_patch14_224_clip_laion2b"
    parser.add_argument("--resize", help="resize", type=int, default=224)
    parser.add_argument("--embed_dim", help="embed_dim", type=int, default=1024)
    parser.add_argument("--n_heads", help="n_heads. paper=12", type=int, default=8)
    parser.add_argument("--num_layers", help="num_layers", type=int, default=4)
    parser.add_argument("--num_freeze_layer", help="num_freeze_layer in encoder", type=int, default=24)

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



    




