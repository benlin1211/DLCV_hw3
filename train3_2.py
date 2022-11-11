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
        image = Image.open(os.path.join(self.image_dir, file_name[0]))
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
            # src_mask
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
        self.word_embedding = nn.Embedding(target_vocab_size, embed_dim)
        self.position_embedding = PositionalEmbedding(seq_length, embed_dim)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_dim, expansion_factor=expansion_factor, n_heads=n_heads) 
                for _ in range(num_layers)
            ]

        )
        self.fc_out = nn.Linear(embed_dim, target_vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, query, key_value, mask):
        enc_out = key_value
        x = query  
        """
        Args:
            x: input vector from target
            enc_out : output from encoder layer
            trg_mask: mask for decoder self attention
        Returns:
            out: output vector
        """

        x = self.word_embedding(x)  #32x10x512
        x = self.position_embedding(x) #32x10x512
        x = self.dropout(x)
        print(x.shape)
     
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
        
        self.target_vocab_size = target_vocab_size

        #self.encoder = TransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads)
        self.encoder = torch.nn.Sequential(*(list(timm.create_model(encoder_model_name, pretrained=True).children())[:-1]))
        #self.encoder = timm.create_model('vit_base_patch16_224', pretrained=True)
        
        #print(self.encoder)     
        self.decoder = TransformerDecoder(target_vocab_size, embed_dim, seq_length, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads)
        
    def nopeak_mask(size):
        np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
        np_mask =  Variable(torch.from_numpy(np_mask) == 0)

        return np_mask
      
    def make_trg_mask(self, trg):
        """
        Args:
            trg: target sequence
        Returns:
            trg_mask: target mask
        """
        batch_size, trg_len = trg.shape
        # returns the lower triangular part of matrix filled with ones
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            batch_size, 1, trg_len, trg_len
        )
        trg_mask[trg_mask==0] = float('-inf')
        # trg_mask = torch.tril(torch.full((trg_len, trg_len), float('-inf'))).expand(
        #     batch_size, 1, trg_len, trg_len
        # )
        #print(trg.size(-1))
        #trg_mask = nn.Transformer.generate_square_subsequent_mask(trg.size(-1))
        
        return trg_mask  

    
    def forward(self, img, tgt):
        """
        Args:
            img: input to encoder (image)
            tgt: input to decoder (tokens)
        out:
            out: final vector which returns probabilities of each target word
        """

        enc_out = self.encoder(img)
        trg_mask = self.make_trg_mask(tgt)
        print("enc_out",enc_out.shape)   # torch.Size([1, 196, 768])
        print("tgt",tgt.shape)           # torch.Size([1, tgt.shape[1]])
        print("trg_mask",trg_mask.shape) # torch.Size([1, 1, tgt.shape[1], tgt.shape[1]])
        #print("trg_mask",trg_mask) # torch.Size([1, 1, tgt.shape[1], tgt.shape[1]])

        outputs = self.decoder(query=tgt, key_value=enc_out, mask=trg_mask)
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
    batch_size = 4
    n_heads = 8
    # Leaning rate
    lr = 1e-4
    weight_decay = 1e-4
    
    # Epoch
    epochs = 30
    drop_step = 20
    root_dir = "./hw3_data"
    
    # model_example = timm.create_model('vit_base_patch16_224', pretrained=True)
    # model_example.eval()
    # url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    # urllib.request.urlretrieve(url, filename)
    # img = Image.open(filename).convert('RGB')
    # from timm.data import resolve_data_config
    # from timm.data.transforms_factory import create_transform
    # config = resolve_data_config({}, model=model_example)
    # tfm = create_transform(**config)
    # print("dog",tfm(img).shape)
    # with torch.no_grad():
    #     out = model_example(tfm(img).unsqueeze(0))
    #     print(out)

    # Tokenizer setting
    tokenizer = Tokenizer.from_file(os.path.join(root_dir, "caption_tokenizer.json"))
    
    # with open("./hw3_data/p2_data/train.json", newline='') as jsonfile:
    #     data = json.load(jsonfile)

    # example_caption = data["annotations"][0:3]
    # example_image = data["images"][0:3]
    # filter_image = [item["file_name"] for item in data["images"] if item["id"] == 60623] # data["images"][1]

    # print(example_caption)
    # print(example_image)
    # print(filter_image)
    
    # c = example_caption[0]["caption"]
    # print(c)
    # tokenized_caption = tokenizer.encode(c)
    # print("id", torch.tensor(tokenized_caption.ids))
    # print("tokens", tokenized_caption.tokens)
    # print("attention_mask", tokenized_caption.attention_mask)
    target_vocab_size = len(tokenizer.get_vocab()) # vocab_size 18022
    seq_length = 196 # I don't know why...

    # model
    model = VisualTransformer(embed_dim=768, encoder_model_name=encoder_model_name, 
                              target_vocab_size=target_vocab_size, seq_length=seq_length, n_heads=n_heads)
    show_n_param(model)

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
    for i in tqdm(range(len_dataloader)):
        data = data_iter.next()
        image, captions = data    
        # print(image.shape)
        # print(captions)

        # preprocessing
        tokenized_captions = tokenizer.encode_batch(captions)
        tokenized_ids = torch.tensor([c.ids for c in tokenized_captions])
        # print(tokenized_ids)
        tgt_padding_masks = torch.tensor([c.attention_mask for c in tokenized_captions])
        # http://juditacs.github.io/2018/12/27/masked-attention.html
        print(tgt_padding_masks) 
        # tokens = [c.tokens for c in tokenized_captions]
        # n_sequences = [c.n_sequences for c in tokenized_captions]

        out = model(image, tokenized_ids)
        print(out.shape)

 
    
    



