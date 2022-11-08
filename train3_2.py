# Ref: https://rwightman.github.io/pytorch-image-models/models/vision-transformer/
import timm
import urllib
import os
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from train3_2_Decoder import Embedding, PositionalEmbedding, MultiHeadAttention, DecoderBlock, TransformerDecoder

from tokenizers import Tokenizer
import json

class VisualTransformer(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, target_vocab_size, seq_length ,num_layers=2, expansion_factor=4, n_heads=8):
        super(VisualTransformer, self).__init__()
        
        """  
        Args:
           embed_dim:  dimension of embedding 
           # src_vocab_size: vocabulary size of source (we don't have it in ViT.)
           target_vocab_size: vocabulary size of target
           seq_length : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention
        
        """
        
        self.target_vocab_size = target_vocab_size

        #self.encoder = TransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads)
        self.encoder = torch.nn.Sequential(*(list(timm.create_model('vit_base_patch16_224', pretrained=True).children())[:-1]))
        #self.encoder = timm.create_model('vit_base_patch16_224', pretrained=True)
        #print(self.encoder)
        config = resolve_data_config({}, model=self.encoder)
        self.transform = create_transform(**config)
        
        self.decoder = TransformerDecoder(target_vocab_size, embed_dim, seq_length, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads)
        
    
    def make_trg_mask(self, tgt):
        """
        Args:
            tgt: target sequence
        Returns:
            trg_mask: target mask
        """
        batch_size, trg_len = tgt.shape
        # returns the lower triangular part of matrix filled with ones
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            batch_size, 1, trg_len, trg_len
        )
        return trg_mask    

    def decode(self,src,tgt):
        """
        for inference
        Args:
            src: input to encoder 
            tgt: input to decoder
        out:
            out_labels : returns final prediction of sequence
        """
        trg_mask = self.make_trg_mask(tgt)
        enc_out = self.encoder(src)
        out_labels = []
        batch_size,seq_len = src.shape[0],src.shape[1]
        #outputs = torch.zeros(seq_len, batch_size, self.target_vocab_size)
        out = tgt
        for i in range(seq_len): #10
            out = self.decoder(out,enc_out,trg_mask) #bs x seq_len x vocab_dim
            # taking the last token
            out = out[:,-1,:]
     
            out = out.argmax(-1)
            out_labels.append(out.item())
            out = torch.unsqueeze(out,axis=0)
          
        
        return out_labels
    
    def forward(self, img, tgt):
        """
        Args:
            src: input to encoder 
            tgt: input to decoder
        out:
            out: final vector which returns probabilities of each target word
        """
        
        img = self.transform(img).unsqueeze(0)
        enc_out = self.encoder(img)

        trg_mask = self.make_trg_mask(tgt)
        print("enc_out",enc_out.shape)
        print("tgt",tgt.shape)        
        print("trg_mask",trg_mask.shape)
        outputs = self.decoder(tgt, enc_out, trg_mask)
        print(outputs.shape)
        return outputs



if __name__ == "__main__":

    model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)  # you can choose between v1, v2 and v3
    print(model)
    """
    src_vocab_size=0
    batch_size = 1
    data_dir = "./hw3_data"
    
    tokenizer = Tokenizer.from_file(os.path.join(data_dir, "caption_tokenizer.json"))
    
    with open(os.path.join(data_dir, "p2_data/train.json"), newline='') as jsonfile:
        data = json.load(jsonfile)
        # 或者這樣
        # data = json.loads(jsonfile.read())
    example_caption = data["annotations"][1]["caption"]
    print(example_caption)

    tokenized_caption = tokenizer.encode(example_caption)
    print("id", tokenized_caption.ids)
    print("tokens", tokenized_caption.tokens)
    target_vocab_size = len(tokenizer.get_vocab()) # vocab_size 18022
    seq_length = 100
    model = VisualTransformer(embed_dim=768, src_vocab_size=src_vocab_size, 
                              target_vocab_size=target_vocab_size, seq_length=seq_length)
    model.eval()

    data_dir = "./hw3_data/p2_data/images/train"
    filename = os.path.join(data_dir,"000000000078.jpg")
    img = Image.open(filename).convert('RGB')

    # tgt = torch.randint( 0, 768,(batch_size, 768))
    tgt = torch.tensor([tokenized_caption.ids])
    print(tgt)
    with torch.no_grad():
        out = model(img, tgt)
    """
    


    # # Get imagenet class mappings
    # url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
    # urllib.request.urlretrieve(url, filename) 
    # with open("imagenet_classes.txt", "r") as f:
    #     categories = [s.strip() for s in f.readlines()]

