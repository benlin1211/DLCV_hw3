# Ref: https://rwightman.github.io/pytorch-image-models/models/vision-transformer/
import timm
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from train3_2_Decoder import Embedding, PositionalEmbedding, MultiHeadAttention, DecoderBlock, TransformerDecoder



class VisualTransformer(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, target_vocab_size, seq_length=786 ,num_layers=2, expansion_factor=4, n_heads=8):
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
        self.encoder = torch.nn.Sequential(*(list(timm.create_model('vit_base_patch16_224', pretrained=True).children())[:-3]))
        # self.encoder = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
        config = resolve_data_config({}, model=self.encoder)
        self.transform = create_transform(**config)

        self.decoder = TransformerDecoder(target_vocab_size, embed_dim, seq_length, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads)
        
    
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
        return trg_mask    

    def decode(self,src,trg):
        """
        for inference
        Args:
            src: input to encoder 
            trg: input to decoder
        out:
            out_labels : returns final prediction of sequence
        """
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src)
        out_labels = []
        batch_size,seq_len = src.shape[0],src.shape[1]
        #outputs = torch.zeros(seq_len, batch_size, self.target_vocab_size)
        out = trg
        for i in range(seq_len): #10
            out = self.decoder(out,enc_out,trg_mask) #bs x seq_len x vocab_dim
            # taking the last token
            out = out[:,-1,:]
     
            out = out.argmax(-1)
            out_labels.append(out.item())
            out = torch.unsqueeze(out,axis=0)
          
        
        return out_labels
    
    def forward(self, img, trg):
        """
        Args:
            src: input to encoder 
            trg: input to decoder
        out:
            out: final vector which returns probabilities of each target word
        """
        trg_mask = self.make_trg_mask(trg)
        img = self.transform(img).unsqueeze(0)
        enc_out = self.encoder(img)

        outputs = self.decoder(trg, enc_out, trg_mask)
        return outputs


# all_densenet_models = timm.list_models('*vit_base*')
# print(all_densenet_models)
encoder_model = timm.create_model('vit_base_patch16_224', pretrained=True)
#encoder_model = timm.create_model('vit_base_patch16_224', pretrained=True)

# class mynet(nn.Module):
#     def __init__(self, encoder_model):
#         super(mynet, self).__init__()
#         self.encoder_model = torch.nn.Sequential(*(list(encoder_model.children())[:-3]))
#         self.fcn1 = torch.nn.Sequential(*(list(encoder_model.children())[-3:-1]))
#         self.fcn2 = torch.nn.Sequential(list(encoder_model.children())[-1])
#     def forward(self, x):
#         x =  self.encoder_model(x)
#         print("encoder_model",x.shape)
#         x = self.fcn1(x)
#         print("fcn1",x.shape)
#         x = self.fcn2(x)
#         print("fcn2",x.shape)
#         return x


# # model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
encoder_model = timm.create_model('vit_base_patch16_224', pretrained=True)
encoder_model.eval()

model = mynet(encoder_model)
print(model)

config = resolve_data_config({}, model=model)
transform = create_transform(**config)

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)
img = Image.open(filename).convert('RGB')
tensor = transform(img).unsqueeze(0) # transform and add batch dimension


with torch.no_grad():
    print("in", tensor.shape)
    out = model(tensor)
    print("out", out.shape)
probabilities = torch.nn.functional.softmax(out[0], dim=0)
print(probabilities.shape)
# prints: torch.Size([1000])

# Get imagenet class mappings
url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
urllib.request.urlretrieve(url, filename) 
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Print top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
# prints class names and probabilities like:
# [('Samoyed', 0.6425196528434753), ('Pomeranian', 0.04062102362513542), ('keeshond', 0.03186424449086189), ('white wolf', 0.01739676296710968), ('Eskimo dog', 0.011717947199940681)]
