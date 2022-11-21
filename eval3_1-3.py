# what is CLIP: 
# https://blog.infuseai.io/openai-的-multimodal-神經網路-下-clip-connecting-text-and-images-2e9962905504

import torch
import os
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset

import clip
from PIL import Image
import json 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import random

def fix_random_seed():
    myseed = 6666  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        

class ImageDataset(Dataset):
    def __init__(self, path, tfm, files = None):
        super(ImageDataset).__init__()
        self.path = path
        #self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".png")], key=lambda x: (int(x.split('/')[-1].split('_')[0]), int(x.split('/')[-1].split('_')[1].split('.')[0]))) # TODO: sort with correct order!
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".png")])
        # Ref: https://stackoverflow.com/questions/54399946/python-glob-sorting-files-of-format-int-int-the-same-as-windows-name-sort

        if files != None:
            self.files = files
        self.transform = tfm

    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx] 
        im = Image.open(fname)
        # 32x32
        im = self.transform(im)
        # im = self.data[idx]
        # if self.mode == "train":
        #     label = int(fname.split("/")[-1].split("_")[0])
        #     print(label)
        
        img_name = fname.split("/")[-1] #.split("_")[0]
        return im, img_name #, label



#  ./hw3_data/p1_data/id2label.json
#  ./hw3_data/p1_data/val/*.png

# image = preprocess(Image.open("hw3_data/p1_data/val/0_499cd.png")).unsqueeze(0).to(device)
# data_target_iter = iter(train_dataloader_target)
# for i in range(len_dataloader):
#     data_source = data_target_iter.next()
#     s_img, s_label = data_source

def same_seeds(seed):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)


def read_json(json_name):
    with open(json_name) as f:
        id2label = json.load(f)
    # id = 0 
    # print("0:",id2label[str(id)])
    return id2label




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="p 3-1",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("src_path", help="src_path")
    parser.add_argument("json_name", help="json_name")
    parser.add_argument("output_root", help="output_root")
    parser.add_argument("--template_prompt", help="template_prompt", default = "This is a {item} image.{object}") # "This is a {object} image.couch": 0.69
    args = parser.parse_args()
    src_path = args.src_path # src_path = "hw3_data/p1_data/val"
    json_name = args.json_name # json_name = './hw3_data/p1_data/id2label.json'
    output_root = args.output_root # output_name = "./output_p1/pred.csv"
    template_prompt = args.template_prompt
    
    print(args)
    seed = 1211
    same_seeds(seed) 

    if torch.cuda.is_available():
        if torch.cuda.device_count()==2:
            device = torch.device("cuda:1")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # device = torch.device("cuda")
    print("Using ", device)
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # Create  prompt and tokenize text.
    #template_prompt = "This is a photo of "
    id2label = read_json(json_name)

    prompt_list = []
    for k, v in id2label.items():
        # print(v)
        prompt = template_prompt.replace("{object}", v)
        # prompt = template_prompt + v
        prompt_list.append(prompt) 
    # print(prompt_list)
    #text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
    text = clip.tokenize(prompt_list).to(device)

    
    

    # Create image dataset
    batch_size = 1
    dataset = ImageDataset(src_path, tfm=preprocess)
    # image = preprocess(Image.open("hw3_data/p1_data/val/0_499.png")).unsqueeze(0).to(device)

    dataloader  = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True, # False
        num_workers=8
    )
    len_dataloader = len(dataloader)
    data_iter = iter(dataloader)
    
    pred_list = []
    filename_list = []
    os.makedirs(output_root, exist_ok = True)
    k = 5
    for i in range(len_dataloader):
        data = data_iter.next()
        image, file_name = data
        # print(file_name)

        image = image.to(device)
        # pytorch中model eval和torch no grad()的区别: https://blog.csdn.net/songyu0120/article/details/103884586
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text) 
            
            logits_per_image, logits_per_text = model(image, text)

            # sprint(logits_per_image)
            # print(logits_per_text)
            # print(logits_per_image.shape)
            # print(logits_per_text.shape)
            probs_img = logits_per_image.softmax(dim=-1)
            #probs_text = logits_per_text.softmax(dim=1).cpu().numpy()

        # print("Label image probs:", probs_img)
        preds_topk_logits, preds_topk_ids = probs_img.topk(k)

        # print(preds_topk_ids)
        # print(preds_topk_logits)

        # read class name
        class_names = []
        list_ = preds_topk_ids[0].cpu().numpy().tolist()
        for id_ in list_:
            class_names.append(id2label[str(id_)])
        # read logits
        preds_topk_logits = preds_topk_logits[0].cpu().numpy().tolist()
        # print("class_names",class_names)
        # print("preds_topk_ids",preds_topk_ids[0].cpu().numpy().tolist())
        # print("preds_topk_logits", preds_topk_logits)
        
        # save image
        correct_name = id2label[str(file_name[0].split('_')[0])] 
        figure = plt.imshow(image[0].permute(1,2,0).cpu().numpy()) # 显示图片
        plt.title(f"Correct label: {correct_name}")
        plt.axis('off') # 不显示坐标轴
        plt.show()
        saveas = os.path.join(output_root,file_name[0])
        print(saveas)
        plt.savefig(saveas)
        plt.clf()

        # create colorbar
        cmap = plt.get_cmap('jet')
        low = cmap(0.5)
        medium_rare = cmap(0.65)
        medium =cmap(0.25)
        high_rare = cmap(0.35)
        high = cmap(0.8)
        c = [high, high_rare, medium, medium_rare, low]

        # save histogram
        x = list(range(0, k))
        figure = plt.bar(x, preds_topk_logits, color=c, label=class_names)
        # https://datagy.io/matplotlib-bar-charts/
        plt.title(f"Correct similarity score: {preds_topk_logits[0]:.3f}")
        plt.legend()
        plt.show()
        saveas = os.path.join(output_root,f'{file_name[0]}_vis.png')
        print(saveas)
        plt.savefig(saveas)
        plt.clf()

        if i==7:
            break
