import torch
import clip
from PIL import Image


if torch.cuda.is_available():
    if torch.cuda.device_count()==2:
        device = torch.device("cuda:1")
    else:
        device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using ", device)

model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("hw3_data/p3_data/images/sheep.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]