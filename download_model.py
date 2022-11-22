import gdown
import clip

url = url
output = "ckpt_encoder_continue"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()
