import gdown
import clip

model, preprocess = clip.load("ViT-B/32")
model.eval()

url = "https://drive.google.com/file/d/1iQ1sfLC7gXM7ukVa8neFsS-Z4gZaHcoV/view?usp=share_link"
output = "ckpt_encoder_continue.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

