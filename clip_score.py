import clip
import torch

def getCLIPScore(image, text, device, w=2.5):
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    # print(image)
    # print(text_ids)
    image_embedding = model.encode_image(image)
    #print(image_embedding.shape)
    text_ids = clip.tokenize(text).to(device)
    #print("text_ids", text_ids)
    text_embedding = model.encode_text(text_ids)
    # cos_similarity =  a*b/|a|/|b|
    cos = torch.nn.CosineSimilarity(dim=1)

    score = cos(image_embedding, text_embedding)

    mean_score = score.mean()
    # print(mean_score)
    return w*max(mean_score, 0)

