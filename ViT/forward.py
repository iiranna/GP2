import numpy as np
from PIL import Image
import torch


imagenet_labels = dict(enumerate(open("classes.txt")))

model = torch.load("model.pth")
model.eval()

img = (np.array(Image.open("benign.png")) / 128) -1
inp = torch.fromnumpy(img).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
logits = model(inp)
probs = torch.nn.functional.softmax(logits, dim = -1)

top_probs, top_ixs = probs[0].topk(k)

for i, (ix_, prob_) in enumerate(zip(top_ixs, top_probs)):
  ix = ix_.item()
  prob = prob.item()
  cls = imagenet_labels[x].strip()
  print(f"{i}: {cls:<45} --- {prob:.4f}")
  
