#                                                                             
# PROGRAMMER: Juhyeon Kim
# DATE CREATED: 21.07.2020                                  
# REVISED DATE: 
# PURPOSE: Predict the class from input image using trained network
#
#
from PIL import Image
import torch
import json

from predict_args import get_parse_args
from predicter import load_checkpoint, process_image

# get parse arguments
args = get_parse_args()

device = torch.device("cuda" if args.gpu else "cpu")

# process an image file
image = Image.open(args.image_path)
image = process_image(image)
image.unsqueeze_(0)
image = image.to(device)
image = image.float()

# load checkpoint
model, optimizer = load_checkpoint(args.checkpoint, device)
model.eval()

# predict the class of the image
with torch.no_grad():
    output = model(image)
    probs = torch.exp(output)
    top_p, top_class = probs.topk(args.top_k, dim=1)

# convert classes to flower names
idx_to_class = {value:key for key, value in model.class_to_idx.items()}
classes = [idx_to_class[x.item()] for x in top_class[0]]
with open(args.category_names, 'r') as f:
    cat_to_names = json.load(f)
flower_names = [cat_to_names[x] for x in classes]
    
for i in range(args.top_k):
    print(f'{i+1} : {flower_names[i]} ({top_p[0][i]:.3f})')