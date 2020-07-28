#                                                                             
# PROGRAMMER: Juhyeon Kim
# DATE CREATED: 27.07.2020                                  
# REVISED DATE: 
# PURPOSE: Loads a checkpoint in order to rebuild the trained model. And then
#          predicts the class from input image using it.
#
from torchvision import models
from torch import optim
import torch
import numpy as np

from trainer import Classifier

# load model from checkpoint
def load_checkpoint(filename, device):
    checkpoint = torch.load(filename)
    
    if(checkpoint['arch'] == 'vgg16'):
        model = models.vgg16(pretrained=True)
    elif(checkpoint['arch'] == 'densenet121'):
        model = models.densenet121(pretrained=True)
    else:
        None
    
    for params in model.parameters():
        params.requires_grad = False
            
    model.classifier = Classifier(checkpoint['hidden_units'], checkpoint['arch'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model = model.to(device)
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['lr'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer

# process the image as input for the model
def process_image(image):
    image = image.resize((256, 256))
    image = image.crop((16, 16, 240, 240))
    image = np.array(image)
    image = image / 255
    
    mean = np.array([0.485,0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    image = image.transpose((2, 0, 1))
    
    return torch.from_numpy(image)

