#                                                                             
# PROGRAMMER: Juhyeon Kim
# DATE CREATED: 27.07.2020                                  
# REVISED DATE: 
# PURPOSE: Builds pre-trained network and defines a new classifier so that train 
#          the classifier layers using backpropagation using the pre-trained 
#          network to get the features
#
from torch import nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, hidden_units, arch):
        super().__init__()
        if(arch == 'vgg16'):
            self.fc1 = nn.Linear(25088, hidden_units)
            self.fc2 = nn.Linear(hidden_units, 102)
        elif(arch == 'densenet121'):
            self.fc1 = nn.Linear(1024, hidden_units)
            self.fc2 = nn.Linear(hidden_units, 102)
        
    def forward(self, x):
        dropout = nn.Dropout(0.02)
        
        x = x.view(x.shape[0], -1)
        x = dropout(F.relu(self.fc1(x)))
        x = F.log_softmax(self.fc2(x), dim=1)
        
        return x
    