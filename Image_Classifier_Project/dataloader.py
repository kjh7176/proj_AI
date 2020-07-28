#                                                                             
# PROGRAMMER: Juhyeon Kim
# DATE CREATED: 27.07.2020                                  
# REVISED DATE: 
# PURPOSE: Creates dataloaders for training, validaiton and test after image transformation
#
#
from torchvision import datasets, transforms
import torch

# preprocess images form directory and return trainloader
def get_trainloader(filedir):
    train_transforms = transforms.Compose([transforms.Resize(225),
                                           transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    train_datasets = datasets.ImageFolder(filedir, transform=train_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    
    return trainloader, train_datasets.class_to_idx

# preprocess images form directory and return validloader
def get_validloader(filedir):
    valid_transforms = transforms.Compose([transforms.Resize(225),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    valid_datasets = datasets.ImageFolder(filedir, transform=valid_transforms)
    
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=64)
    
    return validloader

# preprocess images form directory and return testloader
def get_testloader(filedir):
    test_transforms = transforms.Compose([transforms.Resize(225),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    test_datasets = datasets.ImageFolder(filedir, transform=test_transforms)
    
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64)
    
    return testloader