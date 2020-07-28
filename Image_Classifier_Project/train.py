#                                                                             
# PROGRAMMER: Juhyeon Kim
# DATE CREATED: 21.07.2020                                  
# REVISED DATE: 
# PURPOSE: Train the network with parmaters entered as arguments
#
#
import torch
from torch import nn
from torchvision import models
from torch import optim

from train_args import get_parse_args
from trainer import Classifier
from workspace_utils import active_session
from dataloader import get_trainloader, get_validloader, get_testloader

from predicter import load_checkpoint

# get parse arguments
args = get_parse_args()
    
device = torch.device("cuda" if args.gpu else "cpu")

with active_session():
    
    # preprocess and load image data
    trainloader, class_to_idx = get_trainloader(args.data_dir + '/train')
    validloader = get_validloader(args.data_dir + '/valid')
    testloader = get_testloader(args.data_dir + '/test')

    # load saved model from checkpoint if args.load is True
    if(args.load):
        model, optimizer = load_checkpoint(args.save_dir, device)
    # construct new model if args.load is False
    else:
        if(args.arch == 'vgg16'):
            model = models.vgg16(pretrained=True)
        elif(args.arch == 'densenet121'):
            model = models.densenet121(pretrained=True)
        else:
            None
            
        for params in model.parameters():
            params.requires_grad = False
        
        model.classifier = Classifier(args.hidden_units, args.arch)
        model.class_to_idx = class_to_idx
        model = model.to(device)
        
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    criterion = nn.NLLLoss()

    for e in range(args.epochs):
        training_loss = 0
        validation_loss = 0
        accuracy = 0
            
        # training
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item()
            
        # validation
        model.eval()
        with torch.no_grad():
            for images, labels in validloader:
                images, labels = images.to(device), labels.to(device)
                
                output = model(images)
                loss = criterion(output, labels)
                validation_loss += loss.item()
                
                prob = torch.exp(output)
                top_p, top_index = prob.topk(1, dim=1)
                equals = top_index == labels.view(top_index.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
        # prints loss and accuracy at the end of each epoch
        print(f"{e+1}..training_loss : {training_loss/len(trainloader):.3f}.."
              f"validation_loss : {validation_loss/len(validloader):.3f}.."
              f"accuracy : {accuracy/len(validloader):.3f}")
    
    # test
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            
            output = model(images)
            prob = torch.exp(output)
            top_p, top_index = prob.topk(1, dim=1)
            equals = top_index == labels.view(top_index.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    print(f"\ntesting accuracy : {accuracy/len(testloader):.3f}..")
         
# save the model as a checkpoint
torch.save({'model_state_dict': model.state_dict(),
            'class_to_idx' : model.class_to_idx,
            'optimizer_state_dict': optimizer.state_dict(),
            'arch' : args.arch,
            'hidden_units' : args.hidden_units,
            'lr' : args.learning_rate}, 
           args.save_dir)
    
    
    
