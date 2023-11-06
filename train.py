import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image
import json
from collections import OrderedDict 
import torchvision

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
data_transforms_training = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


data_transforms_testing = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
data_transforms_validating = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])



# TODO: Load the datasets with ImageFolder
image_datasets_training =  datasets.ImageFolder(train_dir, transform = data_transforms_training)
image_datasets_testing = datasets.ImageFolder(test_dir, transform = data_transforms_testing)
image_datasets_validating = datasets.ImageFolder(valid_dir, transform = data_transforms_validating)

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders_training = torch.utils.data.DataLoader(image_datasets_training, batch_size = 64, shuffle = True)
dataloaders_testing = torch.utils.data.DataLoader(image_datasets_testing, batch_size = 64, shuffle = True)
dataloaders_validating = torch.utils.data.DataLoader(image_datasets_validating, batch_size = 64, shuffle = True)

# TODO: Build and train your network

def nn_setup(structure='vgg16', lr=0.001):
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(25088, 2048),
        nn.ReLU(),
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Linear(256, 102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier
    model = model.to('cuda')

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    return model, criterion, optimizer, scheduler

setup_result = nn_setup()
model = setup_result[0]
criterion = setup_result[1]
optimizer = setup_result[2]
print(model)

epochs = 3
print_every = 5
steps = 0
loss_show = []

for e in range(epochs):
    running_loss = 0
    
    for inputs, labels in dataloaders_training:
        steps += 1
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
        
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            model.eval()
            valid_loss = 0
            accuracy = 0
            
            with torch.no_grad():
                for inputs, labels in dataloaders_validating:
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')
                    log_ps = model.forward(inputs)
                    batch_loss = criterion(log_ps, labels)
                    valid_loss += batch_loss.item()
                    
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            average_train_loss = running_loss / print_every
            average_valid_loss = valid_loss / len(dataloaders_validating)
            average_accuracy = accuracy / len(dataloaders_validating)
            
            loss_show.append(average_train_loss)
            
            print(f"Epoch {e+1}/{epochs}.. "
                  f"Loss: {average_train_loss:.3f}.. "
                  f"Validation Loss: {average_valid_loss:.3f}.. "
                  f"Accuracy: {average_accuracy:.3f}")
            
            running_loss = 0
            model.train()
## Testing your network

test_loss = 0
accuracy = 0
model.to('cuda')

with torch.no_grad():
    for inputs, labels in dataloaders_testing:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        log_ps = model.forward(inputs)
        batch_loss = criterion(log_ps, labels)
                    
        test_loss += batch_loss.item()
                    
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
print(f"Test accuracy: {accuracy/len(dataloaders_testing):.3f}")


# TODO: Save the checkpoint 
model.class_to_idx = image_datasets_training.class_to_idx
checkpoint = {
    'input_size': 25088,
    'output_size': 102,
    'structure': 'vgg16',
    'learning_rate': 0.001,
    'classifier': model.classifier,
    'epochs': epochs,
    'optimizer': optimizer.state_dict(),
    'state_dict': model.state_dict(),
    'class_to_idx': model.class_to_idx
}

try:
    torch.save(checkpoint, 'checkpoint.pth')
    print("Checkpoint saved successfully.")
except Exception as e:
    print("An error occurred while saving the checkpoint.")
    print(e)