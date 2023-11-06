import torch
from torch import nn
import torchvision.models as models
from PIL import Image
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

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
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    return model, criterion, optimizer, scheduler

def load_model(path):
    checkpoint = torch.load(path)
    structure = checkpoint['structure']
    model, _, _, _ = nn_setup(structure)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img_pil = Image.open(image)
    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    
    image = img_transforms(img_pil)
    
    return image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to('cuda')
    model.eval()
    img = process_image(image_path).numpy()
    img = torch.from_numpy(np.array([img])).float()

    with torch.no_grad():
        logps = model.forward(img.cuda())
        
    probability = torch.exp(logps).data
    
    return probability.topk(topk)

image_path = "flowers/test/10/image_07090.jpg"
model = load_model('checkpoint.pth')
probs, classes = predict(image_path, model)
print (probs)
print (classes)

# TODO: Display an image along with the top 5 classes
plt.rcdefaults()
fig, ax = plt.subplots()

test_dir = './flowers/test'
index = 1
path = test_dir + '/1/image_06743.jpg'
ps = predict(path, model)
image = process_image(path)

ax1 = imshow(image, ax = plt)
ax1.axis('off')
ax1.title(cat_to_name[str(index)])


a = np.array(ps[0][0].cpu())
b = [cat_to_name[str(index+1)] for index in np.array(ps[1][0].cpu())]

fig,ax2 = plt.subplots(figsize=(5,5))


y_pos = np.arange(5)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(b)
ax2.set_xlabel('Probability')
ax2.invert_yaxis()
ax2.barh(y_pos, a)

plt.show()

