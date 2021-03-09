import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image

from pydub import AudioSegment
import numpy as np
import scipy as sp
from scipy.io.wavfile import read
from scipy.io.wavfile import write     # Imported libaries such as numpy, scipy(read, write), matplotlib.pyplot
from scipy import signal
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd


train_on_gpu = torch.cuda.is_available()

def get_model(path):   
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model = models.resnet18(pretrained=True)
    # Make sure to set parameters as not trainable
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 199)
    # Load in the state dict
    model.load_state_dict(checkpoint['state_dict'])
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params} total gradient parameters.')
    if train_on_gpu:
        model = model.to('cuda')
    # Model basics
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    model.epochs = checkpoint['epochs']
    return model



def process_image(image_path):
    image = Image.open(image_path)
    img = image.resize((256, 256))
    # Center crop
    width = 256
    height = 256
    new_width = 224
    new_height = 224

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img = img.crop((left, top, right, bottom))
    # Convert to numpy, transpose color dimension and normalize
    img = np.array(img).transpose((2, 0, 1)) / 256
    # Standardization
    means = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    stds = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = img - means
    img = img / stds
    img_tensor = torch.Tensor(img).unsqueeze(0)
    img_tensor = torch.Tensor(img)
    return img_tensor

############################################################################
# def predict(image_path, model, topk=5):
#     # Convert to pytorch tensor
#     img_tensor = process_image(image_path)
#     # Resize
#     if train_on_gpu:
#         img_tensor = img_tensor.view(1, 3, 224, 224).cuda()
#     else:
#         img_tensor = img_tensor.view(1, 3, 224, 224)
#     # Set to evaluation
#     with torch.no_grad():
#         model.eval()
#         # Model outputs log probabilities
#         out = model(img_tensor)
#         ps = torch.exp(out)
#         # Find the topk predictions
#         topk, topclass = ps.topk(topk, dim=1)
#         # Extract the actual classes and probabilities
#         top_classes = [
#             model.idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]
#         ]
#         top_p = topk.cpu().numpy()[0]
#         return img_tensor.cpu().squeeze(), top_p, top_classes
#         # return top_classes[0]

#############################################################################

def predict(image_path, model, topk=5):
    """Make a prediction for an image using a trained model

    Params
    --------
        image_path (str): filename of the image
        model (PyTorch model): trained model for inference
        topk (int): number of top predictions to return

    Returns

    """
    real_class = image_path.split('/')[-2]

    # Convert to pytorch tensor
    img_tensor = process_image(image_path)

    # Resize
    if train_on_gpu:
        img_tensor = img_tensor.view(1, 3, 224, 224).cuda()
    else:
        img_tensor = img_tensor.view(1, 3, 224, 224)

    # Set to evaluation
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(img_tensor)
        ps = torch.exp(out)

        # Find the topk predictions
        topk, topclass = ps.topk(topk, dim=1)

        # Extract the actual classes and probabilities
        top_classes = [
            model.idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]
        ]
        top_p = topk.cpu().numpy()[0]

        return img_tensor.cpu().squeeze(), top_p, top_classes, real_class





