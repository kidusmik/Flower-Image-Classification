import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import json


def get_input_args():
    """
    Gets the command-line arguments passed by the user.
    """
    parser = argparse.ArgumentParser('This is the prediction module') 
    
    parser.add_argument('input', type=str,
                        help='Location/path of the image to be predicted') 
    parser.add_argument('checkpoint', type=str,
                        help='Location/path of the saved model') 
    parser.add_argument('--top_k', type=int, default=5,
                        help='The number of top predicted classes to return')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='The learning rate to train the model on') 
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for inference')    
    
    return parser.parse_args() 
    
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # DONE: Process a PIL image for use in a PyTorch model
    image = Image.open(image_path)
    img_pp = image.copy()
    if img_pp.size[0] > img_pp.size[1]:
        img_pp.thumbnail((img_pp.size[0], 256))
    else:
        img_pp.thumbnail((256, img_pp.size[1]))

    left = (img_pp.size[0] - 224) / 2
    upper = (img_pp.size[1] - 224) / 2
    right = left + 224
    lower = upper + 224
    
    img_crop = img_pp.crop((left, upper, right, lower))

    np_img = np.array(img_crop)

    img_mean = np.array([0.485, 0.456, 0.406])
    img_stdev = np.array([0.229, 0.224, 0.225])

    img_nrml_one = np_img / 255
    img_nrml_two = (img_nrml_one - img_mean) / img_stdev

    img_t = img_nrml_two.transpose(2, 0, 1)
    
    return torch.from_numpy(img_t).type(torch.FloatTensor) 


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
        
    if title:
        plt.title(title)
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    # image = image.transpose((1, 2, 0))
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, top_k, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # DONE: Implement the code to predict the class from an image file
    
    model.to(device)
    inv_class_to_idx = {ix: cls for cls, ix in model.class_to_idx.items()} # invert the dictionary

    img = process_image(image_path)
    img = img.to(device)

    with torch.no_grad():
        logps = model.forward(img.unsqueeze(0))
        ps = torch.exp(logps)
        probs, indexes = ps.topk(top_k, dim=1)
    
    classes = []    
    for index in indexes.cpu().numpy().flatten():
            classes.append(inv_class_to_idx[index])

    return probs, classes

def load_cat_to_name(cat_to_name_path):
    with open(cat_to_name_path, 'r') as f:
        cat_to_name = json.load(f)
        
    return cat_to_name