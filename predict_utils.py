"""
This is the "train_utils" file.

The train_utils file contains functions that are needed for the image
prediction module.
"""

import argparse
from PIL import Image
import numpy as np
import torch
import json


def get_input_args():
    """
    Gets the command-line arguments passed by the user. This function
    retrieves  the 5 Command Line Arugment from the user running the
    program from a terminal window.

    Arguments: None

    Returns:
        parser.parse_args() (obj): An object which contains all the passed
        arguments
    """
    parser = argparse.ArgumentParser('This is the prediction module')

    parser.add_argument('input', type=str,
                        help='Location/path of the image to be predicted')
    parser.add_argument('checkpoint', type=str,
                        help='Location/path of the saved model')
    parser.add_argument('--top_k', type=int, default=5,
                        help='The number of top predicted classes to return')
    parser.add_argument('--category_names', type=str,
                        default='cat_to_name.json',
                        help='The learning rate to train the model on')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for inference')

    return parser.parse_args()


def process_image(image_path):
    '''
    Scales, crops, and normalizes a PIL image for a PyTorch model.

    Arguments:
        image_path (str): The path of the image to be processed

    Returns: A tensor of the processed image
    '''
    image = Image.open(image_path)
    # Make a copy of the image so as to not affect the original image
    img_pp = image.copy()
    if img_pp.size[0] > img_pp.size[1]:
        # Width is greater so resize based on the height
        img_pp.thumbnail((img_pp.size[0], 256))
    else:
        # Height is greater so resize based on the width
        img_pp.thumbnail((256, img_pp.size[1]))

    left = (img_pp.size[0] - 224) / 2
    upper = (img_pp.size[1] - 224) / 2
    right = left + 224
    lower = upper + 224
    img_crop = img_pp.crop((left, upper, right, lower))
    np_img = np.array(img_crop)
    # The network expects the images to be normalized with these
    # mean and standarad deviation values
    img_mean = np.array([0.485, 0.456, 0.406])
    img_stdev = np.array([0.229, 0.224, 0.225])
    # Since color values range from 0 to 255 inorder to normalie it
    # to a range of 0 to 1 we can simply divide the alues by 255
    img_nrml_one = np_img / 255
    img_nrml_two = (img_nrml_one - img_mean) / img_stdev
    # PyTorch requires the color channel to be the first dimention,
    # so we can rearrange that by transposing the array
    img_t = img_nrml_two.transpose(2, 0, 1)

    return torch.from_numpy(img_t).type(torch.FloatTensor)


def predict(image_path, model, top_k, device):
    '''
    Predict the class/classes of an image using a trained deep learning model.

    Arguments:
        - image_path (str): Path of the image to be predicted
        - model (obj): The trained model
        - tok_k (int): The number of top most likely predicted classes
        - device (str): The default device (CPU or GPU)

    Returns:
        probs, classes (tuple): A tuple containing the probs and clases
            - probs (tensor):The probabilities of the most likely predicted
                                    classes
            - classes (list): The categories of the most likely predicted
                                    classes
    '''
    model.to(device)
    # Inverts the class_to_idx dictionary so that we can get the categories
    inv_class_to_idx = {
        ix: cls for cls,
        ix in model.class_to_idx.items()}  # invert the dictionary
    # Processes the image
    img = process_image(image_path)
    img = img.to(device)

    with torch.no_grad():
        logps = model.forward(img.unsqueeze(0))
        ps = torch.exp(logps)
        probs, indexes = ps.topk(top_k, dim=1)

    classes = []
    # Converts the indexes from a 2D tensor to a 1D numpy array
    # to continue the operation in the for loop
    for index in indexes.cpu().numpy().flatten():
        classes.append(inv_class_to_idx[index])

    return probs, classes


def load_cat_to_name(cat_to_name_path):
    """
    Loads the file that contains the mapping between class category and name

    Arguments:
        cat_to_name_path (str): The path of the mapping file

    Returns:
        cat_to_name (dict): A dictionary containing key value pairs of category
                                to
                            name mapping
    """
    with open(cat_to_name_path, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name
