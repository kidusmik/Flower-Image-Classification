#!/usr/bin/env python3
import torch
import argparse
from torch import nn

from train_utils import get_image_datasets, get_image_loader, test_saved_model, get_device


def get_input_args():
    """
    Gets the command-line arguments passed by the user.
    """
    parser = argparse.ArgumentParser('This is the model testing module') 
    
    parser.add_argument('checkpoint', type=str,
                        help='Location/path of the saved model')  
    parser.add_argument('data_dir', type=str,
                        help='Path to the directory containing the training data set') 
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU to train the model')    
    
    return parser.parse_args()

def main():
    input_arguments = get_input_args()
    image_datasets = get_image_datasets(input_arguments.data_dir)
    test_loader = get_image_loader(image_datasets)['test']
    criterion = nn.NLLLoss()

    device = get_device(input_arguments.gpu)
    torch.device(device)
    
    test_saved_model(input_arguments.checkpoint, test_loader, device, criterion)

    
# Call to main function to run the program
if __name__ == "__main__":
    main()
    