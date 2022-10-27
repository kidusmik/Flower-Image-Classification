#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PROGRAMMER: Kidus Michael Worku
"""
This is the "test" file.

The test file is the entry point for testing a saved model. It test the checkpoint
or model that is passed by the user with the test images/datasets located in the 
data directory which its location is also passed by the user.

The test module requires two positional arguments and one optional aragument to 
be passed as input from the user running the program from a terminal window. The 
module tests an image based on athe data and checkpoint provided, it tests 
the image with the default setting or according to the optional arguments passed by 
the user such as whether to use GPU for testing. The test module finally prints the 
test loss and test accuracy.

FORMAT
------
./test.py checkpoint data_dir [--gpu]
                                

CLI ARGUMENTS
-------------
- checkpoint(str) (positional): Path/Location of the saved model
- data_dir(str) (positional): Path/location of the training data
- gpu(bool) (optional): Specifies whether to use GPU for training
"""

import torch
import argparse
from torch import nn

from train_utils import get_image_datasets, get_image_loader, test_saved_model, get_device


def get_input_args():
    """
    Gets the command-line arguments passed by the user. This function retrieves 
    the 3 Command Line Arugment from the user running the program from a terminal window.
    
    Arguments: None
    
    Returns:
        parser.parse_args() (obj): An object which contains all the passed arguments
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
    # Retreive the commandline arguments    
    input_arguments = get_input_args()
    # Retreives the image datasets to get the test datasets
    image_datasets = get_image_datasets(input_arguments.data_dir)
    # Retreives the test image loader
    test_loader = get_image_loader(image_datasets)['test']
    criterion = nn.NLLLoss()
    # Retreives the default device to use for training
    device = get_device(input_arguments.gpu)
    torch.device(device)
    # Test the saved model
    test_saved_model(input_arguments.checkpoint, test_loader, device, criterion)

    
# Call to main function to run the program
if __name__ == "__main__":
    main()
    