"""
This is the "get_input_args" file.

The get_input_args file parses and gets the command-line input arguments.
"""

import argparse

def get_input_args(description):
    """
    Gets the command-line arguments passed by the user.
    """
    parser = argparse.ArgumentParser(description=description) 
    
    parser.add_argument('data_dir', type=str,
                        help='Path to the directory containing the training data set') 
    parser.add_argument('--save_dir', type=str, default='saved_models/',
                        help='Path of the directory to save the checkpoints to') 
    parser.add_argument('--arch', type=str, default='vgg',
                        help='Type of CNN model architecture to use', choices=['vgg', 'alexnet', 'resnet'])
    parser.add_argument('--learning_rate', type=int, default=0.001,
                        help='The learning rate to train the model on') 
    parser.add_argument('--hidden_units', type=int, default=667,
                        help='Size of the hidden unit') 
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs to train the model') 
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU to train the model')    
    
    return parser.parse_args() 