#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PROGRAMMER: Kidus Michael Worku
"""
This is the "train" file.

The train file is the entry point for training a model. It trains a model
using a selected model architecture, the available model architectures are
VGG (vgg11), ALEXNET and DENSENET (densenet121).

The train module requires one positional argument and 6 optional araguments
to be passed as input from the user running the program from a terminal
window. The module trains the model based on athe training data provided and
trains the model with the default setting or according to the optional
arguments passed by the user such as the architecture, laerning rate, number
of hidden units, total training epocch and whether to use GPU for the training.
It displays the training loss, vvalidation loss, and validation accuracy while
training. After training is complete it saves the trained model and test it and
displays the test accuracy.

FORMAT
------
./train.py data_dir [--save_dir SAVE_DIR] [--arch {vgg,alexnet,densenet}]
[--learning_rate LEARNING_RATE] [--hidden_units HIDDEN_UNITS] [--epochs EPOCHS]
                                                              [--gpu]

CLI ARGUMENTS
-------------
- data_dir(str) (positional): Path/location of the training data
- save_dir(str) (optional): Location to save the trained the model to
    - default: 'saved_models'
- arch(str) (optional): The model architecture to use for training
    - default: 'vgg'
- learning_rate(float) (optional): The learning rate to train the model
    - default: 0.001
- hidden_units(int) (optional): The number of hidden units in the layer
    - default: 667
- epochs(int) (optional): The total numbaro of epochs to train the model
    - default: 1
- gpu(bool) (optional): Specifies whether to use GPU for training
"""

import torch
from torch import nn, optim
from time import time

import classifier_network
import train_utils


def main():
    # Retreive the commandline arguments
    input_arguments = train_utils.get_input_args()
    # Fetch the model from torch
    model = train_utils.get_model(input_arguments.arch)
    # Keep
    for param in model.parameters():
        param.requires_grad = False

    input_size = None
    if input_arguments.arch == 'vgg':
        input_size = 25088  # Default input size of vgg11 model
    elif input_arguments.arch == 'alexnet':
        input_size = 9216  # Default input size of alexnet model
    elif input_arguments.arch == 'densenet':
        input_size = 1024  # Default input sizez of densenet121 model
    # Create the classifer network for the model
    classifier = classifier_network.Network(input_size, 102,
                                            [input_arguments.hidden_units])
    model.classifier = classifier
    # Retreives the train, validation and test image datasets
    image_datasets = train_utils.get_image_datasets(input_arguments.data_dir)
    # Retreives the train, validation adn test image loader
    image_loader = train_utils.get_image_loader(image_datasets)
    # Retreives the default device to use for training
    device = train_utils.get_device(input_arguments.gpu)
    torch.device(device)

    start_time = time()

    criterion = nn.NLLLoss()
    # Only training the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(),
                           lr=input_arguments.learning_rate)
    model.to(device)

    train_loader = image_loader['train']
    valid_loader = image_loader['valid']
    saved, checkpoint_path = False, None
    model.train()
    running_loss = 0
    print_every = 5

    print('\n===== MODEL TRAINING STARTED =====')
    print('\t[-] ARCHITECTURE: {}'.format(input_arguments.arch.upper()))
    print('\t[-] LEARNING RATE: {}'.format(input_arguments.learning_rate))
    print('\t[-] HIDDEN UNITS: {}'.format(input_arguments.hidden_units))
    print('\t[-] EPOCHS: {}\n'.format(input_arguments.epochs))

    try:
        for epoch in range(input_arguments.epochs):
            steps = 0
            for inputs, labels in train_loader:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in valid_loader:
                            inputs, labels = inputs.to(device),\
                                             labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            valid_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(
                                equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{input_arguments.epochs}.. "
                          f"Step {steps}/{len(train_loader)}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: \
{valid_loss/len(valid_loader):.3f}.. "
                          "Validation accuracy: {:.3f}%"
                          .format(accuracy/len(valid_loader) * 100))
                    running_loss = 0
                    # Sets the model back to training mode
                    model.train()

        end_time = time()
        total_time = end_time - start_time  # Caculates the total elapsd time
        print("\n\t[-] Total Elapsed time:",
              str(int((total_time/3600)))+":"+str(
                  int((total_time % 3600)/60))+":"+str(
                  int((total_time % 3600) % 60)))
        print('===== MODEL TRAINING COMPLETED =====\n')
    except KeyboardInterrupt:
        # If training is interrupted by the user ask the user whether to save
        # the trained model or not
        choice = input('\nDo you want to save the trained model? [y/N]: ')
        if choice == 'y' or choice == 'Y':
            saved, checkpoint_path = train_utils.save_model(
                model.cpu(), optimizer, image_datasets, input_arguments.arch,
                input_arguments.save_dir, input_size,
                input_arguments.learning_rate)
    else:
        # If training completes without interuption or exception it is saved
        # automatically
        saved, checkpoint_path = train_utils.save_model(
            model.cpu(), optimizer, image_datasets, input_arguments.arch,
            input_arguments.save_dir, input_size,
            input_arguments.learning_rate)

    if saved:
        # If the model is saved then this will test the saved trained model
        train_utils.test_saved_model(
            checkpoint_path, image_loader['test'], device, criterion)


# Call to main function to run the program
if __name__ == "__main__":
    main()
