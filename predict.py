#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PROGRAMMER: Kidus Michael Worku
"""
This is the "predict" file.

The predict file is the entry point for predicting an image. It predicts
an image passed by the user using a saved checkpoint or model.

The predict module requires two positional arguments and 3 optional
araguments to be passed as input from the user running the program from a
terminal window. The module predicts an image based on athe image and
checkpoint provided, it predicts the image with the default setting or
according to the optional arguments passed by the user such as the numbar
of top K predictions to return, the image label mapping and whether to use
GPU for the training. The predict module finally prints the top prediction,
the prediction accuracy and a table listing the Top K classes and their
respective probability.

FORMAT
------
./predict.py input checkpoint [--top_k TOP_K] [--category_names CATEGORY_NAMES]
                                              [--gpu]


CLI ARGUMENTS
-------------
- input(str) (positional): Path/location of the image to be predicted
- checkpoint(str) (positional): Path/Location of the saved model
- top_k(int) (optional): The number of top most likely predicted classes
    - default: 5
- category_names(str) (optional): Path/Location of the mapping of categories
                                  to names
    - default: 'cat_to_name'
- gpu(bool) (optional): Specifies whether to use GPU for training
"""
import torch
from pathlib import Path

import classifier_network
import predict_utils
from train_utils import load_checkpoint, get_device


def main():
    # Retreive the commandline arguments
    input_arguments = predict_utils.get_input_args()
    # Load the saved trained model/checkpoint
    model, optimizer = load_checkpoint(input_arguments.checkpoint)
    # Load the mappinag of categories to names
    cat_to_name = predict_utils.load_cat_to_name(
        input_arguments.category_names)
    # Retreives the default device to use for training
    device = get_device(input_arguments.gpu)
    torch.device(device)

    # Assuming image path as such: 'flowers/train/14/image_06058.jpg', this
    # try/except block will try to retreive the category from the path which
    # in this case is '14'. But, if the user passed path format does not match
    # this format, then it will assume the category as the image name itself,
    # which in this case is 'image_06058.jpg'. I did this just in case the user
    # uses a category to name mapping where the keys are the image file names.
    class_integer = None
    img_path = input_arguments.input
    try:
        # Tries to retreive the class integer from path
        class_cat = img_path.split('/')[2]
    except Exception:
        # Otherwise will just retreive the file name
        class_cat = Path(input_arguments.input).name
    # Predict the image
    probs, classes = predict_utils.predict(
        input_arguments.input, model, input_arguments.top_k, device)
    # Flattens the 2D array to 1D
    probs_flat = probs.cpu().numpy().flatten()
    # Retreives the class names of the categories from the mapping
    names = [cat_to_name[cls] for cls in classes]
    # A list comprising of the total number of Ks: [1, 2, ..., k]
    top_ks = list(range(1, input_arguments.top_k+1))

    print('\t\t===== PREDICTION RESULT =====\n')

    if class_cat is not None and class_cat in cat_to_name:
        # Prints the actual class name
        print('[-] {:>25}\t{:<20}'.format(
            'Image Class:', cat_to_name[class_cat]))
    print('[-] {:>25}\t{:<20}'.format('Predicted Image Class:', names[0]))
    print('[-] {:>25}\t{:.3f}%'.format(
        'Prediction Accuracy:', probs_flat[0]*100))
    if class_cat is not None and class_cat in cat_to_name:
        # Prints whether the prediction is correct or not
        print('[-] {:>25}\t{:<10}\n'.format(
            'Prediction Asessment:',
            'Correct' if cat_to_name[class_cat] == names[0]
            else 'Incorrect'))
    else:
        print()

    print('{:<8}\t{:<20}\t{:<12}'.format(
        'Top Ks', 'Image Class', 'Probability'))
    for top_k, probability, name in zip(top_ks, probs_flat, names):
        print('{:<8}\t{:20}\t{:.3f}%'.format(top_k, name, probability*100))


# Call to main function to run the program
if __name__ == "__main__":
    main()
