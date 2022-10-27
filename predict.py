#!/usr/bin/env python3
import torch
from pathlib import Path 

import classifier_network
import predict_utils
from train_utils import load_checkpoint, get_device


def main():
    input_arguments = predict_utils.get_input_args()
    model, optimizer = load_checkpoint(input_arguments.checkpoint)
    cat_to_name = predict_utils.load_cat_to_name(input_arguments.category_names)

    device = get_device(input_arguments.gpu)
    torch.device(device)
    
    # Assuming image path as such: 'flowers/train/14/image_06058.jpg'
    class_integer = None
    img_path = input_arguments.input
    try:
        class_integer = img_path.split('/')[2]
    except Exception:
        class_integer = Path(input_arguments.input).name

    probs, classes = predict_utils.predict(input_arguments.input, model, input_arguments.top_k, device) 
    
    probs_flat = probs.cpu().numpy().flatten()
    names = [cat_to_name[cls] for cls in classes]
    top_ks = list(range(1, input_arguments.top_k+1))
    
    print('\t\t===== PREDICTION RESULT =====\n')
    
    if class_integer is not None and class_integer in cat_to_name:
        print('[-] {:>25}\t{:<20}'.format('Image Class:', cat_to_name[class_integer]))
    print('[-] {:>25}\t{:<20}'.format('Predicted Image Class:', names[0]))
    print('[-] {:>25}\t{:.3f}%'.format('Prediction Accuracy:', probs_flat[0]*100))
    if class_integer is not None and class_integer in cat_to_name:
        print('[-] {:>25}\t{:<10}\n'.format('Prediction Asessment:', 'Correct' if cat_to_name[class_integer] == names[0] else 'Incorrect'))
    else:
        print()
    
    print('{:<8}\t{:<20}\t{:<12}'.format('Top Ks', 'Image Class', 'Probability'))
    for top_k, probability, name in zip(top_ks, probs_flat, names):
        print('{:<8}\t{:20}\t{:.3f}%'.format(top_k, name, probability*100))
    
    
# Call to main function to run the program
if __name__ == "__main__":
    main()
    