#!/usr/bin/env python3
import torch

import classifier_network
import predict_utils
from train_utils import load_checkpoint
from pathlib import Path 


def main():
    input_arguments = predict_utils.get_input_args()
    arch = Path(input_arguments.checkpoint).name.split('_')[0]
    model, optimizer = load_checkpoint(input_arguments.checkpoint, arch)
    cat_to_name = predict_utils.load_cat_to_name(input_arguments.category_names)

    device = "cpu"
    if input_arguments.gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"  
        if device == "cpu":
            print('[DEVICE] ERROR: GPU is not available continueing with CPU')
    if device == 'cuda':
        print('[DEVICE] Using GPU to calculate Prediction')  
    else:
        print('[DEVICE] Using CPU to calculate Prediction')  
    torch.device(device)
    
    # Assuming image path as such: 'flowers/train/12/image_04021.jpg'
    class_integer = None
    img_path = input_arguments.input
    try:
        class_integer = img_path.split('/')[2]
    except Exception:
        pass

    probs, classes = predict_utils.predict(input_arguments.input, model, input_arguments.top_k, device) 
    
    probs_flat = probs.cpu().numpy().flatten()
    names = [cat_to_name[cls] for cls in classes]
    top_ks = list(range(1, input_arguments.top_k+1))
    
    print('\t\t===== PREDICTION RESULT =====\n')
    
    if class_integer is not None and class_integer in cat_to_name:
        print('[-] {:>25}\t{:<20}'.format('Image Label:', cat_to_name[class_integer]))
    print('[-] {:>25}\t{:<20}'.format('Predicted Image Name:', names[0]))
    print('[-] {:>25}\t{:.3f}%'.format('Prediction Accuracy:', probs_flat[0]*100))
    print('[-] {:>25}\t{:<10}\n'.format('Prediction Asessment:', 'Correct' if cat_to_name[class_integer] == names[0] else 'Incorrect'))
    
    print('{:<8}\t{:<20}\t{:<12}'.format('Top Ks', 'Flower Name', 'Probability'))
    for top_k, probability, name in zip(top_ks, probs_flat, names):
        print('{:<8}\t{:20}\t{:.3f}%'.format(top_k, name, probability*100))
    
    
# Call to main function to run the program
if __name__ == "__main__":
    main()
    