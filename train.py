#!/usr/bin/env python3
import torch
from torch import nn, optim
from time import time

import classifier_network
import train_utils


def main():
    input_arguments = train_utils.get_input_args()
    model = train_utils.get_model(input_arguments.arch)
    for param in model.parameters():
        param.requires_grad = False
        
    input_size = None
    if input_arguments.arch == 'vgg':
        input_size = 25088
    elif input_arguments.arch == 'alexnet':
        input_size = 9216
    elif input_arguments.arch == 'densenet':
        input_size = 1024        
        
    classifier = classifier_network.Network(input_size, 102, [input_arguments.hidden_units])
    model.classifier = classifier
    
    image_datasets = train_utils.get_image_datasets(input_arguments.data_dir)
    image_loader = train_utils.get_image_loader(image_datasets)
    
    device = train_utils.get_device(input_arguments.gpu)
    torch.device(device)  
    
    start_time = time()

    criterion = nn.NLLLoss()
    # Only training the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=input_arguments.learning_rate)
    model.to(device);
    
    train_loader = image_loader['train']
    valid_loader = image_loader['valid']
    saved, checkpoint_path = False, None
    
    model.train()
    running_loss = 0
    print_every = 5
    
    # print()
    print('\n===== MODEL TRAINING STARTED =====')
    print('\t[-] ARCHITECTURE: {}'.format(input_arguments.arch.upper()))    
    print('\t[-] LEARNING RATE: {}'.format(input_arguments.learning_rate))
    print('\t[-] HIDDEN UNITS: {}'.format(input_arguments.hidden_units))
    print('\t[-] EPOCHS: {}\n'.format(input_arguments.epochs))
    # print()
    
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
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            valid_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{input_arguments.epochs}.. "
                          f"Step {steps}/{len(train_loader)}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
                          "Validation accuracy: {:.3f}%".format(accuracy/len(valid_loader) * 100))
                    running_loss = 0
                    model.train()     

        end_time = time()
        total_time = end_time - start_time
        print("\n\t[-] Total Elapsed time:",
              str(int((total_time/3600)))+":"+str(int((total_time%3600)/60))+":"
              +str(int((total_time%3600)%60)) )            
        print('===== MODEL TRAINING COMPLETED =====\n')
        # print()
    except KeyboardInterrupt:
        choice = input('Do you want to save the trained model? [y/N]: ')
        if choice == 'y' or choice == 'Y':
                saved, checkpoint_path = train_utils.save_model(model.cpu(), optimizer, image_datasets, input_arguments.arch, input_arguments.save_dir, input_size)
    else:
        saved, checkpoint_path = train_utils.save_model(model.cpu(), optimizer, image_datasets, input_arguments.arch, input_arguments.save_dir, input_size)

    if saved:
        train_utils.test_saved_model(checkpoint_path, image_loader['test'], device, criterion)

        
# Call to main function to run the program
if __name__ == "__main__":
    main()
    