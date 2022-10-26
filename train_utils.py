"""
This is the "train_utils" file.

The train_utils file contains functions that are needed for the model training.
"""

import torch
from torch import optim
from torchvision import datasets, transforms, models
import os
import argparse

import classifier_network


"""
This is the "get_input_args" file.

The get_input_args file parses and gets the command-line input arguments.
"""

def get_input_args():
    """
    Gets the command-line arguments passed by the user.
    """
    parser = argparse.ArgumentParser('This is the model training module') 
    
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

def get_image_datasets(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    image_transforms = get_image_transforms()
    train_datasets = datasets.ImageFolder(train_dir, transform=image_transforms['train'])
    valid_datasets = datasets.ImageFolder(valid_dir, transform=image_transforms['valid'])
    test_datasets = datasets.ImageFolder(test_dir, transform=image_transforms['test'])
    
    return {
        'train': train_datasets,
        'valid': valid_datasets,
        'test': test_datasets
    }
    
    
def get_image_loader(image_datasets):
    train_loader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64)
    test_loader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=64)

    return {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }

    
def get_image_transforms():
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    return {
        'train': train_transforms,
        'valid': valid_transforms,
        'test': test_transforms
    }
 

def get_model(arch):
    """
    Gets the command-line arguments passed by the user.
    """
    print('[MODEL] Fetching {} model architecture ...'.format(arch.upper()))
    if (arch == 'vgg'):
        model = models.vgg11(pretrained=True)
    elif (arch == 'alexnet'):
        model = models.alexnet(pretrained=True)
    elif (arch == 'resnet'):
        model = models.resnet18(pretrained=True)
        
    print('[MODEL] Fetching complete')
    return model


def save_model(model, optimizer, image_datasets, arch, save_dir, input_size):
    print('[SAVE MODEL] Attempting to save model')
    
    checkpoint_dir = save_dir
    checkpoint_name = arch + '_checkpoint.pth'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.train()
    checkpoint = {
        'input_size': input_size,
        'output_size': 102,
        'hidden_layers': [each.out_features for each in model.classifier.hidden_layers],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'image_datasets': image_datasets,
        'arch': arch,
    }
    try:
        torch.save(checkpoint, checkpoint_path)
    except Exception:
        print('[SAVE MODEL] ERROR: Unable to save model, cleaning up ...')
        delete_model(checkpoint_path)
    else:
        print('[SAVE MODEL] Save Complete')
        return True, checkpoint_path
    
    return False, None
        
        
def delete_model(checkpoint_path):
    print('\t[DELETE MODEL] Attempting to delete model')
    try:
        os.remove(checkpoint_path)
    except Exception:
        print('\t[DELETE MODEL] ERROR: Unable to delete model')
    else:
        print('\t[DELETE MODEL] Model deleted')
    

def load_checkpoint(checkpoint_path, arch):
    """I took the below if else statement from the accepted answer of this stackoverflow question:
    https://stackoverflow.com/questions/55759311/runtimeerror-cuda-runtime-error-35-cuda-driver-version-is-insufficient-for
    """
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model = None
    if arch == 'vgg':
        model = models.vgg11()
    elif arch == 'alexnet':
        model = models.alexnet()
    elif arch == 'resnet':
        model = models.resnet18()
        
    image_datasets = checkpoint['image_datasets']
    
    classifier = classifier_network.Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.eval()

    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer


def test_saved_model(checkpoint_path, test_loader, device, criterion, arch):
    print('\n===== SAVED MODEL TESTING STARTED =====')
    model, optimizer = load_checkpoint(checkpoint_path, arch)
    
    model.to(device)
    test_loss = 0
    accuracy = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print('\nResults:')
    print("\t[-] Test Loss: {:.3f}".format(test_loss/len(test_loader)))
    print("\t[-] Test Accuracy: {:.3f}%".format(accuracy/len(test_loader) * 100))
    print('\n===== SAVED MODEL TESTING COMPLETED =====')
