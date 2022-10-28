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


def get_input_args():
    """
    Gets the command-line arguments passed by the user. This function
    retrieves the 7 Command Line Arugment from the user running the program
    from a terminal window.

    Arguments: None

    Returns:
        parser.parse_args() (obj): An object which contains all the passed
        arguments
    """
    parser = argparse.ArgumentParser('This is the model training module')

    parser.add_argument('data_dir', type=str,
                        help='Path to the directory containing \
                        the training data set')
    parser.add_argument('--save_dir', type=str, default='saved_models/',
                        help='Path of the directory to save the \
                        checkpoints to')
    parser.add_argument('--arch', type=str, default='vgg',
                        help='Type of CNN model architecture to use',
                        choices=['vgg', 'alexnet', 'densenet'])
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='The learning rate to train the model on')
    parser.add_argument('--hidden_units', type=int, default=667,
                        help='Size of the hidden unit')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs to train the model')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU to train the model')

    return parser.parse_args()


def get_image_datasets(data_dir):
    """
    Prepares and returns the train, validation and test datasets from the
    data directory

    Arguments:
        data_dir (str): Path of the data directory containing the training,
                        validation and testing data

    Returns: A dictionary which contains the train, validation and test
    datasets
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # Gets the image transforms
    image_transforms = get_image_transforms()
    train_datasets = datasets.ImageFolder(train_dir,
                                          transform=image_transforms['train'])
    valid_datasets = datasets.ImageFolder(valid_dir,
                                          transform=image_transforms['valid'])
    test_datasets = datasets.ImageFolder(test_dir,
                                         transform=image_transforms['test'])

    return {
        'train': train_datasets,
        'valid': valid_datasets,
        'test': test_datasets
    }


def get_image_loader(image_datasets):
    """
    Prepares and returns the train, validation and test data loader from the
    datasets

    Arguments:
        image_datasets (dict): A dictionary containing the train, validation
                                and test datasets

    Returns: A dictionary which contains the train, validation and test data
            loader
    """
    train_loader = torch.utils.data.DataLoader(image_datasets['train'],
                                               batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(image_datasets['valid'],
                                               batch_size=64)
    test_loader = torch.utils.data.DataLoader(image_datasets['test'],
                                              batch_size=64)

    return {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }


def get_image_transforms():
    """
    Prepares and returns the train, validation and test data transforms

    Arguments: None

    Returns: A dictionary which contains the train, validation and test datas
    transforms
    """
    # Random rotation, cropping and flipping
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(
                                               [0.485, 0.456, 0.406],
                                               [0.229, 0.224, 0.225])])
    # Only crop images to 224x224
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              [0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])])
    # Only crop images to 224x224
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              [0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])])

    return {
        'train': train_transforms,
        'valid': valid_transforms,
        'test': test_transforms
    }


def get_model(arch):
    """
    Retreives the model architecture from PyTorch for training. It fetches
    one of the three models which are: VGG, ALEXNET and DENSENET

    Arguments:
        arch (str): The model architecture [vgg, alexnet, resnet]

    Returns:
        model (obj): The fetched model
    """
    print('[MODEL] Fetching {} model architecture ...'.format(arch.upper()))
    if (arch == 'vgg'):
        model = models.vgg11(pretrained=True)
    elif (arch == 'alexnet'):
        model = models.alexnet(pretrained=True)
    elif (arch == 'densenet'):
        model = models.densenet121(pretrained=True)

    print('[MODEL] Fetching complete')
    return model


def save_model(model, optimizer, image_datasets, arch, save_dir, input_size,
               learning_rate):
    """
    Saves the trained model to the specified directory

    Arguments:
        - model (obj): The trained model
        - optimizer (obj): The optimzer user for training
        - image_datasets (dict): The train, validation and test datasets
        - arch (str): The model architecture name
        - save_dir (str): Path to save the trained model or checkpoint
        - input_size (int): The input size of the classifir network

    Returns:
        On succuess:
            True, checkpoint_path (tuple): Indicating success
                - True (bool): indicating the file is saved
                - checkpoint_path (str): The path to the saved model
        On failiure:
            False, None (tuple): Indicating failiure
                - False (bool): Indicating the file is not sad
                - None: Indicating the path to sad model is None
    """
    print('[SAVE MODEL] Attempting to save model')

    checkpoint_dir = save_dir
    checkpoint_name = arch + '_checkpoint.pth'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    # Creates the directory first before continuing
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.train()
    checkpoint = {
        'input_size': input_size,
        'output_size': 102,
        'hidden_layers': [each.out_features for each in
                          model.classifier.hidden_layers],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'image_datasets': image_datasets,
        'arch': arch,
        'learning_rate': learning_rate
    }
    try:
        torch.save(checkpoint, checkpoint_path)
    except Exception:
        print('[SAVE MODEL] ERROR: Unable to save model, cleaning up ...')
        # Clean up if saving fails
        delete_model(checkpoint_path)
    else:
        print('[SAVE MODEL] Save Complete')
        return True, checkpoint_path

    return False, None


def delete_model(checkpoint_path):
    """
    Deletes a checkpoint or saved model specified by the path

    Arguments:
        checkpoint_path (str): Path of the checkpoint to be deleted

    Returns: None
    """
    print('\t[DELETE MODEL] Attempting to delete model')
    try:
        os.remove(checkpoint_path)
    except Exception:
        print('\t[DELETE MODEL] ERROR: Unable to delete model')
    else:
        print('\t[DELETE MODEL] Model deleted')


def load_checkpoint(checkpoint_path):
    """
    Loads a saved checkpoint or model specified by the path

    Arguments:
        checkpoint_path (str): Path to the saved checkpoint

    Returns:
        model, optimizer (tuple): The model and optimizer
            - model (obj): The model that is loaded from the checkpoint
            - optimizer (obj): The optimizer used for training
    """
    # I took the below if else statement from the accepted answer of this
    # stackoverflow question:
    # https://stackoverflow.com/questions/55759311/runtimeerror-cuda-
    # runtime-error-35-cuda-driver-version-is-insufficient-for
    if torch.cuda.is_available():
        # map_location = lambda storage, loc: storage.cuda()
        def map_location(storage, loc): return storage.cuda()
    else:
        map_location = 'cpu'

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    # Get the model architecture name since it is already saved
    arch = checkpoint['arch']

    model = None
    if arch == 'vgg':
        model = models.vgg11()
    elif arch == 'alexnet':
        model = models.alexnet()
    elif arch == 'densenet':
        model = models.densenet121()
    # Get the image datasets
    image_datasets = checkpoint['image_datasets']
    # Build the classifier netwaork from the saved arguments
    classifier = classifier_network.Network(checkpoint['input_size'],
                                            checkpoint['output_size'],
                                            checkpoint['hidden_layers'])

    model.classifier = classifier
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.eval()
    optimizer = optim.Adam(model.classifier.parameters(),
                           lr=checkpoint['learning_rate'])
    # Load the saved state_dict of the optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer


def test_saved_model(checkpoint_path, test_loader, device, criterion):
    """
    Tests a saved checkpoint or model with the test datasets

    Arguments:
        - checkpoint_path (str): Path of the checkpoint to be tested
        - test_loader (obj): The test data loader
        - device (str): The default device to run the test
        - criterion (obj): The criterion to use for testing

    Returns: None
    """
    print('\n===== SAVED MODEL TESTING STARTED =====')
    # Load the checkpoint that is to be tested
    model, optimizer = load_checkpoint(checkpoint_path)

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
    print("\t[-] Test Accuracy: {:.3f}%".format(
        accuracy/len(test_loader) * 100))
    print('\n===== SAVED MODEL TESTING COMPLETED =====')


def get_device(use_gpu):
    """
    Gets the default device to run the operation, which is either
    GPU (CUDA) or CPU

    Arguments:
        use_gpu (bool): Indicates whether to use the gpu o not

    Returns:
        device (str): The device to use
    """
    device = "cpu"
    if use_gpu:
        # Checks if GPU is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print('[DEVICE] ERROR: GPU is not available continuing with CPU')
    if device == 'cuda':
        print('[DEVICE] Using GPU to test model')
    else:
        print('[DEVICE] Using CPU to test model')

    return device
