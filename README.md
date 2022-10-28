# Flower Image Classification

This project is AI Programming with Python Project from Udacity. The project trains and predicts 102 types of flowers. As a part of the AI Programming with Pytho Nanodegree, it serves as a final project by learning and applying skills to implement AI using Python. This project will train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. It will be using [this dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories.

All code follows [PEP8 style guidelines](https://www.python.org/dev/peps/pep-0008/).

## Project Structure

```bash
├── assets  # Assets used for the demonstration files
│   ├── Flowers.png
│   └── inference_example.png
├── cat_to_name.json  # The category to name mapping file
├── classifier_network.py  # The classfier network class
├── flowers -> ../../../data/flowers  # The data used for trainiang
├── Image Classifier Project.html  # Demonstration of the project
├── Image Classifier Project.ipynb  # Demonstration Notebook
├── LICENSE
├── predict.py  # The prediction module
├── predict_utils.py  # Utilities needed by the prediction module
├── README.md
├── saved_models
│   └── vgg_checkpoint.pth  # A sample saved trained VGG model
├── test.py  # The testing module
├── train.py  # The training module
├── train_utils.py  # Utilities needed by the training module
└── workspace-utils.py
```
  
## CLI Modules

The project includes three CLI modules which are `train.py`, `predict.py` and `test.py`. The first file, `train.py`, will train a new network on a dataset and save the model as a checkpoint. The second file, `predict.py`, uses a trained network to predict the class for an input image. The third file, `test.py`, tests the saved checkpoint against the test data sets.

1. Training module
Trains a new network on a data set. It also prints out training loss, validation loss, and validation accuracy as the network trains.

### Basic usage
```
python train.py data_dir [--save_dir SAVE_DIR]
						 [--arch {vgg,alexnet,densenet}] 
                         [--learning_rate LEARNING_RATE]
                         [--hidden_units HIDDEN_UNITS]
                         [--epochs EPOCHS]
                         [--gpu]
```
Or:
```bash
./train.py data_dir [--save_dir SAVE_DIR]
					[--arch {vgg,alexnet,densenet}] 
                    [--learning_rate LEARNING_RATE]
                    [--hidden_units HIDDEN_UNITS]
                    [--epochs EPOCHS]
                    [--gpu]
```

### Options
- `data_dir` -> `str` `positional` Path/location of the training data
- `save_dir` -> `str` `optional` Location to save the trained the model to
  - default: `saved_models`
- `arch` -> `str` `optional` The model architecture to use for training
  - default: `vgg`
- `learning_rate` -> `float` `optional` The learning rate to train the model
  - default: `0.001`
- `hidden_units` -> `int` `optional` The number of hidden units in the layer
  - default: `667`
- `epochs` -> `int` `optional` The total numbaro of epochs to train the model
  - default: 1
- `gpu` -> `bool` `optional` Specifies whether to use GPU for training

2. Predict module
Predict flower name from an image with predict.py along with the probability of that name. That is, user passes in a single image /path/to/image and return the flower name and class probability.

### Basic usage
```
python predict.py input checkpoint [--top_k TOP_K]
								   [--category_names CATEGORY_NAMES]
                                   [--gpu]
```
Or:
```bash
./predict.py input checkpoint [--top_k TOP_K] 
					 		  [--category_names CATEGORY_NAMES]
                              [--gpu]
```

### Options
- `input` -> `string` `positional` Path/location of the image to be predicted
- `checkpoint` -> `string` `positional` Path/Location of the saved model
- `top_k` -> `integer` `optional` The number of top most likely predicted classes
  - default: `5`
- `category_names` -> `string` `optional` Path/Location of the mapping of categories to names
  - default: `cat_to_name`
- `gpu` -> `bool` `optional` Specifies whether to use GPU for training

3. Test module
Tests a trained model or checkpoint that is passed by the user.

### Basic usage
```
python test.py checkpoint data_dir [--gpu]
```
Or:
```bash
./test.py checkpoint data_dir [--gpu]
```

### Options
- `data_dir` -> `str` `positional` Path/location of the training data
- `checkpoint` -> `string` `positional` Path/Location of the saved model
- `gpu` -> `bool` `optional` Specifies whether to use GPU for training


## Getting started

### Install Dependencies

1. **Python 3.6** - Follow instructions to install the latest version of python for your platform in the [python docs](https://docs.python.org/3/using/unix.html#getting-and-installing-the-latest-version-of-python)

2. **Virtual Environment** - We recommend working within a virtual environment whenever using Python for projects. This keeps your dependencies for each project separate and organized. Instructions for setting up a virual environment for your platform can be found in the [python docs](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)

#### Key Pip Dependencies

- [PyTorch](https://pytorch.org/) is a machine learning framework based on the Torch library, used for applications such as computer vision and natural language processing. This project uses PyTorch version `0.4`.

## Details

The project is broken down into multiple steps:
1. Loading and preprocessing the image dataset
2. Training the image classifier on the dataset and saving it
3. Using the trained classifier to predict image content
4. Optionally testing the saved model

To build and train the classifier, the project uses one of the pretrained models from torchvision.models to get the image features and builds and trains a new feed-forward classifier using those features. When training it updates only the weights of the feed-forward network.

The training perfforms these actions:
- Loads a pre-trained network
- Defines a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
- Trains the classifier layers using backpropagation using the pre-trained network to get the features
- Tracks the loss and accuracy on the validation set to determine the best hyperparameters

### Normalizing Images

Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. To do that the project simply divides the color channels b 255.
The network also expects the images to be normalized in a specific way. For the means, it's [0.485, 0.456, 0.406] and for the standard deviations [0.229, 0.224, 0.225]. Therefore the project subtract the means from each color channel, then divides by the standard deviation.
