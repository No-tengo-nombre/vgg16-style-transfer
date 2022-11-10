# Style transfer using VGG16
## EL4106 Computational Intelligence
### Cristóbal Allendes
### Tutor: Giovanni Castiglioni


## Description
This project aims to use the VGG16 network [1][2] to solve the problem of style transfer, by using the features extracted from the convolutional layers, creating an encoder-decoder architecture, and applying White-Coloring Transforms at different levels of detail.


## Install instructions
To install this package, clone the repo using
```
git clone --recursive https://github.com/No-tengo-nombre/el4106-style-transfer
```

Then, navigate to the cloned repository and install it using pip. This corresponds to the following code
```
cd ./el4106-style-transfer
python -m pip install .
```

**IMPORTANT**: At the time of writing this, I have not added a requirements file, so you have to manually install the dependencies (mainly PyTorch).

This will install the `vgg16autoencoder` package, which contains the model code and the functions to train, as well as the pretrained weights for the model.


## Use instructions
To use the package, you have one of two ways:

1. Use the CLI provided by the package. To do this, you simply have to run
```
python -m vgg16st [-T/-E] ...
```
with the appropiate arguments, and then the model will train (if you use `-T`) or evaluate the best model on a given image (if you use `-E`).

2. Create a script that imports the desired functionality and does whatever you need it to do :)


## References
[1] Karen Simonyan and Andrew Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition*. Available https://arxiv.org/abs/1409.1556

[2] PyTorch, *vgg16 - Torchvision main documentation*. Available https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html
