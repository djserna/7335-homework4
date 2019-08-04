
from IPython.core.interactiveshell import InteractiveShell
import seaborn as sns
# PyTorch
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Data science tools
import numpy as np
import pandas as pd
import os

# Image manipulations
from PIL import Image
# Useful for examining network
from torchsummary import summary
# Timing utility
from timeit import default_timer as timer

# Visualizations
import matplotlib.pyplot as plt
#%matplotlib inline
plt.rcParams['font.size'] = 14

# Printing out all outputs
InteractiveShell.ast_node_interactivity = 'all'

useGPUForTraining = False
multipleGPUsAvailable = False
selectedModel = None
numberOfInputs = None

def setParameters():
    print("Initialize program parameters.")

def eda():
    print("Perform exploratory data analysis.")    

def dataAugmentation():
    print("Image transformation.")
    
def loadPreTrainedModel(modelName):
    if (modelName == "vgg16"):
        selectedModel = models.vgg16(pretrained=True)
        numberOfInputs = selectedModel.classifier[6].in_features
    elif (modelName == "vgg19"):
        selectedModel = models.vgg19(pretrained=True)
        numberOfInputs = selectedModel.classifier[6].in_features
    elif (modelName == "resnet50"):
        selectedModel = models.resnet50(pretrained=True)
        numberOfInputs = selectedModel.fc.in_features
    elif (modelName == "squeezenet1_0"):
        selectedModel = models.squeezenet1_0(pretrained=True)
         #not sure this is accurate but closest I could find for in_features on this model
        numberOfInputs = len(selectedModel.features)
    elif (modelName == "squeezenet1_1"):
        selectedModel = models.squeezenet1_1(pretrained=True)
        #not sure this is accurate but closest I could find for in_features on this model
        numberOfInputs = len(selectedModel.features)
    elif (modelName == "Inception3"):
        selectedModel = models.inception_v3(pretrained=True)
        numberOfInputs = selectedModel.fc.in_features
    
    for parameter in selectedModel.parameters():
        parameter.requires_grad = False #not sure why we do this.
        
    
    if useGPUForTraining == True:
        selectedModel = selectedModel.to('cuda')
    
    if multipleGPUsAvailable == True:
        selectedModel = nn.DataParallel(selectedModel)
    
    print(selectedModel)
    print(numberOfInputs)
    
def adjustClassifiers():
    print("use our own custom classifiers.")
    
def displayResults():
    print("display analysis.")
    
setParameters();
eda();
dataAugmentation();
loadPreTrainedModel("vgg19");
adjustClassifiers();
displayResults();