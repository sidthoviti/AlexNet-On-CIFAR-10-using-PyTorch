import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
import os
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import *
from alexnet_models import *

# Set up preprocessing of CIFAR-10 images to 3x224x224 with normalization
preprocess = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                    (0.2023, 0.1994, 0.2010))])

# Download CIFAR-10 and split into training, validation, and test sets

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=preprocess)

# Split the training set into training and validation sets randomly.
# CIFAR-10 train contains 50,000 examples, so let's split 80%-20%.

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [40000, 10000])

# Download the test set.
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=preprocess)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16,
                                               shuffle=True, num_workers=2)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16,
                                             shuffle=False, num_workers=2)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16,
                                              shuffle=False, num_workers=2)

#Choose the most avaliable GPU
device = torch.device(get_freer_gpu()) if torch.cuda.is_available() else torch.device("cpu")
print("GPU device: ", device)

module = AlexNetModule(10)
module_LRN = AlexNetModuleLRN(10)

models = [module, module_LRN]
model_names = ['Alexnet without LRN', 'Alexnet with LRN']

criterion = nn.CrossEntropyLoss().to(device)

optimizers = []
for model in models:
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizers.append(optimizer)
    print(optimizers)

for i, model in enumerate(models):
    print(f'{str(model_names[i])} has {count_parameters(model):,} trainable parameters')

dataloaders = { 'train': train_dataloader, 'val': val_dataloader }

#Training
for i, model in enumerate(models):
    print(f"Training: {model_names[i]}")
    best_model, val_acc_history, loss_acc_history = train_model(model, dataloaders, device, criterion, optimizers[i], 10, model_names[i])
    plot_data(val_acc_history, loss_acc_history, i)
    print('='*50)

#Testing
for i, model in enumerate(models):
    print(f"Model: {model_names[i]}")
    model.load_state_dict(torch.load(f'{model_names[i]}.pth'))
    test_loss, test_acc, test_pred_label, test_true_label  = evaluate(model, device, test_dataloader, criterion)
    print(f'Test Loss:{test_loss:.3f} | Test acc: {test_acc:.2f}%')