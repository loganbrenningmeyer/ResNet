import os
import lightning as L
import matplotlib.pyplot as plt

from torch import optim, nn, utils, Tensor
from torch.utils.data import DataLoader
import lightning as L

import torchvision.models as models
from torchvision.datasets import ImageFolder

import albumentations as album
from albumentations.pytorch import ToTensorV2

import numpy as np
import cv2
import json
import ast

class ResNet(L.LightningModule):
    def __init__(self, params):
        super().__init__()

        self.batch_size = params["batch_size"]
        self.num_classes = params["num_classes"]
        self.input_channels = params["sample_shape"][0]

        # Set to ResNet34 with pretrained weights
        weights = models.ResNet34_Weights.DEFAULT
        self.arch = models.resnet34(weights=weights)

        self.arch.fc = nn.Linear(512, self.num_classes, bias=True)

        self.objective = nn.CrossEntropyLoss()
        self.learning_rate = params["learning_rate"]

    def configure_optimizers(self):
        return optim.Adam(self.arch.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        samples, labels = batch

        # Gather predictions
        preds = self.arch(samples)

        # Calculate objective loss
        loss = self.objective(preds, labels)

        # Log loss
        print(f"Batch {batch_idx} loss: {loss}")
        self.log('train_loss', loss, batch_size=self.batch_size, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        samples, labels = batch

        # Gather preditions
        preds = self.arch(samples)

        # Calculate objective loss
        loss = self.objective(preds, labels)

        # Log loss
        self.log('val_loss', loss, batch_size=self.batch_size, on_step=True, on_epoch=True)

    def forward(self, samples):
        return self.arch(samples)

class Transforms:
    def __init__(self):
        self.transforms = album.Compose([album.RandomResizedCrop(size=[224, 224]),
                              album.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                              ToTensorV2()])
        
    def __call__(self, image):
        image = np.array(image) 
        aug_image = self.transforms(image=image)
        return aug_image['image']


def main():
    #augment_data("imagenet-mini/train/n01440764")

    with open("class-list.txt", 'r') as f:
        data = f.read()

    class_names = ast.literal_eval(data)

    transform = Transforms()

    # Read datasets using ImageFolder
    train_dataset = ImageFolder("imagenet-mini/train", transform=transform)
    val_dataset = ImageFolder("imagenet-mini/val", transform=transform)

    # print(f"Num Classes: {len(dataset.classes)}")
    # #for idx, class_name in enumerate(dataset.classes):
    #     #print(f"{idx}: {class_name}")
    # print(f"Image: {dataset[0][0].shape}")
    # print(f"Target: {dataset[0][1]}\n")

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)

    # Initialize model
    params = {"batch_size": 32,
              "learning_rate": 1e-3,
              "num_classes": len(train_dataset.classes),
              "sample_shape": train_dataset[0][0].shape}
    model = ResNet(params=params)
    
    # ---- Training Model -----
    trainer = L.Trainer()
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# ResNet50

if __name__ == "__main__":
    main()