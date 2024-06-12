from typing import Any
import albumentations as album
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
Noise augmentations to try

ISO noise
Multiplicative noise
Gaussian noise
Randomtone curve
'''
class Transform:
    def __init__(self, params):

        self.transforms = [album.Resize(224, 224)]

        for param in params:
            if param == 'iso':
                self.transforms += [album.ISONoise(p=1)]
            elif param == 'mult':
                self.transforms += [album.MultiplicativeNoise(p=1)]
            elif param == 'gauss':
                self.transforms += [album.GaussNoise(p=1)]
            elif param == 'tone':
                self.transforms += [album.RandomToneCurve(p=1)]

        self.transforms += [album.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        
        self.transforms = album.Compose(self.transforms)

    def __call__(self, image):
        return self.transforms(image=image)['image']

if __name__ == '__main__':
    image = cv2.imread('image02.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create Transform object
    iso_transform = Transform(['iso'])
    mult_transform = Transform(['mult'])
    gauss_transform = Transform(['gauss'])
    tone_transform = Transform(['tone'])

    iso_image = iso_transform(image=image)
    iso_image = (iso_image + 1) / 2

    mult_image = mult_transform(image=image)
    mult_image = (mult_image + 1) / 2

    gauss_image = gauss_transform(image=image)
    gauss_image = (gauss_image + 1) / 2

    tone_image = tone_transform(image=image)
    tone_image = (tone_image + 1) / 2

    transform = Transform([])
    image = transform(image=image)
    image = (image + 1) / 2

    # -- Plot image/augmented image
    fig, axs = plt.subplots(1, 5, figsize=(6, 12))

    axs[0].imshow(image)
    axs[1].imshow(np.array(iso_image))
    axs[2].imshow(np.array(mult_image))
    axs[3].imshow(np.array(gauss_image))
    axs[4].imshow(np.array(tone_image))

    plt.show()
