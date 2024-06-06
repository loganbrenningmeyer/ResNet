import lightning as L
from torch import optim, nn, utils, Tensor

'''
Data:

224 x 224 px image (cropped)
Per-pixel mean subtracted
'''

# ResNet34
class ResNet34(L.LightningModule):
    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):


# ResNet50
