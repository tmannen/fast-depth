# coding: utf-8
import models
model = models.MobileNetSkipAdd(10)
model
from PIL import Image
test = Image.open("../data/bcs_floor6_play_only_formatted/images/0001.png")
test
import numpy as np
np.asarray(test)
nptest = np.asarray(test)
nptest.shape
import torch
import dataloaders.transforms as transforms
import dataloaders.sun3d as sun3d
dataloader = sun3d.Sun3DDataset("../data/bcs_floor6_play_only_formatted/images/", "val")
gg = dataloader.val_transform(nptest, nptest)
img = gg[0]
img_t = np.transpose(img, (2, 0, 1)).shape
img_t = np.transpose(img, (2, 0, 1))
np.expand_dims(img_t, 0)
np.expand_dims(img_t, 0).shape
final_maybe = np.expand_dims(img_t, 0)
model(torch.from_numpy(final_maybe).float())
model(torch.from_numpy(final_maybe).float()).shape
