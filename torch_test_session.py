# coding: utf-8
import models
model = models.MobileNetSkipAdd(10)
model
from PIL import Image
test = Image.open("../data/bcs_floor6_play_only_formatted/images/0001.png")
test
test.array()
test.asarray()
import numpy as np
np.asarray(test)
nptest = np.asarray(test)
nptest.shape
model(nptest)
import torch
torch.fromnumpy(nptest)
torch.from_numpy(nptest)
totest = torch.from_numpy(nptest)
model(totest)
model[0]
model.layers[0]
totest.shape
import dataloaders.transforms as transforms
transform = transforms.Compose([
            transforms.Resize(250.0 / iheight), # this is for computational efficiency, since rotation can be slow
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop((228, 304)),
            transforms.HorizontalFlip(do_flip),
            transforms.Resize(self.output_size),
        ])
import dataloaders.sun3d as sun3d
dataloader = sun3d.Sun3DDataset("../data/bcs_floor6_play_only_formatted/images/", "val")
dataloder.val_transform(totest, totest)
dataloader.val_transform(totest, totest)
dataloader.val_transform(nptest, nptest)
gg = dataloader.val_transform(nptest, nptest)
gg.shape
gg[0].shape
model(gg)
model(gg[0])
model(torch.from_numpy(gg[0]))
img = gg[0]
np.transpose(img, (2, 0, 1))
np.transpose(img, (2, 0, 1)).shape
img_t = np.transpose(img, (2, 0, 1)).shape
img_t.unsqueeze()
img_t = np.transpose(img, (2, 0, 1))
np.expand_dims(img_t, 0)
np.expand_dims(img_t, 0).shape
final_maybe = np.expand_dims(img_t, 0)
model(torch.from_numpy(final_maybe))
model(torch.from_numpy(final_maybe).type(torch.DoubleTensor))
tens = torch.from_numpy(final_maybe).type(torch.DoubleTensor)
tens
tens = torch.from_numpy(final_maybe).double()
tens
tens = torch.from_numpy(final_maybe).double()
model(torch.from_numpy(final_maybe).double())
model(torch.from_numpy(final_maybe).float())
model(torch.from_numpy(final_maybe).float()).shape
get_ipython().run_line_magic('history', '')
get_ipython().run_line_magic('history', '> ipythonhistory_16072019')
