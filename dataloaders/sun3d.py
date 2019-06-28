import numpy as np
from PIL import Image
import os
import os.path
import dataloaders.transforms as transforms
from dataloaders.dataloader import MyDataloader

iheight, iwidth = 480, 640 # raw image size

def rgb_depth_loader(path):
    rgb = np.array(Image.open(path))
    # Just expect depth to be in the same root as images folder and use image path to get depth?
    depth = np.load(path.replace("images", "depth").replace("png", "npy"))
    return rgb, depth

class Sun3DDataset(MyDataloader):
    def __init__(self, root, split, modality='rgb'):
        self.split = split
        super(Sun3DDataset, self).__init__(root, split, modality, loader=rgb_depth_loader)
        self.output_size = (224, 224)

    def make_dataset(self, dir, class_to_idx):
        # Expects 'blaa/blaa/images', with 'depth' in same folder as 'images' later
        images = []
        dir = os.path.expanduser(dir)

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, 0) # 0 as dummy class
                    images.append(item)

        return images

    # Copied from NYUDataset - correct?
    def train_transform(self, rgb, depth):
        s = np.random.uniform(1.0, 1.5) # random scaling
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Resize(250.0 / iheight), # this is for computational efficiency, since rotation can be slow
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop((228, 304)),
            transforms.HorizontalFlip(do_flip),
            transforms.Resize(self.output_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np) # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose([
            transforms.Resize(250.0 / iheight),
            transforms.CenterCrop((228, 304)),
            transforms.Resize(self.output_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np