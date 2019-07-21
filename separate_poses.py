"""
Take the poses in poses.txt and make a folder called 'poses', with the ID of the image and the pose it has.
Makes it more intuitive to load with dataloader?
"""

import os
import numpy as np

def separate(rootpath):
    posepath = os.path.join(rootpath, "poses")
    if not os.path.exists(posepath):
        os.mkdir(posepath)

    imagenames = os.listdir(os.path.join(rootpath, "images"))
    imagenames = [name.split(".")[0] for name in imagenames]

    poses = np.loadtxt(os.path.join(path, "poses.txt"))
    # poses.shape = (poses.shape[0], 4, 4)

    for idx, name in enumerate(imagenames):
        np.save(os.path.join(posepath, name+".npy"), poses[idx])


# TODO: use args so this is inputtable
path = "../data/bcs_floor6_play_only_formatted/"
separate(path)
