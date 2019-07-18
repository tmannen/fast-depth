%run torch_test_session.py
outputs = []
def hook(module, input, output):
    outputs.append(output)
model.conv13.register_forward_hook(hook)
model(torch.from_numpy(final_maybe).float())
outputs
outputs[0]
outputs[0].shape
imed = outputs[0]
np.random.randn((10, 1024, 7, 7))
np.random.randn((10, 1024, 7, 7))
test = np.random.randn(10, 1024, 7, 7)
from utils_mvs_temporal import *
%autoreload 2
%load_ext autoreload
%autoreload 2
from utils_mvs_temporal import *
K, poses = get_camera_values("../data/bcs_floor6_play_only_formatted/")
reload(utils_mvs_temporal.py)
%paste
K, poses = get_camera_values("../data/bcs_floor6_play_only_formatted/")
import os
K, poses = get_camera_values("../data/bcs_floor6_play_only_formatted/")
poses
poses.shape
poses[:10]
testposes = poses[:10]
testposes.shape
distances = pose_distance_measure(testposes)
distances.shape
kernel_out = matern_kernel(distances)
kernel_out.shape
kernel_out
kernel_out.dot(test)
test.T
test.T.shape
kernel_out.dot(test.T)
test.squeeze()
test.squeeze().shape
test.shape
testre = test.reshape((10, 1024*7*7))
testre.shape
kernel_out.dot(test)
kernel_out.dot(testre)
kernel_out.dot(testre).shape
torch.sqrt(10)
torch.sqrt(torch.Tensor(19))
history
