import numpy as np

def pose_distance_measure(poses):
    """
    Page 4 in the paper: https://arxiv.org/pdf/1904.06397.pdf

    Get all the poses for a sequence and calculate them at the same time?

    """

    distances = np.zeros((poses.shape[0], poses.shape[0]))

    for idx in range(poses.shape[0]):
        # Can the for loop be vectorised away?
        t_j = poses[idx, :3, -1]
        ti_minus_tj_norm = np.linalg.norm(poses[:, :3, -1] - t_j, ord=2, axis=1)**2

        r_j = poses[idx, :3, :3]
        r_is = poses[:, :3, :3]
        tr_in = np.transpose(r_is, axes=(0,2,1)).dot(r_j)
        tr_calc = (2./3)*np.trace(np.eye(3) - tr_in, axis1=1, axis2=2)
        tr_calc[tr_calc<0] = 0 # there's a rounding error that makes distances from itself to be < 0, so make them 0
        result = np.sqrt(ti_minus_tj_norm + tr_calc) # Has the result of one with all others, so shape (100, 1)?
        distances[idx, :] = result

    return distances

def matern_kernel(distances):
    # TODO: try the mobilenet encoder on a sequence and apply this kernel to the encoded image?
    # These should be trainable right? Values from paper for now
    gamma = 13.82
    l = 1.098
    return gamma**2 * (1 + (np.sqrt(3)*distances/l)) * np.exp(-(np.sqrt(3)*distances/l))

def get_camera_values(path):
    K = np.loadtxt(path)
    poses = np.loadtxt(path)
    poses.shape = (poses.shape[0], 4, 4)
    return K, poses
