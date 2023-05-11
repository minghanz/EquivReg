import torch
# import torch.distributions as dist
import logging

import numpy as np

def solve_R(f1, f2):
    """f1 and f2: (b*)m*3
    only work for batch_size=1
    """
    S = torch.matmul(f1.transpose(-1, -2), f2)  # 3*3
    U, sigma, V = torch.svd(S)
    R = torch.matmul(V, U.transpose(-1, -2))
    det = torch.det(R)
    # logging.info(R)
    diag_1 = torch.tensor([1, 1, 0], device=R.device, dtype=R.dtype)
    diag_2 = torch.tensor([0, 0, 1], device=R.device, dtype=R.dtype)
    det_mat = torch.diag(diag_1 + diag_2 * det)

    # det_mat = torch.eye(3, device=R.device, dtype=R.dtype)
    # det_mat[2, 2] = det

    det_mat = det_mat.unsqueeze(0)
    # logging.info(det_mat)
    R = torch.matmul(V, torch.matmul(det_mat, U.transpose(-1, -2)))
    logging.debug(f'det(R)={det}')
    # logging.info(V.shape)
    
    return R

def angle_of_R(R):
    # logging.info("R_diff", R_diff)
    cos_angle_diff = (torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)  - 1) / 2
    # logging.info("cos_angle_diff", cos_angle_diff)
    cos_angle_diff = torch.clamp(cos_angle_diff, -1, 1)
    angle_diff = torch.acos(cos_angle_diff)
    angle_diff = angle_diff / np.pi * 180
    return angle_diff

def angle_diff_func(R1, R2):
    R_diff = torch.matmul(torch.inverse(R1), R2)
    angle_diff = angle_of_R(R_diff)
    return angle_diff
