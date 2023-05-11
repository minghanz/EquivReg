import torch
import torch.optim as optim
from torch import autograd
import numpy as np
from tqdm import trange
import trimesh
from utils import libmcubes
from common import make_3d_grid
from utils.libsimplify import simplify_mesh
from utils.libmise import MISE
import time
import logging

# from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations, axis_angle_to_matrix

from transforms import SubSamplePairBatchIP, CentralizePairBatchIP, RotatePairBatchIP
class Generator3D(object):
    '''  Generator class for Occupancy Networks.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution_0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        use_sampling (bool): whether z should be sampled
        simplify_nfaces (int): number of faces the mesh should be simplified to
        preprocessor (nn.Module): preprocessor for inputs
    '''

    def __init__(self, model, points_batch_size=100000,
                 threshold=0.5, refinement_step=0, device=None,
                 resolution_0=16, upsampling_steps=3,
                 with_normals=False, padding=0.1, use_sampling=False,
                 simplify_nfaces=None,
                 preprocessor=None, 
                 rotate=-1, 
                 noise=0, 
                 centralize=False,
                 n1=0, n2=0, subsamp=True, reg_benchmark=False, transform_test=None, **kwargs,
                 ):
        self.model = model.to(device)
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.device = device
        self.resolution_0 = resolution_0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.padding = padding
        self.use_sampling = use_sampling
        self.simplify_nfaces = simplify_nfaces
        self.preprocessor = preprocessor

        self.transform_test = transform_test

        # # self.rotate = rotate
        # # self.noise = noise      # noise only effective when not sampling different points
        # self.subsamp = subsamp
        # self.sub_op = SubSamplePairBatchIP(n1, n2, n2, device) if subsamp else None
        # # self.rotate_op = RotatePairBatchIP()
        # self.centralize = centralize
        # self.ctr_op = CentralizePairBatchIP() if centralize else None

    def generate_latent_conditioned(self, data):
        self.model.eval()
        # device = self.device
        # stats_dict = {}

        if self.transform_test is not None:
            self.transform_test(data)

        inputs = data['inputs']
        inputs_2 = data['inputs_2']
        
        input_max = torch.max(torch.abs(inputs))
        norm_max = torch.max(torch.norm(inputs, dim=-1))
        logging.debug(f"max inf norm, max 2 norm, {input_max}, {norm_max}")
        
        # rot_d = {}
        # rot_d['angles'] = data['T21.deg']
        # # rot_d['trot'] = trot
        # rot_d['rotmats'] = data['T21']
        # # rot_d['t'] = t

        # Encode inputs
        # t0 = time.time()
        with torch.no_grad():
            c = self.model.encode_inputs(inputs)
            c_rot = self.model.encode_inputs(inputs_2)

        return c, c_rot
