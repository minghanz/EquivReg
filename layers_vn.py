import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.autograd.profiler as profiler
from torch_batch_svd import svd

EPS = 1e-6

class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        return x_out


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2, global_relu=False):
        super(VNLeakyReLU, self).__init__()
        self.global_relu = global_relu
        self.share_nonlinearity = share_nonlinearity
        if global_relu:
            self.map_to_dir = mean_pool
        else:
            if share_nonlinearity == True:
                self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
            else:
                self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        if self.global_relu:
            d = self.map_to_dir(x, -1, keepdim=True)
        else:
            d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1-self.negative_slope) * (mask*x + (1-mask)*(x-(dotprod/(d_norm_sq+EPS))*d))
        return x_out


class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm=False, 
                negative_slope=0.2, global_relu=False):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.negative_slope = negative_slope
        self.global_relu = global_relu
        
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.batchnorm = VNBatchNorm(out_channels, dim=dim)
        
        if global_relu:
            self.map_to_dir = mean_pool
        else:
            if share_nonlinearity == True:
                self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
            else:
                self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        p = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        # BatchNorm
        if self.use_batchnorm:
            p = self.batchnorm(p)
        # LeakyReLU
        if self.global_relu:
            d = self.map_to_dir(p, -1, keepdim=True)
        else:
            d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (p*d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1-self.negative_slope) * (mask*p + (1-mask)*(p-(dotprod/(d_norm_sq+EPS))*d))
        return x_out


class VNLinearAndLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm=False, negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope
        
        self.linear = VNLinear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        
        # BatchNorm
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Conv
        x = self.linear(x)
        # InstanceNorm
        if self.use_batchnorm:
            x = self.batchnorm(x)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        norm = torch.sqrt((x*x).sum(2))
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn
        
        return x


class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super(VNMaxPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
    
    def forward(self, x, keepdim=False):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]
        if keepdim:
            x_max = x_max.unsqueeze(-1)
        return x_max


def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)


class VNStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2):
        super(VNStdFeature, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame
        
        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels//2, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        self.vn2 = VNLinearLeakyReLU(in_channels//2, in_channels//4, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels//4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels//4, 3, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        z0 = x
        z0 = self.vn1(z0)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)
        
        if self.normalize_frame:
            # make z0 orthogonal. u2 = v2 - proj_u1(v2)
            v1 = z0[:,0,:]
            #u1 = F.normalize(v1, dim=1)
            v1_norm = torch.sqrt((v1*v1).sum(1, keepdims=True))
            u1 = v1 / (v1_norm+EPS)
            v2 = z0[:,1,:]
            v2 = v2 - (v2*u1).sum(1, keepdims=True)*u1
            #u2 = F.normalize(u2, dim=1)
            v2_norm = torch.sqrt((v2*v2).sum(1, keepdims=True))
            u2 = v2 / (v2_norm+EPS)

            # compute the cross product of the two output vectors        
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)
        
        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)
        
        return x_std, z0


def get_graph_feature_lrf(x, k=20, idx=None, ball_radius=0, lrf_cross=False):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)      # B*3*N
    if idx is None:
        idx = knn(x, k=k, ball_radius=ball_radius, include_self=True)
    # device = torch.device('cuda')
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()          # B*N*3
    feature = x.view(batch_size*num_points, -1)[idx, :]             #(B*N*K)*3
    feature = feature.view(batch_size, num_points, k, 3)            # B*N*K*3
    feature_mean = feature.mean(2, keepdim=True)                    # B*N*1*3
    feature_ctrd = feature - feature_mean
    feature_cov = torch.matmul(feature_ctrd.transpose(-1, -2), feature_ctrd) / (k-1)    # B*N*3*3

    # with profiler.record_function("SVD"):
    feature_cov = feature_cov.reshape(-1, 3, 3)
    U, sigma, V = svd(feature_cov)
    U = U.reshape(batch_size, num_points, 3, 3)
    sigma = sigma.reshape(batch_size, num_points, 3)
    # U, sigma, V = torch.svd(feature_cov)
    ### U: B*N*3*3, sigma: B*N*3

    # with profiler.record_function("FEATURE GEN"):
    x = x.view(batch_size, num_points, 1, 3)
    dummy_ones = torch.ones_like(x)
    sign_u = torch.where(torch.matmul(x, U) > 0, dummy_ones, -dummy_ones)   # B*N*1*3

    # print("sigma", sigma.shape, sigma)   # B*N*3
    sigma = torch.clamp_min(sigma, 1e-8)
    sigma = torch.sqrt(torch.clamp_min(sigma, 1e-8))
    scale_1 = sigma[..., [0]] - sigma[..., [1]]   # B*N*1
    scale_3 = sigma[..., [1]] - sigma[..., [2]]
    scale_2 = torch.min(scale_1, scale_3)
    scale_u = torch.stack([scale_1, scale_2, scale_3], dim=-1) # B*N*1*3
    # print("scale_u", scale_u.shape, scale_u)   # B*N*1*3
    U = U * scale_u * sign_u
    U = U.transpose(2, 3) # B*N*3*3 # make sure the xyz dimension is at dimension 3, dimension 2 is channel (there are 3 channels corresponding to 3 principal directions)

    if lrf_cross:
        u3 = U[:, :, [2]]   # B*N*1*3 
        v = torch.cross(x, u3, dim=3)
        features = torch.cat([x, u3, v], dim=2)
    else:
        # feature_mean_deviation = feature_mean - x   # B*N*1*3
        # features = torch.cat([x, feature_mean_deviation, U], dim=2) # B*N*5*3
        features = torch.cat([x, U], dim=2) # B*N*4*3

    # print("x", x.shape, x)
    # print("feature_mean_deviation", feature_mean_deviation.shape, feature_mean_deviation)
    # print("U", U.shape, U)
    features = features.permute(0, 2, 3, 1) # B*(3or4or5)*3*N
    # print("features", features.shape, features)
    return features

def get_graph_feature_cross(x, k=20, idx=None, ball_radius=0):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)      # B*3*N
    if idx is None:
        idx = knn(x, k=k, ball_radius=ball_radius)
    # device = torch.device('cuda')
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()          # B*N*3
    feature = x.view(batch_size*num_points, -1)[idx, :]             #(B*N*K)*3
    feature = feature.view(batch_size, num_points, k, num_dims, 3)  # B*N*K*1*3
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)    # B*N*K*1*3
    cross = torch.cross(feature, x, dim=-1)     # B*N*K*1*3
    
    feature = torch.cat((feature-x, x, cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()
    ### B*N*K*3*3 -> B*3*3*N*K
    return feature


def knn(x, k, ball_radius=0, include_self=False):
    ### x: B*3*N
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)    # B*N*N
    # pairwise_distance_ori = pairwise_distance.clone()
    
    n_ele = pairwise_distance.shape[1]
    mask = torch.eye(n_ele, dtype=x.dtype, device=x.device).unsqueeze(0).expand(pairwise_distance.shape[0], -1, -1)
    pairwise_distance = pairwise_distance * (1-mask) - mask

    # dummy_large_dist = - torch.ones_like(pairwise_distance)
    # pairwise_distance = torch.where(pairwise_distance >= -1e-8, dummy_large_dist, pairwise_distance)
    if ball_radius > 0:
        ### use ball query instead of knn
        rsqr = ball_radius * ball_radius
        random_sqr_dist = - torch.rand_like(pairwise_distance) * rsqr   # (-rsqr, 0]
        pairwise_distance = torch.where(pairwise_distance >= -rsqr, random_sqr_dist, pairwise_distance)
    
    if include_self:
        idx = pairwise_distance.topk(k=k-1, dim=-1)[1]   # (batch_size, num_points, k-1)
        n_points = pairwise_distance.shape[1]
        batch_size = pairwise_distance.shape[0]
        idx_self = torch.arange(n_points, device=idx.device, dtype=idx.dtype).reshape(1, -1, 1).expand(batch_size, -1, -1)  # batch_size, num_points, 1
        idx = torch.cat([idx_self, idx], dim=2)          # (batch_size, num_points, k)
    else:
        idx = pairwise_distance.topk(k=k, dim=-1)[1]

    # value, idx = pairwise_distance.topk(k=k, dim=-1)   # (batch_size, num_points, k)
    # value = torch.gather(pairwise_distance_ori, -1, idx)
    # value = torch.sqrt(-value)
    # # value = -value # torch.sqrt(-value)
    # d_max = value.max(-1)[0]
    # print("d_max", d_max.shape, d_max)
    # d_min = value.min(-1)[0]
    # print("d_min", d_min.shape, d_min)
    return idx