import torch
import torch.nn as nn
from layers import ResnetBlockFC, VNResnetBlockFC
from layers_vn import VNLinear, VNLeakyReLU, VNBatchNorm, VNLinearLeakyReLU, VNMaxPool, get_graph_feature_cross, mean_pool, get_graph_feature_lrf

import torch.autograd.profiler as profiler


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class SimplePointnet(nn.Module):
    ''' PointNet-based encoder network.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.fc_0 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.fc_0(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))

        # Recude to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))

        return c


class ResnetPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)

        # Recude to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))

        return c
class VNResnetPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, pooling='mean', ball_radius=0, 
                    init_lrf=False, lrf_cross=False, n_knn=20, global_relu=False):
        super().__init__()
        self.c_dim = c_dim

        self.init_lrf = init_lrf
        self.lrf_cross = lrf_cross
        self.global_relu = global_relu
        if self.init_lrf:
            if lrf_cross:
                self.fc_pos = VNLinear(3, 2*hidden_dim // 3)
            else:
                self.fc_pos = VNLinear(4, 2*hidden_dim // 3)
        else:
            # self.fc_pos = nn.Linear(dim, 2*hidden_dim)
            self.fc_pos = VNLinear(3, 2*hidden_dim // 3)

        self.block_0 = VNResnetBlockFC(2*hidden_dim//3, hidden_dim//3, global_relu=global_relu)
        self.block_1 = VNResnetBlockFC(2*hidden_dim//3, hidden_dim//3, global_relu=global_relu)
        self.block_2 = VNResnetBlockFC(2*hidden_dim//3, hidden_dim//3, global_relu=global_relu)
        self.block_3 = VNResnetBlockFC(2*hidden_dim//3, hidden_dim//3, global_relu=global_relu)
        self.block_4 = VNResnetBlockFC(2*hidden_dim//3, hidden_dim//3, global_relu=global_relu)
        # self.fc_c = nn.Linear(hidden_dim, c_dim)
        self.fc_c = VNLinear(hidden_dim//3, c_dim//3)

        # self.actvn = nn.ReLU()
        self.actvn = VNLeakyReLU(hidden_dim//3, share_nonlinearity=False, negative_slope=0, global_relu=global_relu)

        # self.pool = maxpool
        self.pooling = pooling
        if self.pooling == 'max':
            self.pool = VNMaxPool(2*hidden_dim//3)
            self.pool_0 = VNMaxPool(hidden_dim//3)
            self.pool_1 = VNMaxPool(hidden_dim//3)
            self.pool_2 = VNMaxPool(hidden_dim//3)
            self.pool_3 = VNMaxPool(hidden_dim//3)
            self.pool_4 = VNMaxPool(hidden_dim//3)

        elif self.pooling == 'mean':
            self.pool = mean_pool
            self.pool_0 = mean_pool
            self.pool_1 = mean_pool
            self.pool_2 = mean_pool
            self.pool_3 = mean_pool
            self.pool_4 = mean_pool

        self.n_knn = n_knn
        self.ball_radius = ball_radius

    def forward(self, p):
        batch_size, T, D = p.size()

        p = p.transpose(1, 2)   # B*dimension*npoints
        p = p.unsqueeze(1)
        # print("0", x.shape) # B, 1, 3, N
        if self.init_lrf:

            # with profiler.profile() as prof:
            with torch.no_grad():
                feat = get_graph_feature_lrf(p, k=self.n_knn, ball_radius=self.ball_radius, lrf_cross=self.lrf_cross) # B*4*3*N
            # print(prof.key_averages().table(row_limit=5))

            net = self.fc_pos(feat)   # B, F, 3, N
        else:
            feat = get_graph_feature_cross(p, k=self.n_knn, ball_radius=self.ball_radius)
            # print("1", feat.shape)  # B, 3, 3, N, knn
            x = self.fc_pos(feat)
            # print("2", x.shape)     # B, F, 3, N, knn
            net = self.pool(x)
            # print("3", x.shape)     # B, F, 3, N

        net = self.block_0(net)
        # print("block_0 net.shape", net.shape)   # [B, 342(hidden_dim//3), 3, 1024(N)]
        # pooled = self.pool(net, dim=3, keepdim=True).expand(net.size())
        pooled = self.pool_0(net, keepdim=True)
        # print("pooled.shape", pooled.shape)
        pooled = pooled.expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_1(net)
        # print("block_1 net.shape", net.shape)
        # pooled = self.pool(net, dim=3, keepdim=True).expand(net.size())
        pooled = self.pool_1(net, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_2(net)
        # print("block_2 net.shape", net.shape)
        # pooled = self.pool(net, dim=3, keepdim=True).expand(net.size())
        pooled = self.pool_2(net, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_3(net)
        # print("block_3 net.shape", net.shape)
        # pooled = self.pool(net, dim=3, keepdim=True).expand(net.size())
        pooled = self.pool_3(net, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_4(net)     # B*F*3*N

        # Reduce to  B x F x 3
        # net = self.pool(net, dim=3)
        net = self.pool_4(net)
        # print("block_4 net.shape", net.shape)

        c = self.fc_c(self.actvn(net))

        c = torch.flatten(c, 1) # B*F

        return c

        # # output size: B x T X F
        # net = self.fc_pos(p)

        # net = self.block_0(net)
        # pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        # net = torch.cat([net, pooled], dim=2)

        # net = self.block_1(net)
        # pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        # net = torch.cat([net, pooled], dim=2)

        # net = self.block_2(net)
        # pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        # net = torch.cat([net, pooled], dim=2)

        # net = self.block_3(net)
        # pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        # net = torch.cat([net, pooled], dim=2)

        # net = self.block_4(net)

        # # Recude to  B x F
        # net = self.pool(net, dim=1)

        # c = self.fc_c(self.actvn(net))

        # return c
