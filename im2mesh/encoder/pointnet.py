import torch
import torch.nn as nn

from im2mesh.layers import ResnetBlockFC, VNResnetBlockFC
import im2mesh.vnn as vnn


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class VNResnetPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, pooling='mean'):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = vnn.Linear(3, 2*hidden_dim // 3)

        self.block_0 = VNResnetBlockFC(2*hidden_dim//3, hidden_dim//3)
        self.block_1 = VNResnetBlockFC(2*hidden_dim//3, hidden_dim//3)
        self.block_2 = VNResnetBlockFC(2*hidden_dim//3, hidden_dim//3)
        self.block_3 = VNResnetBlockFC(2*hidden_dim//3, hidden_dim//3)
        self.block_4 = VNResnetBlockFC(2*hidden_dim//3, hidden_dim//3)

        self.fc_c = vnn.Linear(hidden_dim//3, c_dim//3)

        self.actvn = vnn.LeakyReLU(hidden_dim//3, share_nonlinearity=False, negative_slope=0)

        if pooling == 'max':
            self.pool_pos = vnn.MaxPool(2*hidden_dim//3)
            self.pool_0 = vnn.MaxPool(2*hidden_dim//3)
            self.pool_1 = vnn.MaxPool(2*hidden_dim//3)
            self.pool_2 = vnn.MaxPool(2*hidden_dim//3)
            self.pool_3 = vnn.MaxPool(2*hidden_dim//3)
            self.pool_4 = vnn.MaxPool(2*hidden_dim//3)
        elif pooling == 'mean':
            self.pool_pos = vnn.MeanPool()
            self.pool_0 = vnn.MeanPool()
            self.pool_1 = vnn.MeanPool()
            self.pool_2 = vnn.MeanPool()
            self.pool_3 = vnn.MeanPool()
            self.pool_4 = vnn.MeanPool()

        self.n_knn = 20

    def forward(self, p):
        batch_size, T, D = p.size()
        ## Translation invariant
        p_mean = p.mean(1).unsqueeze(1)
        p = p - p_mean

        p = p.transpose(1, -1).unsqueeze(1)

        net = vnn.util.get_graph_feature_cross(p, k=self.n_knn)
        net = self.fc_pos(net)
        net = self.pool_pos(net)

        net = self.block_0(net)
        pooled = self.pool_0(net, dim=3, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_1(net)
        pooled = self.pool_1(net, dim=3, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_2(net)
        pooled = self.pool_2(net, dim=3, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_3(net)
        pooled = self.pool_3(net, dim=3, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_4(net)

        # Reduce to  B x F x 3
        net = self.pool_4(net, dim=3)

        c = self.fc_c(self.actvn(net))

        c = torch.flatten(c, 1) # B x (3F)

        return c


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
