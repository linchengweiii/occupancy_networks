import torch

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature_tuple(x, k=20, idx=None, x_coord=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)

    x_mean = x.mean(2).unsqueeze(2)
    x -= x_mean
    if idx is None:
        if x_coord is None: # dynamic knn graph
            idx = knn(x, k=k)
        else:          # fixed knn graph with input point coordinates
            idx = knn(x_coord, k=k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x += x_mean

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    feature += x_mean.view(batch_size, 1, 1, num_dims, 3).repeat(1, 1, k, 1, 1)

    return feature, x


def get_graph_feature(x, k=20, idx=None, x_coord=None):
    feature, x = get_graph_feature_tuple(x, k, idx, x_coord)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()
  
    return feature


def get_graph_feature_cross(x, k=20, idx=None):
    feature, x = get_graph_feature_tuple(x, k, idx)
    
    cross = torch.cross(feature, x, dim=-1)
    feature = torch.cat((feature-x, x, cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()
  
    return feature
