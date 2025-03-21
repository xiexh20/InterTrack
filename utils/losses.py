"""
all loss utils
"""
import torch
from pytorch3d.ops import knn_points


def rigid_loss(pts, pts_fixed, k_nn=10):
    """
    local as rigid as possible loss
    Parameters
    ----------
    pts : (B, N, D)
    pts_fixed : (1, N, D)

    Returns
    -------

    """
    B, N = pts.shape[:2]
    # dis_mat_fixed = torch.cdist(pts_fixed, pts_fixed) ** 2  # (1, N, N)
    # dis_mat_kpt = torch.cdist(pts, pts) ** 2  # (B, N, N), this is memory intensive!
    closest_pts = knn_points(pts_fixed, pts_fixed, K=k_nn)
    indices = closest_pts.idx  # 1, N, K
    dist_fixed = closest_pts.dists  # (1, N, K), this is squared distance

    # Create a tensor for the batch dimension indices
    batch_indices = torch.arange(B)[:, None, None]
    # Expand the batch indices to match the shape of y
    batch_indices = batch_indices.expand(-1, N, k_nn)
    # Use advanced indexing to get the values
    pts_nn = pts[batch_indices, indices]  # B, N, K, 3
    dist_pts = torch.norm(pts_nn - pts[:, :, None], dim=-1)  # squared distance

    # _, nn_dist_ind_fixed = torch.topk(dis_mat_fixed, k=k_nn, largest=False, dim=2)  # find the k nearest neighbours
    # index the k-nearest neighbour in this frame
    # dis_mat_knn_fixed = torch.gather(dis_mat_fixed, dim=2, index=nn_dist_ind_fixed)  # 1, N, K
    # index the k-nearest neighbour in other frames
    # dis_mat_knn_kpt = torch.gather(dis_mat_kpt, dim=2, index=nn_dist_ind_fixed.repeat(B, 1, 1))  # B, N, K
    # final_dist = (dis_mat_knn_fixed - dis_mat_knn_kpt) ** 2 # (B, N, K)

    final_dist = (dist_pts - dist_fixed) ** 2
    return final_dist.sum(-1).mean()

def chamfer_distance(s1, s2, w1=1., w2=1., norm='l1'):
    """
    :param s1: B x N x 3
    :param s2: B x M x 3
    :param w1: weight for distance from s1 to s2
    :param w2: weight for distance from s2 to s1
    """


    assert s1.is_cuda and s2.is_cuda
    closest_dist_in_s2 = knn_points(s1, s2, K=1)
    closest_dist_in_s1 = knn_points(s2, s1, K=1)

    if norm == 'l1':
        return (closest_dist_in_s2.dists**0.5 * w1).mean(axis=1).squeeze(-1) + (closest_dist_in_s1.dists**0.5 * w2).mean(axis=1).squeeze(-1)
    elif norm == 'l2':
        # squared distance
        return (closest_dist_in_s2.dists * w1).mean(axis=1).squeeze(-1) + (
                    closest_dist_in_s1.dists * w2).mean(axis=1).squeeze(-1)
    else:
        raise ValueError(f"Unknown norm specification: {norm}!")
