import numpy as np
import torch

import hadamard_cuda


def hadamard_transform_torch(u, normalize=False):
    """Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
    n must be a power of 2.
    Parameters:
        u: Tensor of shape (..., n)
        normalize: if True, divide the result by 2^{m/2} where m = log_2(n).
    Returns:
        product: Tensor of shape (..., n)
    """
    batch_size, n = u.shape
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    x = u[..., np.newaxis]
    for d in range(m)[::-1]:
        x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
    return x.squeeze(-2) / 2**(m / 2) if normalize else x.squeeze(-2)


class HadamardTransformCuda(torch.autograd.Function):
    '''The unnormalized Hadamard transform (i.e. without dividing by sqrt(2))
    '''
    @staticmethod
    def forward(ctx, u):
        return hadamard_cuda.hadamard_transform(u)

    @staticmethod
    def backward(ctx, grad):
        return HadamardTransformCuda.apply(grad)


class HadamardTransformCudaOptimized(torch.autograd.Function):
    '''The unnormalized Hadamard transform (i.e. without dividing by sqrt(2))
    '''
    @staticmethod
    def forward(ctx, u):
        return hadamard_cuda.hadamard_transform_optimized(u)

    @staticmethod
    def backward(ctx, grad):
        return HadamardTransformCudaOptimized.apply(grad)


def hadamard_transform_cuda(u, normalize=False):
    """Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
    n must be a power of 2.
    Parameters:
        u: Tensor of shape (..., n)
        normalize: if True, divide the result by 2^{m/2} where m = log_2(n).
    Returns:
        product: Tensor of shape (..., n)
    """
    _, n = u.shape
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    output = HadamardTransformCuda.apply(u)
    return output / 2**(m / 2) if normalize else output
