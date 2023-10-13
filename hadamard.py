import time

import numpy as np
import torch
from scipy.linalg import hadamard

import hadamard_cuda


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def hadamard_transform_cuda_optimized(u, normalize=False):
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
    output = HadamardTransformCudaOptimized.apply(u)
    return output / 2**(m / 2) if normalize else output


def test_hadamard_transform():
    m = 14
    n = 1 << m
    batch_size = 1 << 12
    u = torch.rand((batch_size, n), dtype=torch.float16, requires_grad=True, device=device)

    start = time.perf_counter()
    result_torch = hadamard_transform_torch(u)
    grad_torch, = torch.autograd.grad(result_torch.sum(), u, retain_graph=True)
    print("[Torch           ] Elapsed: ", time.perf_counter() - start)

    start = time.perf_counter()
    result_cuda = hadamard_transform_cuda(u)
    grad_cuda, = torch.autograd.grad(result_cuda.sum(), u, retain_graph=True)
    print("[CUDA            ] Elapsed: ", time.perf_counter() - start)

    start = time.perf_counter()
    result_cuda_optimized = hadamard_transform_cuda_optimized(u)
    grad_cuda_optimized, = torch.autograd.grad(result_cuda_optimized.sum(), u, retain_graph=True)
    print("[CUDA (Optimized)] Elapsed: ", time.perf_counter() - start)

    # Explicit construction from scipy
    H = torch.tensor(hadamard(n), dtype=torch.float16, device=device)
    result_explicit = u @ H.t()
    print("------------------")
    print("[Torch           ] L-inf", (result_torch - result_explicit).abs().max().item())
    print("[Torch           ] L-1", (result_torch - result_explicit).abs().mean().item())
    print("[CUDA            ] L-inf", (result_cuda - result_explicit).abs().max().item())
    print("[CUDA            ] L-1", (result_cuda - result_explicit).abs().mean().item())
    print("[CUDA (Optimized)] L-inf", (result_cuda_optimized - result_explicit).abs().max().item())
    print("[CUDA (Optimized)] L-1", (result_cuda_optimized - result_explicit).abs().mean().item())
    print("------------------")
    print("[CUDA            ] Grad L-inf", (grad_cuda - grad_torch).abs().max().item())
    print("[CUDA            ] Grad L-1", (grad_cuda - grad_torch).abs().mean().item())
    print("[CUDA (Optimized)] Grad L-inf", (grad_cuda_optimized - grad_torch).abs().max().item())
    print("[CUDA (Optimized)] Grad L-1", (grad_cuda_optimized - grad_torch).abs().mean().item())


hadamard_transform = hadamard_transform_cuda

if __name__ == '__main__':
    test_hadamard_transform()
