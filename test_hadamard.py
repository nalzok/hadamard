import torch
from scipy.linalg import hadamard

from hadamard import hadamard_transform_torch, hadamard_transform_cuda


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Repeat:
    def __init__(self, hadamard_impl, dummy_in, times):
        self.hadamard_impl = hadamard_impl
        self.dummy_in = dummy_in
        self.times = times

    def __enter__(self):
        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = self.hadamard_impl(self.dummy_in)
        torch.cuda.current_stream().wait_stream(s)

        # Captures the graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(self.times):
                self.dummy_in = self.hadamard_impl(self.dummy_in)

        return g, self.dummy_in

    def __exit__(self, type, value, traceback):
        pass


class Benchmark:
    def __init__(self, name):
        self.name = name
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start_event.record()
        return self.start_event

    def __exit__(self, type, value, traceback):
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = self.start_event.elapsed_time(self.end_event)
        print(f"[{self.name.ljust(16)}] Elapsed: ", elapsed_time_ms)


def test_hadamard_transform():
    m = 10
    n = 1 << m
    batch_size = 1 << 0
    u = torch.rand((batch_size, n), dtype=torch.float16, device=device)

    dummy = torch.clone(u)
    with Benchmark("Torch (loop)"):
        for _ in range(1024):
            dummy = hadamard_transform_torch(dummy)

    with Repeat(hadamard_transform_torch, u, 1024) as (g, result):
        with Benchmark("Torch (graph)"):
            g.replay()
        result_torch = torch.clone(result)

    dummy = torch.clone(u)
    with Benchmark("CUDA (loop)"):
        for _ in range(1024):
            dummy = hadamard_transform_cuda(dummy)
    
    with Repeat(hadamard_transform_cuda, u, 1024) as (g, result):
        with Benchmark("CUDA (graph)"):
            g.replay()
        result_cuda = torch.clone(result)

    # Explicit construction from scipy
    # H = torch.tensor(hadamard(n), dtype=torch.float16, device=device)
    # result_explicit = u @ H.t()
    # print("---------")
    # print("[Torch] L-inf", (result_torch - result_explicit).abs().max().item())
    # print("[Torch] L-1", (result_torch - result_explicit).abs().mean().item())
    # print("[CUDA ] L-inf", (result_cuda - result_explicit).abs().max().item())
    # print("[CUDA ] L-1", (result_cuda - result_explicit).abs().mean().item())


if __name__ == '__main__':
    test_hadamard_transform()
