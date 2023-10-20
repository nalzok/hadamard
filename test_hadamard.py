import torch

from hadamard import hadamard_transform_torch, hadamard_transform_cuda


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
    dtype = torch.float64
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for m in range(20):
        print(f"{m = }")

        n = 1 << m
        batch_size = 1 << 2
        u = torch.rand((batch_size, n), dtype=dtype, device=device)
        u = torch.arange(batch_size * n, dtype=dtype, device=device).reshape(batch_size, n)

        with Benchmark("Torch"):
            result_torch = hadamard_transform_torch(u)

        with Benchmark("CUDA"):
            result_cuda = hadamard_transform_cuda(u)

        print("Error (L-inf):", (result_torch - result_cuda).abs().max().item())
        print("Error (L-1):", (result_torch - result_cuda).abs().mean().item())


if __name__ == '__main__':
    torch.random.manual_seed(42)
    test_hadamard_transform()
