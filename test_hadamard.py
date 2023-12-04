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
    def __init__(self, dtype, name, show):
        self.dtype = dtype
        self.name = name
        self.show = show
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start_event.record()
        return self.start_event

    def __exit__(self, type, value, traceback):
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = self.start_event.elapsed_time(self.end_event)
        if self.show:
            print(self.dtype, f"[{self.name.ljust(8)}] Elapsed: {elapsed_time_ms:.4f}ms")


def test_hadamard_transform():
    device = torch.device("cuda")

    n = 1 << 13
    batch_size = 1
    u = torch.rand((batch_size, n), device=device)
    ground_truth = hadamard_transform_torch(u)

    for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        print()

        for i in range(3):
            v = u.to(dtype)

            with Benchmark(dtype, "Torch", i == 2):
                result_torch = hadamard_transform_torch(v)

            with Benchmark(dtype, "CUDA", i == 2):
                result_cuda = hadamard_transform_cuda(v)

            if i == 2:
                error = (result_cuda - ground_truth).abs().double()
                print(dtype, "Error (L-inf):", error.max().item())
                print(dtype, "Error (L-1):", error.mean().item())


if __name__ == '__main__':
    torch.random.manual_seed(42)
    test_hadamard_transform()
