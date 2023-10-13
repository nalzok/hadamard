#include <torch/extension.h>
#include <cuda_fp16.h>


extern void fwtBatchGPU(half *x, int batchSize, int log2N);
extern void fwtBatchGPUOptimized(half *x, int batchSize, int log2N);

torch::Tensor hadamard_transform(torch::Tensor x) {
  TORCH_CHECK(x.device().type() == torch::kCUDA, "x must be a CUDA tensor");
  auto n = x.size(-1);
  auto log2N = long(log2(n));
  TORCH_CHECK(n == 1 << log2N, "n must be a power of 2");
  auto output = x.clone();  // Cloning makes it contiguous.
  auto batchSize = x.numel() / (1 << log2N);
  fwtBatchGPU((half *)output.data_ptr<at::Half>(), batchSize, log2N);
  return output;
}

torch::Tensor hadamard_transform_optimized(torch::Tensor x) {
  TORCH_CHECK(x.device().type() == torch::kCUDA, "x must be a CUDA tensor");
  auto n = x.size(-1);
  auto log2N = long(log2(n));
  TORCH_CHECK(n == 1 << log2N, "n must be a power of 2");
  auto output = x.clone();  // Cloning makes it contiguous.
  auto batchSize = x.numel() / (1 << log2N);
  fwtBatchGPUOptimized((half *)output.data_ptr<at::Half>(), batchSize, log2N);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hadamard_transform", &hadamard_transform, "Fast Hadamard transform");
  m.def("hadamard_transform_optimized", &hadamard_transform_optimized, "Faster Hadamard transform");
}
