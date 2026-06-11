// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "geminifs.cuh"
#include "io_deamon.cuh"

namespace py = pybind11;

namespace {

void check_cuda_launch(const char* name) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::ostringstream oss;
    oss << name << " failed: " << cudaGetErrorString(err);
    throw std::runtime_error(oss.str());
  }
}

void require_success(bool ok, const std::string& message) {
  if (!ok) {
    throw std::runtime_error(message);
  }
}

// RAII NVTX range: pushes a named range on construction and pops it on
// destruction, so the range is correctly closed even if the scope exits via
// an exception.
class NvtxRange {
 public:
  explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};

std::vector<size_t> to_shape(const std::vector<uint64_t>& input) {
  std::vector<size_t> shape;
  shape.reserve(input.size());
  for (uint64_t value : input) {
    shape.push_back(static_cast<size_t>(value));
  }
  return shape;
}

class PyGeminiFS {
 public:
  PyGeminiFS(const std::string& config_file_path, int gpu_file_nums,
             const std::vector<uint64_t>& gpu_file_shape, bool reset)
      : fs_(std::make_unique<GeminiFS>(config_file_path, gpu_file_nums,
                                       to_shape(gpu_file_shape), reset)) {}

  bool is_initialized() const { return fs_->is_initialized(); }

  void launch_io_deamon_kernels() { fs_->launch_io_deamon_kernels(); }

  void stop_io_deamon_kernels() { fs_->stop_io_deamon_kernels(); }

  bool is_io_deamon_kernel_launched() const {
    return fs_->is_io_deamon_kernel_launched();
  }

  uint32_t open_file(int device_id) {
    GPUFileId file_id{};
    require_success(
        fs_->geminifs_gpu_open_file(device_id, file_id),
        "GeminiFS.open_file failed for device_id=" + std::to_string(device_id));
    return static_cast<uint32_t>(file_id);
  }

  bool close_file(int device_id, uint32_t gpu_file_id) {
    require_success(
        fs_->geminifs_gpu_close_file(device_id, static_cast<GPUFileId>(gpu_file_id)),
        "GeminiFS.close_file failed for device_id=" + std::to_string(device_id) +
            ", gpu_file_id=" + std::to_string(gpu_file_id));
    return true;
  }

  bool register_tensor(const torch::Tensor& tensor, uint64_t granularity) {
    require_success(
        fs_->geminifs_register_tensor_with_gpu(tensor, granularity),
        "GeminiFS.register_tensor failed");
    return true;
  }

  bool register_tensors(const std::vector<torch::Tensor>& tensors,
                        uint64_t granularity) {
    require_success(
        fs_->geminifs_register_tensors_with_gpu(tensors, granularity),
        "GeminiFS.register_tensors failed");
    return true;
  }

  bool unregister_tensor(const torch::Tensor& tensor) {
    require_success(fs_->geminifs_unregister_tensor_from_gpu(tensor),
                    "GeminiFS.unregister_tensor failed");
    return true;
  }

  uintptr_t get_client_ctrl_ptr(int client_gpu_id, int server_gpu_id) {
    DeamonManager* manager = fs_->get_deamon_manager();
    if (manager == nullptr) {
      throw std::runtime_error(
          "GeminiFS.get_client_ctrl_ptr failed: daemon manager is unavailable");
    }
    DeamonState* deamon = manager->get_deamon(client_gpu_id);
    if (deamon == nullptr) {
      throw std::runtime_error(
          "GeminiFS.get_client_ctrl_ptr failed: client daemon is unavailable");
    }
    nvl_queue_client_ctrl_t* client = deamon->get_client_ctrl(server_gpu_id);
    if (client == nullptr) {
      throw std::runtime_error(
          "GeminiFS.get_client_ctrl_ptr failed: client control pointer is unavailable");
    }
    return reinterpret_cast<uintptr_t>(client);
  }

 private:
  std::unique_ptr<GeminiFS> fs_;
};

void launch_remote_io_xfer_batch(uintptr_t desc_ptr, uint64_t num_descs,
                                 bool is_read, uintptr_t stream_ptr) {
  NvtxRange nvtx_range("launch_remote_io_xfer_batch");

  if (num_descs == 0) {
    return;
  }
  if (desc_ptr == 0) {
    throw std::runtime_error(
        "launch_remote_io_xfer_batch requires non-zero desc_ptr");
  }

  nvl_batch_io_ctx_t ctx;
  ctx.descs = reinterpret_cast<const nvl_batch_io_desc_t*>(desc_ptr);
  ctx.num_descs = num_descs;
  ctx.is_read = is_read;

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  constexpr unsigned int kThreadsPerBlock = 256;
  unsigned int blocks = static_cast<unsigned int>(
      (num_descs + kThreadsPerBlock - 1) / kThreadsPerBlock);
  deamon_remote_io_xfer_batch_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      ctx);
  check_cuda_launch("deamon_remote_io_xfer_batch_kernel");
}

}  // namespace

PYBIND11_MODULE(geminifs_ops, m) {
  py::class_<PyGeminiFS>(m, "GeminiFS")
      .def(py::init<const std::string&, int, const std::vector<uint64_t>&, bool>(),
           py::arg("config_file_path"), py::arg("gpu_file_nums"),
           py::arg("gpu_file_shape"), py::arg("reset") = false)
      .def("is_initialized", &PyGeminiFS::is_initialized)
      .def("launch_io_deamon_kernels", &PyGeminiFS::launch_io_deamon_kernels,
           py::call_guard<py::gil_scoped_release>())
      .def("stop_io_deamon_kernels", &PyGeminiFS::stop_io_deamon_kernels,
           py::call_guard<py::gil_scoped_release>())
      .def("is_io_deamon_kernel_launched",
           &PyGeminiFS::is_io_deamon_kernel_launched)
      .def("open_file", &PyGeminiFS::open_file, py::arg("device_id"))
      .def("close_file", &PyGeminiFS::close_file, py::arg("device_id"),
           py::arg("gpu_file_id"))
      .def("register_tensor", &PyGeminiFS::register_tensor, py::arg("tensor"),
           py::arg("granularity") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("register_tensors", &PyGeminiFS::register_tensors,
           py::arg("tensors"), py::arg("granularity") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("unregister_tensor", &PyGeminiFS::unregister_tensor,
           py::arg("tensor"), py::call_guard<py::gil_scoped_release>())
      .def("get_client_ctrl_ptr", &PyGeminiFS::get_client_ctrl_ptr,
           py::arg("client_gpu_id"), py::arg("server_gpu_id"));

  m.def("launch_remote_io_xfer_batch", &launch_remote_io_xfer_batch,
        py::arg("desc_ptr"), py::arg("num_descs"), py::arg("is_read"),
        py::arg("stream_ptr") = 0,
        py::call_guard<py::gil_scoped_release>());
}
