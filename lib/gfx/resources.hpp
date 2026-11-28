#pragma once

#include "types.hpp"

namespace aurora::gfx {
inline constexpr bool UseTextureBuffer = true;

// The sizes of the buffers a frame's draws are written to (AuroraConfig::bufferSizes, a size left at 0 taking its
// default here) and their places, in this order, in a frame's staging buffer.
struct BufferSizes {
  uint64_t vertex = 5242880;         // 5 MiB
  uint64_t uniform = 25165824;       // 24 MiB
  uint64_t index = 2097152;          // 2 MiB
  uint64_t storage = 8388608;        // 8 MiB
  uint64_t textureUpload = 25165824; // 24 MiB
  // The storage buffer's part past a frame's own storage that keeps vertex arrays from frame to frame
  // (push_cached_storage); on the GPU only, and none by default.
  uint64_t arrayCache = 0;

  uint64_t uniform_offset() const noexcept { return vertex; }
  uint64_t index_offset() const noexcept { return uniform_offset() + uniform; }
  uint64_t storage_offset() const noexcept { return index_offset() + index; }
  uint64_t texture_upload_offset() const noexcept { return storage_offset() + storage; }
  uint64_t staging() const noexcept { return texture_upload_offset() + (UseTextureBuffer ? textureUpload : 0); }
};
// The sizes the buffers were created with.
const BufferSizes& buffer_sizes() noexcept;

namespace detail {
struct Resources {
  wgpu::Buffer vertexBuffer;
  wgpu::Buffer uniformBuffer;
  wgpu::Buffer indexBuffer;
  wgpu::Buffer storageBuffer;
  wgpu::BindGroupLayout staticBindGroupLayout;
  wgpu::BindGroup staticBindGroup;
  wgpu::BindGroupLayout uniformBindGroupLayout;
  wgpu::BindGroup uniformBindGroup;
  wgpu::Limits limits;
  AuroraStats stats{};
};

Resources& resources() noexcept;
} // namespace detail
} // namespace aurora::gfx
