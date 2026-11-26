#pragma once

#include "hash.hpp"
#include "../internal.hpp"
#include "../webgpu/gpu.hpp"

#include <aurora/gfx.h>
#include <aurora/gfx.hpp>
#include <aurora/math.hpp>
#include <dolphin/gx/GXEnum.h>
#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

namespace aurora::gfx {
using BindGroupRef = HashType;
using PipelineRef = HashType;
using SamplerRef = HashType;
using ShaderRef = HashType;

struct ClipRect {
  int32_t x;
  int32_t y;
  int32_t width;
  int32_t height;

  bool operator==(const ClipRect& rhs) const { return memcmp(this, &rhs, sizeof(*this)) == 0; }
  bool operator!=(const ClipRect& rhs) const { return !(*this == rhs); }
};

using webgpu::Viewport;

// Whether a viewport's depth range leaves [0, 1]. GX takes any range and saturates the depth at
// the 24-bit buffer's ends; the GPU's viewport cannot, so such draws apply the range themselves.
inline bool depth_range_outside(const Viewport& viewport) {
  return viewport.znear < 0.f || viewport.zfar > 1.f || viewport.znear > 1.f || viewport.zfar < 0.f;
}

struct TextureRef;
using TextureHandle = std::shared_ptr<TextureRef>;
using AfterSubmitCallback = std::function<void()>;
using EndFrameCallback = std::function<void(wgpu::CommandEncoder&, std::vector<AfterSubmitCallback>)>;
} // namespace aurora::gfx
