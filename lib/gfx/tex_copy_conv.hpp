#pragma once

#include "types.hpp"

#include <dolphin/gx/GXEnum.h>

#include <array>

namespace aurora::gfx::tex_copy_conv {

enum class SampleFilter : uint8_t {
  Nearest,
  Linear,
};

struct alignas(16) Uniforms {
  Vec2<float> offset;
  Vec2<float> scale{1.f, 1.f};
  uint32_t opaqueAlpha = 0;
  // The box filter: 0 for none, else the source pixels per side each texel averages; the size of
  // a source pixel in texture coordinates.
  uint32_t boxTaps = 0;
  Vec2<float> srcTexel;
};
static_assert(sizeof(Uniforms) == 32);

struct ConvRequest {
  GXTexFmt fmt;
  GXPixelFmt srcFmt;
  wgpu::TextureView srcView; // View of resolved EFB / offscreen color/depth
  Range uniformRange;        // Uniforms
  TextureHandle dst;         // Destination texture
  SampleFilter sampleFilter = SampleFilter::Nearest;
};

bool needs_conversion(GXTexFmt fmt);

void initialize();
void shutdown();
void run(const wgpu::CommandEncoder& cmd, const ConvRequest& req);
void blit(const wgpu::CommandEncoder& cmd, const ConvRequest& req);
// Blits into dstRect of dst over its contents; dst has the EFB format. The request's dst is unused.
void blit_rect(const wgpu::CommandEncoder& cmd, const ConvRequest& req, const wgpu::TextureView& dst,
               const ClipRect& dstRect);

bool snapshot_depth_supported() noexcept;
void snapshot_depth(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& srcDepth, uint32_t msaaSamples,
                    const wgpu::TextureView& dst);
// Resolves a multisampled depth buffer to a single-sampled one (each pixel's first sample) that a depth copy can read.
// With a region (in the buffer's pixels), only the region and a texel around it are resolved, the rest of the view
// left undefined: the copy reads only its source rectangle. The view stays valid for the copy that follows.
wgpu::TextureView resolve_depth(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& srcDepth, uint32_t width,
                                uint32_t height, const ClipRect* region = nullptr);

} // namespace aurora::gfx::tex_copy_conv
