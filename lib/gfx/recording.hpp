#pragma once

#include "tex_reinterpret_conv.hpp"
#include "types.hpp"
#include "../imgui.hpp"

#include <cstddef>
#include <memory>
#include <string>

namespace aurora::gfx::detail {

struct FramePacket;

struct RecordedFrame {
  FramePacket* packet;
  size_t frameSlot;
};

void begin_recording(FramePacket& packet, size_t frameSlot);
void prepare_recording();
webgpu::TextureWithSampler xfb_present_source();
void record_xfb_present(imgui::DrawData drawData, wgpu::BindGroup rmlBindGroup, bool rmlOverlay);
RecordedFrame end_recording();
void shutdown_recording();
void increment_merged_draw_count() noexcept;

namespace testing {
struct OffscreenCacheStats {
  uint64_t bytes;
  size_t entries;
};
void suppress_render_worker(bool suppress) noexcept;
void seed_offscreen_cache(uint32_t width, uint32_t height, wgpu::TextureFormat colorFormat,
                          wgpu::TextureFormat depthFormat);
OffscreenCacheStats offscreen_cache_stats() noexcept;
bool is_offscreen_cached(uint32_t width, uint32_t height, wgpu::TextureFormat colorFormat,
                         wgpu::TextureFormat depthFormat);
webgpu::TextureWithSampler recorded_present_source(PresentTargetId target);
} // namespace testing

} // namespace aurora::gfx::detail

namespace aurora::gfx::readback {
struct Job;
}

namespace aurora::gfx {
struct ColorPassDescriptor {
  const char* label = nullptr;
  wgpu::TextureView colorView;
  wgpu::TextureView resolveView;
  wgpu::TextureFormat colorFormat = wgpu::TextureFormat::Undefined;
  wgpu::TextureView depthStencilView;
  wgpu::TextureFormat depthStencilFormat = wgpu::TextureFormat::Undefined;
  wgpu::Extent3D targetSize;
  uint32_t sampleCount = 1;
  wgpu::LoadOp colorLoadOp = wgpu::LoadOp::Clear;
  wgpu::StoreOp colorStoreOp = wgpu::StoreOp::Store;
  wgpu::Color clearColor{0.f, 0.f, 0.f, 0.f};
  bool hasDepth = false;
  wgpu::LoadOp depthLoadOp = wgpu::LoadOp::Undefined;
  wgpu::StoreOp depthStoreOp = wgpu::StoreOp::Undefined;
  float depthClearValue = 0.f;
  bool hasStencil = false;
  wgpu::LoadOp stencilLoadOp = wgpu::LoadOp::Undefined;
  wgpu::StoreOp stencilStoreOp = wgpu::StoreOp::Undefined;
  uint32_t stencilClearValue = 0;
};

void finish();
void begin_color_pass(const ColorPassDescriptor& desc);
void end_color_pass();
void queue_texture_copy(wgpu::TexelCopyTextureInfo src, wgpu::TexelCopyTextureInfo dst, wgpu::Extent3D size);
// efbScale: width and height are in EFB coordinates, and the framebuffer renders at the EFB's render scale
// (GXCreateScaledFrameBuffer); otherwise they are the framebuffer's pixels.
void begin_offscreen(uint32_t width, uint32_t height, bool efbScale = false);
void end_offscreen();
// The open offscreen framebuffer's size in EFB coordinates and the EFB's render size, whose scale it renders at;
// zero on the EFB and in an offscreen framebuffer of its own scale.
Vec2<uint32_t> offscreen_logical_size() noexcept;
Vec2<uint32_t> offscreen_efb_size() noexcept;
bool has_normal_attachment() noexcept;
RenderTargetLayout get_render_target_layout() noexcept;
namespace tex_palette_conv {
struct ConvRequest;
}
void queue_palette_conv(tex_palette_conv::ConvRequest req);
void queue_reinterpret_conv(tex_reinterpret_conv::ConvRequest req);

Range push_verts(const uint8_t* data, size_t length, size_t alignment);
template <typename T>
Range push_verts(ArrayRef<T> data, size_t alignment) {
  return push_verts(reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(T), alignment);
}
Range push_indices(const uint8_t* data, size_t length, size_t alignment);
template <typename T>
Range push_indices(ArrayRef<T> data, size_t alignment) {
  return push_indices(reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(T), alignment);
}
Range push_uniform(const uint8_t* data, size_t length);
// A uniform block written in place: begin_uniform aligns the frame's uniform buffer for it and returns the buffer to
// append it to (null outside a recording frame), end_uniform its range. It saves the block's copy push_uniform makes.
ByteBuffer* begin_uniform(size_t& begin);
Range end_uniform(size_t begin, const ByteBuffer& target);
template <typename T>
Range push_uniform(const T& data) {
  return push_uniform(reinterpret_cast<const uint8_t*>(&data), sizeof(T));
}
Range push_storage(const uint8_t* data, size_t length);
// A vertex array in the storage buffer, kept from frame to frame by its contents: an array the cache holds is not
// copied again, one it does not hold enters it through the frame's staging storage. An address whose contents change
// is pushed with each frame's storage (push_storage) instead, as are arrays that find the cache full, which is emptied
// at the next frame's start. The range's offset is from the storage buffer's start.
Range push_cached_storage(const uint8_t* data, size_t length);
// Empties the array cache, for a new storage buffer.
void reset_array_cache() noexcept;
template <typename T>
Range push_storage(ArrayRef<T> data) {
  return push_storage(reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(T));
}
template <typename T>
Range push_storage(const T& data) {
  return push_storage(reinterpret_cast<const uint8_t*>(&data), sizeof(T));
}
Range push_texture_data(const uint8_t* data, uint32_t bytesPerRow, uint32_t rowsPerImage);

template <typename DrawData>
void push_draw_command(DrawData data);
template <typename DrawData>
DrawData* get_last_draw_command();
template <typename PipelineConfig>
PipelineRef pipeline_ref(const PipelineConfig& config);

// boxFilterTaps: 0 for a plain copy, else the number of source pixels per side that each texel
// averages (the copy's box filter). dstRect: the part of the texture to write, over its contents;
// empty for the whole texture.
void resolve_pass_into(TextureHandle texture, ClipRect rect, bool clearColor, bool clearAlpha, bool clearDepth,
                       Vec4<float> clearColorValue, float clearDepthValue, GXTexFmt resolveFormat,
                       GXPixelFmt sourceFormat, uint32_t boxFilterTaps = 0, ClipRect dstRect = {});
// A display copy's place in the emulated XFB: the frame buffer in main memory it writes into.
struct XfbCopy {
  // The copy's destination address and the frame buffer's line pitch in bytes.
  uintptr_t dest = 0;
  uint32_t stride = 0;
  // The lines the copy writes (after its Y scale), and the lines of the frame buffer the video
  // interface shows, which size a frame buffer the copy starts.
  uint32_t lines = 0;
  uint32_t fbLines = 0;
  // Whether the Y scale stretches the source lines over a different number of lines.
  bool scaled = false;
  // Render target pixels per frame buffer pixel.
  Vec2<float> scale{1.f, 1.f};
};
void copy_disp(ClipRect rect, const XfbCopy& xfb, bool clearColor, bool clearAlpha, bool clearDepth,
               Vec4<float> clearColorValue, float clearDepthValue);
void clear_rect(ClipRect rect, bool clearColor, bool clearAlpha, bool clearDepth, Vec4<float> clearColorValue,
                float clearDepthValue);
uint32_t align_uniform(uint32_t value);
Vec2<uint32_t> get_render_target_size() noexcept;
void set_viewport(const Viewport& viewport) noexcept;
void set_scissor(const ClipRect& scissor) noexcept;
void push_debug_group(std::string label);
void insert_debug_marker(std::string label);

struct ReadbackCapture {
  bool active = false;
  bool offscreen = false;
  uint32_t width = 0;
  uint32_t height = 0;
};
ReadbackCapture readback_capture() noexcept;
bool push_readback_op(std::shared_ptr<readback::Job> job, bool efbSource);
} // namespace aurora::gfx
