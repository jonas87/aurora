#pragma once

#include "types.hpp"

#include <dolphin/gx/GXEnum.h>

namespace aurora::gfx::tex_reinterpret_conv {

// Reads an EFB copy through a texture object of another format, as the hardware does: the copy's
// texels become the bytes the hardware writes in the copy's format and tile layout, and those bytes
// are decoded in the object's format and tile layout. The result has the copy's logical size, not
// Aurora's render scale, since only the bytes carry meaning across formats.

// The copy formats whose bytes can be reconstructed: one byte per texel (an 8-bit channel or the
// intensity, stored in every channel of the copy texture), or RGB565.
enum class Source : uint32_t {
  Bytes8 = 0,
  Rgb565 = 1,
};

// The formats the bytes can be decoded in.
enum class Target : uint32_t {
  I8 = 0,
  Ia8 = 1,
  Rgb565 = 2,
  Rgb5a3 = 3,
};

struct ConvRequest {
  Source source;
  Target target;
  uint32_t sourceWidth;  // the copy's logical size
  uint32_t sourceHeight;
  uint32_t targetWidth;  // the texture object's size
  uint32_t targetHeight;
  TextureHandle src;
  TextureHandle dst;
};

// Whether a texture object of `textureFormat` reading a copy of `copyFormat` needs the bytes
// reinterpreted, and how: false when the object reads the copy's texels as they are.
bool plan(GXTexFmt copyFormat, GXTexFmt textureFormat, Source& source, Target& target);

void initialize();
void shutdown();
void run(const wgpu::CommandEncoder& cmd, const ConvRequest& req);

} // namespace aurora::gfx::tex_reinterpret_conv
