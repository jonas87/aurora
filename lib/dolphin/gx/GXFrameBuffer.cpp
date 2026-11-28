#include "gx.hpp"
#include "__gx.h"

#include "../../gfx/readback/copy_layout.hpp"
#include "../../gfx/texture.hpp"
#include "../../gfx/recording.hpp"
#include "../../window.hpp"
#include "../../webgpu/gpu.hpp"
#include "../../gx/texture.hpp"
#include "../vi/vi_internal.hpp"

#include <algorithm>
#include <cmath>
#include <optional>

namespace {
// The lines a display copy of height source lines writes at the Y scale register's value scale.
u32 get_num_xfb_lines(u32 height, u32 scale) {
  if (height == 0 || scale == 0) {
    return 0;
  }
  u32 actualHeight = (height - 1) * 0x100 / scale + 1;

  u32 newScale = scale;
  if (newScale > 0x80 && newScale < 0x100) {
    while (newScale % 2 == 0) {
      newScale /= 2;
    }
    if (height % newScale == 0) {
      actualHeight++;
    }
  }

  if (actualHeight > 0x400) {
    actualHeight = 0x400;
  }
  return actualHeight;
}

// The size a texture copy writes: its source's, or half of it through the box filter. The copy's destination size gives
// the length of its rows only.
u32 written_size(s32 source) {
  const u32 size = static_cast<u32>(std::max(source, 0));
  return g_gxState.texCopyMipmap ? size / 2 : size;
}

struct PartialCopy {
  const void* base;
  u32 row;
};

// A copy into part of a larger one that this frame made: rows as long, in the same format, starting at one of its tile
// rows. In a later frame such a copy replaces the larger one instead.
std::optional<PartialCopy> find_partial_copy(const void* dest) {
  const auto format = aurora::gfx::readback::format_info(g_gxState.texCopyFmt);
  const u32 lines = written_size(g_gxState.texCopySrc.height);
  if (!format || lines == 0 || written_size(g_gxState.texCopySrc.width) != g_gxState.texCopyDstWidth) {
    return std::nullopt;
  }
  const uintptr_t rowBytes =
      static_cast<uintptr_t>(aurora::gfx::readback::tiles_x(*format, g_gxState.texCopyDstWidth)) * format->tileBytes;
  const auto address = reinterpret_cast<uintptr_t>(dest);
  const auto frame = aurora::gx::texture::current_frame();
  for (const auto& [base, copy] : g_gxState.copyTextures) {
    const auto start = reinterpret_cast<uintptr_t>(base);
    if (!copy || !copy.whole || copy.frame != frame || copy.key.format != g_gxState.texCopyFmt ||
        copy.width != g_gxState.texCopyDstWidth || address < start || (address - start) % rowBytes != 0) {
      continue;
    }
    const u64 row = static_cast<u64>((address - start) / rowBytes) * format->tileHeight;
    if (lines < copy.height && row + lines <= copy.height) {
      return PartialCopy{base, static_cast<u32>(row)};
    }
  }
  return std::nullopt;
}
} // namespace

namespace aurora::gx {
Vec2<uint32_t> scale_copy_dst(u32 logicalWidth, u32 logicalHeight) {
  if (g_gxState.viewportPolicy == AURORA_VIEWPORT_NATIVE) {
    return {logicalWidth, logicalHeight};
  }

  float scaleX = 1.f;
  float scaleY = 1.f;
  if (!logical_scale(scaleX, scaleY)) {
    return {logicalWidth, logicalHeight};
  }
  const auto scaledWidth = std::max<u32>(static_cast<u32>(std::lround(static_cast<float>(logicalWidth) * scaleX)), 1);
  const auto scaledHeight = std::max<u32>(static_cast<u32>(std::lround(static_cast<float>(logicalHeight) * scaleY)), 1);
  return {scaledWidth, scaledHeight};
}

void copy_tex(const void* dest, GXBool clear) noexcept {
  const auto rect = map_logical_scissor(g_gxState.texCopySrc);
  const auto texCopyFmt = g_gxState.texCopyFmt;
  gfx::TextureHandle target;
  gfx::ClipRect dstRect{};
  if (const auto part = find_partial_copy(dest)) {
    const auto& whole = g_gxState.copyTextures[part->base];
    const float scale = static_cast<float>(whole.handle->size.height) / static_cast<float>(whole.height);
    const auto top = static_cast<float>(part->row) * scale;
    const auto bottom = static_cast<float>(part->row + written_size(g_gxState.texCopySrc.height)) * scale;
    target = whole.handle;
    dstRect.y = static_cast<int32_t>(std::lround(top));
    dstRect.width = static_cast<int32_t>(whole.handle->size.width);
    dstRect.height = static_cast<int32_t>(std::lround(bottom)) - dstRect.y;
    texture::touch_copy_texture(part->base);
  } else {
    const auto [dstWidth, dstHeight] = scale_copy_dst(g_gxState.texCopyDstWidth, g_gxState.texCopyDstHeight);
    const auto copy = texture::resolve_copy_texture({
        .dest = dest,
        .width = dstWidth,
        .height = dstHeight,
        .format = texCopyFmt,
    });
    auto& current = g_gxState.copyTextures[dest];
    current.width = g_gxState.texCopyDstWidth;
    current.height = g_gxState.texCopyDstHeight;
    current.pixelFormat = g_gxState.pixelFmt;
    current.frame = texture::current_frame();
    current.whole = written_size(g_gxState.texCopySrc.width) == current.width &&
                    written_size(g_gxState.texCopySrc.height) == current.height;
    target = copy.handle;
  }

  const auto clearColor = clear && g_gxState.colorUpdate;
  const auto clearAlpha = clear && g_gxState.alphaUpdate && efb_has_alpha(g_gxState.pixelFmt);
  const auto clearDepth = clear && g_gxState.depthUpdate;
  // The box filter averages 2x2 logical pixels per texel: at Aurora's render scale, that many more
  // physical ones, so that a scaled copy keeps the filter's footprint.
  u32 boxFilterTaps = 0;
  if (g_gxState.texCopyMipmap && g_gxState.texCopySrc.width > 0) {
    const float scale = static_cast<float>(rect.width) / static_cast<float>(g_gxState.texCopySrc.width);
    boxFilterTaps = std::max<u32>(2, static_cast<u32>(std::lround(2.f * scale)));
  }
  gfx::resolve_pass_into(target, rect, clearColor, clearAlpha, clearDepth, g_gxState.clearColor, clear_depth_value(),
                         texCopyFmt, g_gxState.pixelFmt, boxFilterTaps, dstRect);
}

void copy_disp(GXBool clear, bool yScale) noexcept {
  if (!g_config.emulateXfb) {
    return;
  }

  const auto& src = g_gxState.dispCopySrc;
  const auto rect = map_logical_scissor(src);
  const auto srcWidth = static_cast<u32>(std::max(src.width, 0));
  const auto srcHeight = static_cast<u32>(std::max(src.height, 0));

  // The destination's line pitch (BP 0x4D, in 32-byte units), and the lines the copy writes at its
  // Y scale (BP 0x4E, 256 over the scale), when the copy enables it.
  u32 stride = (srcWidth + 15) / 16 * 32;
  if (g_gxState.bpRegValid.test(0x4D)) {
    stride = (g_gxState.bpRegCache[0x4D] & 0x3FF) * 32;
  }
  u32 lines = srcHeight;
  if (yScale && g_gxState.bpRegValid.test(0x4E)) {
    const u32 scale = g_gxState.bpRegCache[0x4E] & 0x1FF;
    if (scale != 0 && scale != 0x100 && srcHeight != 0) {
      lines = get_num_xfb_lines(srcHeight, scale);
    }
  }

  // Render target pixels per frame buffer pixel: the EFB's scale.
  Vec2<float> scale{1.f, 1.f};
  if (srcWidth != 0 && srcHeight != 0) {
    const auto [physicalWidth, physicalHeight] = scale_copy_dst(srcWidth, srcHeight);
    scale = {static_cast<float>(physicalWidth) / static_cast<float>(srcWidth),
             static_cast<float>(physicalHeight) / static_cast<float>(srcHeight)};
  }

  const gfx::XfbCopy xfb{
      .dest = reinterpret_cast<uintptr_t>(g_gxState.texCopyDest),
      .stride = stride,
      .lines = lines,
      .fbLines = vi::configured_xfb_lines(),
      .scaled = lines != srcHeight,
      .scale = scale,
  };
  const auto clearColor = clear && g_gxState.colorUpdate;
  const auto clearAlpha = clear && g_gxState.alphaUpdate && efb_has_alpha(g_gxState.pixelFmt);
  const auto clearDepth = clear && g_gxState.depthUpdate;
  gfx::copy_disp(rect, xfb, clearColor, clearAlpha, clearDepth, g_gxState.clearColor, clear_depth_value());
}

void clear_efb(const gfx::ClipRect& logicalRect, bool wholeTarget, bool clearColor, bool clearAlpha, bool clearDepth,
               GXColor color, u32 z) noexcept {
  gfx::ClipRect rect;
  if (wholeTarget) {
    const auto [width, height] = gfx::get_render_target_size();
    rect = {0, 0, static_cast<int32_t>(width), static_cast<int32_t>(height)};
  } else {
    rect = map_logical_scissor(logicalRect);
  }
  const Vec4 clearColorValue{
      static_cast<float>(color.r) / 255.f,
      static_cast<float>(color.g) / 255.f,
      static_cast<float>(color.b) / 255.f,
      static_cast<float>(color.a) / 255.f,
  };
  gfx::clear_rect(rect, clearColor, clearAlpha && efb_has_alpha(g_gxState.pixelFmt), clearDepth, clearColorValue,
                  clear_depth_value(z));
}
} // namespace aurora::gx

extern "C" {
GXRenderModeObj GXNtsc480IntDf = {
    VI_TVMODE_NTSC_INT,
    640,
    480,
    480,
    40,
    0,
    640,
    480,
    VI_XFBMODE_DF,
    0,
    0,
    {6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6},
    {8, 8, 10, 12, 10, 8, 8},
};
GXRenderModeObj GXNtsc480Int = {
    VI_TVMODE_NTSC_INT,
    640,
    480,
    480,
    40,
    0,
    640,
    480,
    VI_XFBMODE_DF,
    0,
    0,
    {6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6},
    {0, 0, 21, 22, 21, 0, 0},
};
GXRenderModeObj GXPal528IntDf = {
    VI_TVMODE_PAL_INT,
    704,
    528,
    480,
    40,
    0,
    640,
    480,
    VI_XFBMODE_DF,
    0,
    0,
    {6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6},
    {8, 8, 10, 12, 10, 8, 8},
};
GXRenderModeObj GXMpal480IntDf = {
    VI_TVMODE_PAL_INT,
    640,
    480,
    480,
    40,
    0,
    640,
    480,
    VI_XFBMODE_DF,
    0,
    0,
    {6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6},
    {8, 8, 10, 12, 10, 8, 8},
};

void GXAdjustForOverscan(GXRenderModeObj* rmin, GXRenderModeObj* rmout, u16 hor, u16 ver) {
  const GXRenderModeObj in = *rmin;
  const u16 hor2 = hor * 2;
  const u16 ver2 = ver * 2;
  const u32 efbHeight = in.efbHeight;

  *rmout = in;
  rmout->fbWidth = in.fbWidth - hor2;
  rmout->efbHeight = efbHeight - (ver2 * efbHeight) / in.xfbHeight;

  if (in.xFBmode == VI_XFBMODE_SF && (in.viTVmode & 2) != 2) {
    rmout->xfbHeight = in.xfbHeight - ver;
  } else {
    rmout->xfbHeight = in.xfbHeight - ver2;
  }

  rmout->viWidth = in.viWidth - hor2;
  rmout->viHeight = in.viHeight - ver2;
  rmout->viXOrigin = in.viXOrigin + hor;
  rmout->viYOrigin = in.viYOrigin + ver;
}

void GXSetDispCopySrc(u16 left, u16 top, u16 wd, u16 ht) {
  __gx->cpDispHeight = ht;
  GX_WRITE_AURORA(GX_AURORA_LOAD_DISP_COPY_SRC);
  GX_WRITE_U32(left);
  GX_WRITE_U32(top);
  GX_WRITE_U32(wd);
  GX_WRITE_U32(ht);
}

void GXSetTexCopySrc(u16 left, u16 top, u16 wd, u16 ht) {
  GX_WRITE_AURORA(GX_AURORA_LOAD_COPY_SRC);
  GX_WRITE_U32(left);
  GX_WRITE_U32(top);
  GX_WRITE_U32(wd);
  GX_WRITE_U32(ht);
}

void GXSetDispCopyDst(u16 wd, u16 ht) {
  // The line pitch in 32-byte units: sixteen pixels of two bytes each.
  u32 reg = 0;
  SET_REG_FIELD(0, reg, 10, 0, wd >> 4);
  SET_REG_FIELD(0, reg, 8, 24, 0x4D);
  __gx->cpDispStride = reg;
}

void GXSetTexCopyDst(u16 wd, u16 ht, GXTexFmt fmt, GXBool mipmap) {
  GX_WRITE_AURORA(GX_AURORA_LOAD_COPY_DST);
  GX_WRITE_U32(wd);
  GX_WRITE_U32(ht);
  GX_WRITE_U32(fmt);
  GX_WRITE_U8(mipmap != GX_FALSE);
}

void GXSetDispCopyFrame2Field(GXCopyMode mode) {}

void GXSetCopyClamp(GXFBClamp clamp) {}

u32 GXSetDispCopyYScale(f32 vscale) {
  const u32 scale = static_cast<u32>(256.0f / vscale) & 0x1FF;
  u32 reg = 0;
  SET_REG_FIELD(0, reg, 9, 0, scale);
  SET_REG_FIELD(0, reg, 8, 24, 0x4E);
  GX_WRITE_RAS_REG(reg);
  __gx->cpDispYScale = scale != 0x100;
  __gx->bpSent = 1;
  return get_num_xfb_lines(__gx->cpDispHeight, scale);
}

void GXSetCopyClear(GXColor color, u32 depth) {
  // BP 0x4F: clear color R + A
  u32 reg0 = 0;
  SET_REG_FIELD(0, reg0, 8, 0, color.r);
  SET_REG_FIELD(0, reg0, 8, 8, color.a);
  SET_REG_FIELD(0, reg0, 8, 24, 0x4F);
  GX_WRITE_RAS_REG(reg0);

  // BP 0x50: clear color B + G
  u32 reg1 = 0;
  SET_REG_FIELD(0, reg1, 8, 0, color.b);
  SET_REG_FIELD(0, reg1, 8, 8, color.g);
  SET_REG_FIELD(0, reg1, 8, 24, 0x50);
  GX_WRITE_RAS_REG(reg1);

  // BP 0x51: clear Z (24-bit)
  u32 reg2 = 0;
  SET_REG_FIELD(0, reg2, 24, 0, depth);
  SET_REG_FIELD(0, reg2, 8, 24, 0x51);
  GX_WRITE_RAS_REG(reg2);
  __gx->bpSent = 1;
}

void GXSetCopyFilter(GXBool aa, const u8 sample_pattern[12][2], GXBool vf, const u8 vfilter[7]) {}

void GXSetDispCopyGamma(GXGamma gamma) {}

void GXCopyDisp(void* dest, GXBool clear) {
  if (__gx->cpDispStride != 0) {
    GX_WRITE_RAS_REG(__gx->cpDispStride);
  }
  GX_WRITE_AURORA(GX_AURORA_LOAD_COPY_DEST);
  GX_WRITE_U64(reinterpret_cast<u64>(dest));

  u32 reg = 0;
  SET_REG_FIELD(0, reg, 1, 10, __gx->cpDispYScale != 0);
  SET_REG_FIELD(0, reg, 1, 11, clear != GX_FALSE);
  SET_REG_FIELD(0, reg, 1, 14, 1);
  SET_REG_FIELD(0, reg, 8, 24, 0x52);
  GX_WRITE_RAS_REG(reg);
  aurora::gx::fifo::publish();
}

void GXCopyTex(void* dest, GXBool clear) {
  GX_WRITE_AURORA(GX_AURORA_LOAD_COPY_DEST);
  GX_WRITE_U64(reinterpret_cast<u64>(dest));

  SET_REG_FIELD(0, __gx->cpTex, 1, 11, clear != GX_FALSE);
  SET_REG_FIELD(0, __gx->cpTex, 1, 14, 0);
  SET_REG_FIELD(0, __gx->cpTex, 8, 24, 0x52);
  GX_WRITE_RAS_REG(__gx->cpTex);
  aurora::gx::fifo::publish();
}

f32 GXGetYScaleFactor(u16 efbHeight, u16 xfbHeight) {
  u32 height1 = xfbHeight;
  f32 scale1 = static_cast<f32>(xfbHeight) / static_cast<f32>(efbHeight);
  u32 height2 = get_num_xfb_lines(efbHeight, static_cast<u32>(256.0f / scale1) & 0x1FF);

  while (height2 > xfbHeight) {
    height1--;
    scale1 = static_cast<f32>(height1) / static_cast<f32>(efbHeight);
    height2 = get_num_xfb_lines(efbHeight, static_cast<u32>(256.0f / scale1) & 0x1FF);
  }

  f32 scale2 = scale1;
  while (height2 < xfbHeight) {
    scale2 = scale1;
    height1++;
    scale1 = static_cast<f32>(height1) / static_cast<f32>(efbHeight);
    height2 = get_num_xfb_lines(efbHeight, static_cast<u32>(256.0f / scale1) & 0x1FF);
  }

  return scale2;
}

u16 GXGetNumXfbLines(u16 efbHeight, f32 yScale) {
  return get_num_xfb_lines(efbHeight, static_cast<u32>(256.0f / yScale) & 0x1FF);
}
// TODO GXClearBoundingBox
// TODO GXReadBoundingBox
}
