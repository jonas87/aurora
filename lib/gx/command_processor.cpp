#include "command_processor.hpp"

#include "../gfx/recording.hpp"
#include "../internal.hpp"
#include "dolphin/gd/GDGeometry.h"
#include "dolphin/gx/GXAurora.h"
#include "gx.hpp"
#include "pipeline.hpp"
#include "readback.hpp"
#include "regs.hpp"
#include "shader_info.hpp"
#include "texture.hpp"

#include <tracy/Tracy.hpp>

#include <bit>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace aurora::gx::fifo {
namespace {
constexpr Module Log{"aurora::gx::fifo"};

// The indices are written in place, the buffer grown once per primitive.
u16 prepare_idx_buffer(ByteBuffer& buf, GXPrimitive prim, u16 vtxStart, u16 vtxCount) noexcept {
  const auto reserve = [&buf](u32 count) {
    return reinterpret_cast<u16*>(buf.append_uninitialized(count * sizeof(u16)));
  };
  if (prim == GX_QUADS) {
    const u32 count = ((u32(vtxCount) + 3) / 4) * 6;
    u16* out = reserve(count);
    for (u16 v = 0; v < vtxCount; v += 4) {
      const u16 idx0 = vtxStart + v;
      *out++ = idx0;
      *out++ = idx0 + 1;
      *out++ = idx0 + 2;
      *out++ = idx0 + 2;
      *out++ = idx0 + 3;
      *out++ = idx0;
    }
    return static_cast<u16>(count);
  }
  if (prim == GX_TRIANGLES) {
    u16* out = reserve(vtxCount);
    for (u16 v = 0; v < vtxCount; ++v) {
      out[v] = vtxStart + v;
    }
    return vtxCount;
  }
  if (prim == GX_TRIANGLEFAN || prim == GX_TRIANGLESTRIP) {
    const u32 count = vtxCount < 3 ? vtxCount : 3 + (u32(vtxCount) - 3) * 3;
    u16* out = reserve(count);
    for (u16 v = 0; v < vtxCount; ++v) {
      const u16 idx = vtxStart + v;
      if (v < 3) {
        *out++ = idx;
      } else if (prim == GX_TRIANGLEFAN) {
        *out++ = vtxStart;
        *out++ = idx - 1;
        *out++ = idx;
      } else if ((v & 1) == 0) {
        *out++ = idx - 2;
        *out++ = idx - 1;
        *out++ = idx;
      } else {
        *out++ = idx - 1;
        *out++ = idx - 2;
        *out++ = idx;
      }
    }
    return static_cast<u16>(count);
  }
  if (prim == GX_LINES || prim == GX_LINESTRIP || prim == GX_POINTS) {
    u16* out = reserve(6);
    out[0] = 0;
    out[1] = 1;
    out[2] = 3;
    out[3] = 3;
    out[4] = 2;
    out[5] = 0;
    return 6;
  }
  UNLIKELY FATAL("unsupported primitive type {}", static_cast<u32>(prim));
}

// GX FIFO opcodes - use CP_ prefix to avoid clashing with GXCommandList.h macros
constexpr u8 CP_CMD_NOP = GX_NOP;
constexpr u8 CP_CMD_LOAD_CP_REG = GX_LOAD_CP_REG;
constexpr u8 CP_CMD_LOAD_XF_REG = GX_LOAD_XF_REG;
constexpr u8 CP_CMD_LOAD_INDX_A = GX_LOAD_INDX_A;
constexpr u8 CP_CMD_LOAD_INDX_B = GX_LOAD_INDX_B;
constexpr u8 CP_CMD_LOAD_INDX_C = GX_LOAD_INDX_C;
constexpr u8 CP_CMD_LOAD_INDX_D = GX_LOAD_INDX_D;
constexpr u8 CP_CMD_CALL_DL = GX_CMD_CALL_DL;
constexpr u8 CP_CMD_INVAL_VTX = GX_CMD_INVL_VC;
constexpr u8 CP_CMD_LOAD_BP_REG = GX_LOAD_BP_REG & GX_OPCODE_MASK;

// Primitive type mask
constexpr u8 CP_OPCODE_MASK = GX_OPCODE_MASK;
constexpr u8 CP_VAT_MASK = GX_VAT_MASK;

struct FogRangeLutKey {
  std::array<u16, 10> rangeK;
  f32 rangeCenter;
  f32 renderWidth;
  u32 targetWidth;

  bool operator==(const FogRangeLutKey&) const = default;
};

struct FogRangeLutEntry {
  FogRangeLutKey key;
  std::vector<f32> factors;
};

constexpr size_t MaxFogRangeLuts = 32;
std::vector<FogRangeLutEntry> sFogRangeLuts;

struct DrawCache {
  uint64_t targetLayoutKey = 0;
  PipelineConfig config{};
  ShaderInfo shaderInfo{};
  gfx::PipelineRef pipelineRef{};
  GXBindGroups bindGroups{};
  uint64_t bindGeneration = 0;
  GXVtxFmt fmt = GX_MAX_VTXFMT;
  u8 lineMode = 0;
  bool hasPipeline = false;
  gfx::Range uniformRange{};
  // The current position matrix the uniform block was built with: a compact block holds that matrix
  // (ShaderInfo::compactMatrices).
  u32 uniformPnMtx = 0;
  gfx::Range fogRange{};
  FogRangeLutKey fogRangeKey{};
  bool hasFogRange = false;
  GXVtxFmt lastDrawFmt = GX_MAX_VTXFMT;
};
DrawCache sDrawCache;

FogRangeLutKey fog_range_lut_key() noexcept {
  const auto& state = g_gxState.fog;
  const f32 logicalWidth = std::max(g_gxState.logicalViewport.width, 1.f);
  const f32 renderWidth = std::max(g_gxState.renderViewport.width, 1.f);
  return {
      .rangeK = state.rangeK,
      .rangeCenter = ((static_cast<f32>(state.rangeCenter) - g_gxState.logicalViewport.left) / logicalWidth) * 2.f -
                     1.f + (g_gxState.renderViewport.left / renderWidth) * 2.f,
      .renderWidth = renderWidth,
      .targetWidth = gfx::get_render_target_size().x,
  };
}

std::vector<f32> build_fog_range_lut(const FogRangeLutKey& key) {
  std::array<f32, 10> rangeK;
  for (u32 i = 0; i < rangeK.size(); ++i) {
    const u32 source = (i & ~1u) | (1u - (i & 1u));
    rangeK[i] = static_cast<f32>(key.rangeK[source]) / 64.f;
  }

  std::vector<f32> lut(key.targetWidth);
  for (u32 x = 0; x < key.targetWidth; ++x) {
    const f32 screenX = ((static_cast<f32>(x) + 0.5f) / key.renderWidth) * 2.f - 1.f;
    const f32 offset = screenX - key.rangeCenter;
    const f32 rangeIndex = std::clamp(9.f - std::abs(offset) * 9.f, 0.f, 9.f);
    const u32 lower = static_cast<u32>(rangeIndex);
    const u32 upper = std::min(lower + 1, 9u);
    const f32 fraction = rangeIndex - static_cast<f32>(lower);
    const f32 k = std::max(rangeK[lower] * (1.f - fraction) + rangeK[upper] * fraction, 0.000001f);
    lut[x] = std::sqrt(offset * offset + k * k) / k;
  }
  return lut;
}

const std::vector<f32>& resolve_fog_range_lut(const FogRangeLutKey& key) {
  for (const auto& entry : sFogRangeLuts) {
    if (entry.key == key) {
      return entry.factors;
    }
  }
  if (sFogRangeLuts.size() == MaxFogRangeLuts) {
    sFogRangeLuts.erase(sFogRangeLuts.begin());
  }
  sFogRangeLuts.emplace_back(FogRangeLutEntry{key, build_fog_range_lut(key)});
  return sFogRangeLuts.back().factors;
}

gfx::Range push_fog_range_lut(const FogRangeLutKey& key) {
  const auto& lut = resolve_fog_range_lut(key);
  return gfx::push_storage(reinterpret_cast<const u8*>(lut.data()), lut.size() * sizeof(f32));
}

u8 line_mode_for_prim(GXPrimitive prim) noexcept {
  switch (prim) {
  case GX_LINES:
    return 1;
  case GX_LINESTRIP:
    return 2;
  case GX_POINTS:
    return 3;
  default:
    return 0;
  }
}
} // namespace

static void handle_draw(u8 cmd, ByteReader& reader) noexcept;
static void handle_aurora(ByteReader& reader) noexcept;

ProcessResult process(const u8* data, u32 size) noexcept {
  ZoneScoped;
  ByteReader reader{{data, size}};

  while (!reader.empty()) {
    const u8 cmd = reader.read<u8>();
    u8 opcode = cmd & CP_OPCODE_MASK;

    switch (opcode) {
    case CP_CMD_NOP:
      continue;

    case CP_CMD_LOAD_BP_REG: {
      const u32 value = reader.read<u32>();
      handle_bp(value);
      if (reg_get(value, 8, 24) == GX_BP_REG_DRAWDONE) {
        return {static_cast<u32>(reader.offset()), true};
      }
      break;
    }

    case CP_CMD_LOAD_CP_REG: {
      const u8 addr = reader.read<u8>();
      handle_cp(addr, reader.read<u32>());
      break;
    }

    case CP_CMD_LOAD_XF_REG: {
      const u32 header = reader.read<u32>();
      const u32 count = ((header >> 16) & 0xFFFF) + 1;
      const u16 addr = header & 0xFFFF;
      handle_xf(addr, reader.take(count * sizeof(u32)));
      break;
    }

    case CP_CMD_LOAD_INDX_A:
    case CP_CMD_LOAD_INDX_B:
    case CP_CMD_LOAD_INDX_C:
    case CP_CMD_LOAD_INDX_D: {
      ZoneScopedN("LOAD_INDX");
      const u32 arrayType = GX_POS_MTX_ARRAY + (opcode - CP_CMD_LOAD_INDX_A) / 0x08;
      const u16 srcArrayIdx = reader.read<u16>();
      const u16 addrLen = reader.read<u16>();

      const u16 len = (addrLen >> 12) + 1;
      const u16 dstAddr = addrLen & 0x0FFF;
      auto const& array = g_gxState.arrays[arrayType];
      const u32 srcOffset = static_cast<u32>(srcArrayIdx) * array.stride;
      const u32 srcSize = static_cast<u32>(len) * sizeof(u32);
      AURORA_ASSERT(array.data != nullptr, "indexed XF load from unmapped array {}", arrayType);
      AURORA_ASSERT(srcOffset <= array.size && srcSize <= array.size - srcOffset,
                    "indexed XF load outside array {}: offset={}, size={}, array size={}", arrayType, srcOffset,
                    srcSize, array.size);
      auto const* srcData = static_cast<const u8*>(array.data) + srcOffset;
      if (!copy_xf_data(dstAddr, srcData, len, array.le ? std::endian::little : std::endian::big)) {
#ifndef NDEBUG
        Log.debug("Unimplemented indexed XF load (opcode 0x{:02X}, dstAddr=%04x)", opcode, dstAddr);
#endif
      }
      break;
    }

    case CP_CMD_CALL_DL: {
      // Call display list: 8 bytes (address + size)
      Log.warn("Ignoring nested GX_CMD_CALL_DL");
      reader.skip(8);
      break;
    }

    case CP_CMD_INVAL_VTX: {
      for (auto& array : g_gxState.arrays) {
        array.cachedRange = {};
      }
      g_gxState.dirty |= DirtyImmediates;
      break;
    }

    case GX_AURORA: {
      handle_aurora(reader);
      break;
    }

    // Draw commands: 0x80-0xBF
    case GX_DRAW_QUADS:
    case GX_DRAW_TRIANGLES:
    case GX_DRAW_TRIANGLE_STRIP:
    case GX_DRAW_TRIANGLE_FAN:
    case GX_DRAW_LINES:
    case GX_DRAW_LINE_STRIP:
    case GX_DRAW_POINTS: {
      handle_draw(cmd, reader);
      break;
    }

    default:
      // Check if it's a draw command (0x80-0xBF range)
      if (cmd >= 0x80) {
        handle_draw(cmd, reader);
      } else {
        // Hex dump surrounding bytes for debugging
        {
          const size_t pos = reader.offset();
          size_t dumpStart = (pos > 17) ? pos - 17 : 0;
          size_t dumpEnd = (pos + 16 < size) ? pos + 16 : size;
          std::string hex;
          for (size_t i = dumpStart; i < dumpEnd; i++) {
            if (i == pos - 1)
              hex += fmt::format("[{:02x}]", data[i]);
            else
              hex += fmt::format(" {:02x}", data[i]);
          }
          Log.error("  hex dump (pos {}-{}):{}", dumpStart, dumpEnd - 1, hex);
        }
        FATAL("command_processor: unknown opcode 0x{:02X} at pos {}", cmd, reader.offset() - 1);
      }
      break;
    }
  }
  return {size, false};
}

[[noreturn]] static void handle_draw_overrun(size_t totalVtxBytes, const ByteReader& reader) noexcept {
  // Hex dump around the draw command for debugging
  const size_t pos = reader.offset();
  const size_t size = reader.size();
  const u8* data = reader.data();
  size_t cmdPos = pos - 2 - 1; // opcode byte position (before vtxCount and pos++)
  size_t dumpStart = (cmdPos > 16) ? cmdPos - 16 : 0;
  size_t dumpEnd = (cmdPos + 32 < size) ? cmdPos + 32 : size;
  std::string hex;
  for (size_t i = dumpStart; i < dumpEnd; i++) {
    if (i == cmdPos)
      hex += fmt::format("[{:02x}]", data[i]);
    else
      hex += fmt::format(" {:02x}", data[i]);
  }
  Log.error("  hex dump around draw cmd (pos {}-{}):{}", dumpStart, dumpEnd - 1, hex);
  FATAL("draw vertex data overrun: need {} bytes at pos {}, have {}", totalVtxBytes, pos, reader.remaining());
}

static u32 calc_vtx_size(GXVtxFmt fmt) noexcept {
  u32 vtxSize = 0;
  const auto& vtxFmt = g_gxState.vtxFmts[fmt];
  for (int i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
    const auto& attrFmt = vtxFmt.attrs[i];
    switch (g_gxState.vtxDesc[i]) {
    case GX_NONE:
      break;
    case GX_DIRECT: {
      const auto attr = static_cast<GXAttr>(i);
      vtxSize += comp_type_size(attr, attrFmt.type) * comp_cnt_count(attr, attrFmt.cnt);
      break;
    }
    case GX_INDEX8:
      vtxSize += i == GX_VA_NRM && attrFmt.cnt == GX_NRM_NBT3 ? 3 : 1;
      break;
    case GX_INDEX16:
      vtxSize += i == GX_VA_NRM && attrFmt.cnt == GX_NRM_NBT3 ? 6 : 2;
      break;
    }
  }
  g_gxState.lastVtxFmt = fmt;
  g_gxState.lastVtxSize = vtxSize;
  return vtxSize;
}

static void push_gx_draw(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, gfx::Range vertRange, gfx::Range idxRange,
                         u32 numIndices) noexcept {
  auto& state = g_gxState;
  auto& cache = sDrawCache;

  DrawImmediateData immediates{.vtxStart = vertRange.offset, .currentPnMtx = state.currentPnMtx};
  for (int i = GX_VA_POS; i <= GX_VA_TEX7; ++i) {
    if (state.vtxDesc[i] != GX_INDEX8 && state.vtxDesc[i] != GX_INDEX16) {
      continue;
    }
    auto& array = state.arrays[i];
    if (array.cachedRange.size == 0) {
      array.cachedRange = gfx::push_cached_storage(static_cast<const uint8_t*>(array.data), array.size);
    }
    immediates.arrayStart[i - GX_VA_POS] = array.cachedRange.offset;
  }

  const u8 lineMode = line_mode_for_prim(prim);
  const auto targetLayoutKey = gfx::get_render_target_layout().key;
  const bool pipelineValid = cache.hasPipeline && (state.dirty & DirtyPipeline) == 0 && cache.fmt == fmt &&
                             cache.lineMode == lineMode && cache.targetLayoutKey == targetLayoutKey;
  if (!pipelineValid) {
    const bool hadPipeline = cache.hasPipeline;
    const auto prevSampledTextures = cache.shaderInfo.sampledTextures;
    const auto prevSampledIndTextures = cache.shaderInfo.sampledIndTextures;
    populate_pipeline_config(cache.config, prim, fmt);
    cache.shaderInfo = build_shader_info(cache.config.shaderConfig);
#ifndef NDEBUG
    validate_shader_config(cache.config.shaderConfig, cache.shaderInfo, g_gxState.numTexGens);
#endif
    cache.pipelineRef = gfx::pipeline_ref(cache.config);
    cache.targetLayoutKey = targetLayoutKey;
    cache.fmt = fmt;
    cache.lineMode = lineMode;
    cache.hasPipeline = true;
    state.dirty = (state.dirty & ~DirtyPipeline) | DirtyUniform;
    if (!hadPipeline || prevSampledTextures != cache.shaderInfo.sampledTextures ||
        prevSampledIndTextures != cache.shaderInfo.sampledIndTextures) {
      cache.bindGeneration = 0;
    }
  }

  const bool bindGroupsValid =
      (state.dirty & DirtyTextures) == 0 && cache.bindGeneration == texture::current_bind_generation();
  if (!bindGroupsValid) {
    const auto prevBindGroup = cache.bindGroups.textureBindGroup;
    resolve_sampled_textures(cache.shaderInfo);
    cache.bindGroups = build_bind_groups(cache.shaderInfo);
    cache.bindGeneration = texture::current_bind_generation();
    state.dirty &= ~DirtyTextures;
    // For texture_size_bias uniform
    if (cache.bindGroups.textureBindGroup != prevBindGroup) {
      state.dirty |= DirtyUniform;
    }
  }

  const bool uniformValid = (state.dirty & DirtyUniform) == 0 && cache.uniformRange.size != 0 &&
                            (!cache.shaderInfo.compactMatrices || cache.uniformPnMtx == state.currentPnMtx);
  if (!uniformValid) {
    cache.uniformRange = build_uniform(cache.shaderInfo);
    cache.uniformPnMtx = state.currentPnMtx;
    state.dirty &= ~DirtyUniform;
  }
  if (cache.config.shaderConfig.fogRangeEnabled) {
    const auto key = fog_range_lut_key();
    if (!cache.hasFogRange || cache.fogRangeKey != key) {
      cache.fogRange = push_fog_range_lut(key);
      cache.fogRangeKey = key;
      cache.hasFogRange = true;
    }
  }
  immediates.fogRangeBase = cache.fogRange.offset / sizeof(u32);

  state.dirty &= ~DirtyImmediates;

  uint32_t instanceCount = 1;
  if (prim == GX_LINES) {
    instanceCount = vtxCount / 2;
  } else if (prim == GX_LINESTRIP) {
    instanceCount = vtxCount - 1;
  } else if (prim == GX_POINTS) {
    instanceCount = vtxCount;
  }
  cache.lastDrawFmt = fmt;
  gfx::push_draw_command(DrawData{
      .pipeline = cache.pipelineRef,
      .bindGroups = cache.bindGroups,
      .vertRange = vertRange,
      .idxRange = idxRange,
      .uniformRange = cache.uniformRange,
      .immediateData = immediates,
      .vtxCount = vtxCount,
      .indexCount = numIndices,
      .instanceCount = instanceCount,
      .dstAlpha = state.dstAlpha,
  });
}

static void handle_draw_unmerged(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, gfx::Range vertRange) noexcept {
  ZoneScoped;
  u32 numIndices = 0;
  gfx::Range idxRange;

  if (prim != GX_TRIANGLES) {
    ZoneScopedN("build idx buffer");
    static ByteBuffer idxBuf;
    numIndices = prepare_idx_buffer(idxBuf, prim, 0, vtxCount);
    idxRange = gfx::push_indices(idxBuf.data(), idxBuf.size(), 4);
    idxBuf.clear();
  }

  push_gx_draw(prim, fmt, vtxCount, vertRange, idxRange, numIndices);
}

// Truncate vertex count to a valid number for the given primitive type
static constexpr u16 trunc_vtx_count(GXPrimitive prim, u16 vtxCount) noexcept {
  switch (prim) {
  case GX_QUADS:
    return vtxCount & ~3;
  case GX_TRIANGLES:
    return vtxCount - vtxCount % 3;
  case GX_TRIANGLESTRIP:
  case GX_TRIANGLEFAN:
    return vtxCount < 3 ? 0 : vtxCount;
  case GX_LINES:
    return vtxCount & ~1;
  case GX_LINESTRIP:
    return vtxCount < 2 ? 0 : vtxCount;
  default:
    return vtxCount;
  }
}

// One component of a vertex attribute, big-endian as the FIFO carries it, or little-endian from an
// array the game marks as native (le).
static float read_vertex_component(const u8* data, GXCompType type, u8 frac, bool le = false) noexcept {
  const auto byte = [&](u32 index, u32 size) { return static_cast<u32>(data[le ? size - 1 - index : index]); };
  float value = 0.f;
  switch (type) {
  case GX_U8:
    value = static_cast<float>(data[0]);
    break;
  case GX_S8:
    value = static_cast<float>(static_cast<s8>(data[0]));
    break;
  case GX_U16:
    value = static_cast<float>(static_cast<u16>((byte(0, 2) << 8) | byte(1, 2)));
    break;
  case GX_S16:
    value = static_cast<float>(static_cast<s16>((byte(0, 2) << 8) | byte(1, 2)));
    break;
  case GX_F32: {
    const u32 bits = (byte(0, 4) << 24) | (byte(1, 4) << 16) | (byte(2, 4) << 8) | byte(3, 4);
    return std::bit_cast<float>(bits);
  }
  default:
    return 0.f;
  }
  return value / static_cast<float>(1u << frac);
}

// ZFreeze: latches the depth plane of a triangle culled by GX_CULL_ALL from its first three
// vertices, which the co-planar triangles that follow take (GXSetCoPlanar). The positions are
// transformed as the vertex shader transforms them, by the current position matrix and the
// projection as uploaded, and the plane is taken in normalised device coordinates. Only direct
// positions are read, after any direct matrix indices; other layouts leave the plane as it was.
// The hardware, as Dolphin follows it, latches the plane from every triangle while the co-planar
// flag is off and from a draw's last triangle; this takes it only from culled triangles, the
// reference triangle games draw for the purpose, and from their first.
static void latch_coplanar_plane(GXVtxFmt fmt, std::span<const u8> vertexData, u32 vtxSize, u16 vtxCount) noexcept {
  if (vtxCount < 3 || g_gxState.vtxDesc[GX_VA_POS] != GX_DIRECT) {
    return;
  }
  u32 offset = 0;
  for (u32 attr = GX_VA_PNMTXIDX; attr <= GX_VA_TEX7MTXIDX; ++attr) {
    if (g_gxState.vtxDesc[attr] == GX_DIRECT) {
      ++offset;
    }
  }
  const auto& attrFmt = g_gxState.vtxFmts[fmt].attrs[GX_VA_POS];
  u32 size = 0;
  switch (attrFmt.type) {
  case GX_U8:
  case GX_S8:
    size = 1;
    break;
  case GX_U16:
  case GX_S16:
    size = 2;
    break;
  case GX_F32:
    size = 4;
    break;
  default:
    return;
  }
  const u32 count = attrFmt.cnt == GX_POS_XYZ ? 3 : 2;
  if (offset + count * size > vtxSize) {
    return;
  }

  const u32 matrixIndex = g_gxState.currentPnMtx;
  Mat3x4<float> matrix{Vec4{1.f, 0.f, 0.f, 0.f}, Vec4{0.f, 1.f, 0.f, 0.f}, Vec4{0.f, 0.f, 1.f, 0.f}};
  if (matrixIndex < MaxPnMtx) {
    matrix = g_gxState.pnMtx[matrixIndex].pos;
  }
  const auto proj = uploaded_projection();

  float x[3];
  float y[3];
  float z[3];
  for (u32 v = 0; v < 3; ++v) {
    const u8* data = vertexData.data() + v * vtxSize + offset;
    float pos[3] = {0.f, 0.f, 0.f};
    for (u32 k = 0; k < count; ++k) {
      pos[k] = read_vertex_component(data + k * size, attrFmt.type, attrFmt.frac);
    }
    const Vec4<float>* rows[3] = {&matrix.m0, &matrix.m1, &matrix.m2};
    float view[4];
    for (u32 r = 0; r < 3; ++r) {
      const auto& row = *rows[r];
      view[r] = row[0] * pos[0] + row[1] * pos[1] + row[2] * pos[2] + row[3];
    }
    view[3] = 1.f;
    float clip[4];
    for (u32 r = 0; r < 4; ++r) {
      const auto& row = proj[r];
      clip[r] = row[0] * view[0] + row[1] * view[1] + row[2] * view[2] + row[3] * view[3];
    }
    if (clip[3] == 0.f) {
      return;
    }
    x[v] = clip[0] / clip[3];
    y[v] = clip[1] / clip[3];
    z[v] = clip[2] / clip[3];
  }

  // z = a * x + b * y + c through the three points; a constant depth if they are collinear on
  // screen.
  const float dx1 = x[1] - x[0];
  const float dy1 = y[1] - y[0];
  const float dz1 = z[1] - z[0];
  const float dx2 = x[2] - x[0];
  const float dy2 = y[2] - y[0];
  const float dz2 = z[2] - z[0];
  const float det = dx1 * dy2 - dx2 * dy1;
  float a = 0.f;
  float b = 0.f;
  if (std::abs(det) > 1e-12f) {
    a = (dz1 * dy2 - dz2 * dy1) / det;
    b = (dx1 * dz2 - dx2 * dz1) / det;
  }
  const float c = z[0] - a * x[0] - b * y[0];
  g_gxState.coplanarPlane = Vec4{a, b, c, 0.f};
  g_gxState.coplanarPlaneValid = true;
  g_gxState.dirty |= DirtyPipeline | DirtyUniform;
}

// The transform unit keeps the tangent and binormal of the last vertex sent with NBT normals, and
// a vertex without them reads the kept ones: games send a culled triangle with GX_NRM_NBT normals
// to set them for the draws that follow. Latches the last vertex's tangent and binormal, direct or
// indexed, addressed and read as the vertex shader reads them (attr_load_nbt_slice), culled or
// not.
static void latch_nbt(GXVtxFmt fmt, std::span<const u8> vertexData, u32 vtxSize, u16 vtxCount) noexcept {
  const auto& state = g_gxState;
  const auto& attrFmt = state.vtxFmts[fmt].attrs[GX_VA_NRM];
  const auto type = state.vtxDesc[GX_VA_NRM];
  if (vtxCount == 0 || type == GX_NONE || (attrFmt.cnt != GX_NRM_NBT && attrFmt.cnt != GX_NRM_NBT3)) {
    return;
  }
  const bool nbt3 = attrFmt.cnt == GX_NRM_NBT3;

  // The normal's offset in the vertex: the attributes before it, as calc_vtx_size counts them.
  u32 offset = 0;
  for (int i = GX_VA_PNMTXIDX; i < GX_VA_NRM; ++i) {
    const auto attr = static_cast<GXAttr>(i);
    const auto& fmtAttr = state.vtxFmts[fmt].attrs[i];
    switch (state.vtxDesc[i]) {
    case GX_DIRECT:
      offset += comp_type_size(attr, fmtAttr.type) * comp_cnt_count(attr, fmtAttr.cnt);
      break;
    case GX_INDEX8:
      offset += 1;
      break;
    case GX_INDEX16:
      offset += 2;
      break;
    default:
      break;
    }
  }

  const u32 compSize = comp_type_size(GX_VA_NRM, attrFmt.type);
  const u32 indexSize = type == GX_INDEX8 ? 1 : 2;
  const u32 attrSize = type == GX_DIRECT ? 9 * compSize : (nbt3 ? 3 * indexSize : indexSize);
  if (offset + attrSize > vtxSize) {
    return;
  }
  const u8* vertex = vertexData.data() + static_cast<size_t>(vtxCount - 1) * vtxSize + offset;
  const auto& array = state.arrays[GX_VA_NRM];

  // Slice 1 is the tangent, slice 2 the binormal, in the order the vertex data holds them.
  Vec4<float> slices[2]{};
  for (u32 slice = 1; slice <= 2; ++slice) {
    const u8* data = nullptr;
    bool le = false;
    if (type == GX_DIRECT) {
      data = vertex + slice * 3 * compSize;
    } else {
      const u8* indexData = vertex + (nbt3 ? slice * indexSize : 0);
      const u32 index = type == GX_INDEX8 ? indexData[0] : (static_cast<u32>(indexData[0]) << 8) | indexData[1];
      const size_t start = static_cast<size_t>(index) * array.stride + slice * 3 * compSize;
      if (array.data == nullptr || start + 3 * compSize > array.size) {
        return;
      }
      data = static_cast<const u8*>(array.data) + start;
      le = array.le;
    }
    for (u32 k = 0; k < 3; ++k) {
      slices[slice - 1][k] = read_vertex_component(data + k * compSize, attrFmt.type, attrFmt.frac, le);
    }
  }
  if (slices[0] != state.latchedTangent || slices[1] != state.latchedBinormal) {
    g_gxState.latchedTangent = slices[0];
    g_gxState.latchedBinormal = slices[1];
    // Only a pipeline that reads the latched vectors needs its uniform again: a draw with another
    // pipeline builds its own, and every draw with NBT normals changes them.
    if (sDrawCache.hasPipeline && sDrawCache.shaderInfo.latchedNbt) {
      g_gxState.dirty |= DirtyUniform;
    }
  }
}

static void draw_prim(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, ByteReader& reader) noexcept {
  ZoneScoped;
  u32 vtxSize;
  if (g_gxState.lastVtxFmt == fmt)
    LIKELY { vtxSize = g_gxState.lastVtxSize; }
  else
    UNLIKELY { vtxSize = calc_vtx_size(fmt); }

  u32 totalVtxBytes = vtxCount * vtxSize;
  if (totalVtxBytes > reader.remaining())
    UNLIKELY { handle_draw_overrun(totalVtxBytes, reader); }
  const auto vertexData = reader.take(totalVtxBytes);
  latch_nbt(fmt, vertexData, vtxSize, vtxCount);

  vtxCount = trunc_vtx_count(prim, vtxCount);
  if (vtxCount == 0)
    UNLIKELY { return; }

  // GX_CULL_ALL draws no triangle; the first one's depth plane is latched for co-planar draws.
  if (g_gxState.cullMode == GX_CULL_ALL && prim != GX_LINES && prim != GX_LINESTRIP && prim != GX_POINTS)
    UNLIKELY {
      latch_coplanar_plane(fmt, vertexData, vtxSize, vtxCount);
      return;
    }

  const bool cleanState = g_gxState.dirty == 0 && fmt == sDrawCache.lastDrawFmt && sDrawCache.lineMode == 0 &&
                          prim != GX_LINES && prim != GX_LINESTRIP && prim != GX_POINTS;
  auto* lastDraw = cleanState ? gfx::get_last_draw_command<DrawData>() : nullptr;
  const bool canMerge = lastDraw != nullptr && lastDraw->instanceCount == 1;

  // Push raw vertex data to buffer. Merged draws must remain contiguous with the previous range.
  gfx::Range vertRange = gfx::push_verts(vertexData.data(), vtxCount * vtxSize, canMerge ? 0 : 4);

  // Try to merge with previous draw call
  if (canMerge) {
    u32 numIndices = 0;
    gfx::Range idxRange;
    static ByteBuffer idxBuf;
    const bool hadIndexRange = lastDraw->idxRange.size != 0;
    if (lastDraw->indexCount == 0 && prim != GX_TRIANGLES) {
      // Generate triangle index buffer for previous draw
      lastDraw->indexCount = prepare_idx_buffer(idxBuf, GX_TRIANGLES, 0, lastDraw->vtxCount);
    }
    if (lastDraw->indexCount != 0) {
      numIndices += prepare_idx_buffer(idxBuf, prim, lastDraw->vtxCount, vtxCount);
      idxRange = gfx::push_indices(idxBuf.data(), idxBuf.size(), hadIndexRange ? 0 : 4);
      idxBuf.clear();
    }
    CHECK(lastDraw->vertRange.offset + lastDraw->vertRange.size == vertRange.offset,
          "Non-consecutive vertex ranges ({} < {})", lastDraw->vertRange.offset + lastDraw->vertRange.size,
          vertRange.offset);
    if (hadIndexRange) {
      CHECK(lastDraw->idxRange.offset + lastDraw->idxRange.size == idxRange.offset,
            "Non-consecutive index ranges ({} < {})", lastDraw->idxRange.offset + lastDraw->idxRange.size,
            idxRange.offset);
    }
    lastDraw->vertRange.size += vertRange.size;
    if (lastDraw->idxRange.size == 0) {
      lastDraw->idxRange = idxRange;
    } else {
      lastDraw->idxRange.size += idxRange.size;
    }
    lastDraw->vtxCount += vtxCount;
    lastDraw->indexCount += numIndices;
    gfx::detail::increment_merged_draw_count();
    return;
  }

  handle_draw_unmerged(prim, fmt, vtxCount, vertRange);
}

static void handle_draw(u8 cmd, ByteReader& reader) noexcept {
  const auto fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
  const auto prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
  draw_prim(prim, fmt, reader.read<u16>(), reader);
}

void handle_aurora(ByteReader& reader) noexcept {
  ZoneScoped;
  const u16 subCmd = reader.read<u16>();

  if (subCmd == GX_AURORA_LOAD_VIEWPORT_RENDER) {
    const f32 left = reader.read<f32>();
    const f32 top = reader.read<f32>();
    const f32 width = reader.read<f32>();
    const f32 height = reader.read<f32>();
    const f32 nearZ = reader.read<f32>();
    const f32 farZ = reader.read<f32>();
    set_render_viewport({
        .left = left,
        .top = top,
        .width = width,
        .height = height,
        .znear = nearZ,
        .zfar = farZ,
    });
  } else if (subCmd == GX_AURORA_LOAD_SCISSOR_RENDER) {
    const s32 left = reader.read<s32>();
    const s32 top = reader.read<s32>();
    const s32 width = reader.read<s32>();
    const s32 height = reader.read<s32>();
    set_render_scissor({left, top, width, height});
  } else if (subCmd == GX_AURORA_LOAD_PROJECTION_FULL) {
    auto& proj = g_gxState.proj;
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        proj[r][c] = reader.read<f32>();
      }
    }
    // Invalidate projection XF regs
    for (u32 reg = 0x20; reg <= 0x26; ++reg) {
      g_gxState.xfRegValid.reset(reg);
    }
    g_gxState.dirty |= DirtyUniform;
  } else if (subCmd >= GX_AURORA_LOAD_ARRAYBASE && subCmd <= (GX_AURORA_LOAD_ARRAYBASE | 0x0f)) {
    const u32 attrIdx = subCmd - GX_AURORA_LOAD_ARRAYBASE + GX_VA_POS;
    const u64 arrayAddr = reader.read<u64>();
    const u32 arraySize = reader.read<u32>();
    const bool le = reader.read<u8>() == 1;

    auto& array = g_gxState.arrays[attrIdx];
    const auto newData = reinterpret_cast<void*>(arrayAddr);
    if (array.data != newData || array.size != arraySize || array.le != le) {
      if (array.le != le) {
        // Endianness is baked into the shader
        g_gxState.dirty |= DirtyPipeline;
      }
      array.data = newData;
      array.size = arraySize;
      array.le = le;
      array.cachedRange = {};
      g_gxState.dirty |= DirtyImmediates;
    }
  } else if (subCmd == GX_AURORA_LOAD_TEXOBJ) {
    const auto texMapId = reader.read<u8>();
    CHECK(texMapId < MaxTextures, "invalid texture map id {}", texMapId);
    auto& slot = g_gxState.loadedTextures[texMapId];
    const auto newData = reinterpret_cast<const void*>(reader.read<u64>());
    const u32 newWidth = reader.read<u32>();
    const u32 newHeight = reader.read<u32>();
    const auto newFormat = static_cast<GXTexFmt>(reader.read<u32>());
    const auto newTlut = static_cast<GXTlut>(reader.read<u32>());
    u8 newFlags = slot.flags & ~0x80u; // Reset no-cache flag
    const u8 loadFlags = reader.read<u8>();
    if ((loadFlags & 1u) != 0) {
      newFlags |= 1u;
    } else {
      newFlags &= ~1u;
    }
    if ((loadFlags & 2u) != 0) {
      newFlags |= 0x40u;
    } else {
      newFlags &= ~0x40u;
    }
    const u32 newTexObjId = reader.read<u32>();
    const u32 newTexDataVersion = reader.read<u32>();
    if (slot.data != newData || slot.mWidth != newWidth || slot.mHeight != newHeight ||
        slot.mFormat != static_cast<u32>(newFormat) || slot.tlut != newTlut || slot.flags != newFlags ||
        slot.texObjId != newTexObjId || slot.texDataVersion != newTexDataVersion) {
      slot.data = newData;
      slot.mWidth = newWidth;
      slot.mHeight = newHeight;
      slot.mFormat = newFormat;
      slot.tlut = newTlut;
      slot.flags = newFlags;
      slot.texObjId = newTexObjId;
      slot.texDataVersion = newTexDataVersion;
      g_gxState.dirty |= DirtyTextures;
    }
  } else if (subCmd == GX_AURORA_SET_TEX_IMG_OBJ) {
    const auto texMapId = reader.read<u8>();
    CHECK(texMapId < MaxTextures, "invalid texture map id {}", texMapId);
    auto& slot = g_gxState.loadedTextures[texMapId];
    const auto newData = reinterpret_cast<const void*>(reader.read<u64>());
    const u32 objWidth = reader.read<u32>();
    const u32 objHeight = reader.read<u32>();
    const u32 objFormat = reader.read<u32>();
    const u32 objTexObjId = reader.read<u32>();
    const u32 objTexDataVersion = reader.read<u32>();
    const bool objStreaming = (reader.read<u8>() & 1u) != 0;
    // The map keeps its size and format, as the image register leaves them: the object names the image only when
    // they are its own, and otherwise the image goes without an identity, as GX_AURORA_SET_TEX_IMG_PTR leaves it.
    // The no-cache mark is cleared and the streamed mark is the object's, as a load sets them.
    const bool sameKind = slot.width() == objWidth && slot.height() == objHeight && slot.format() == objFormat;
    const u32 newTexObjId = sameKind ? objTexObjId : 0;
    const u32 newTexDataVersion = sameKind ? objTexDataVersion : 0;
    const u8 newFlags = static_cast<u8>((slot.flags & ~0xC0u) | (sameKind && objStreaming ? 0x40u : 0u));
    if (slot.data != newData || slot.texObjId != newTexObjId || slot.texDataVersion != newTexDataVersion ||
        slot.flags != newFlags) {
      slot.data = newData;
      slot.texObjId = newTexObjId;
      slot.texDataVersion = newTexDataVersion;
      slot.flags = newFlags;
      g_gxState.dirty |= DirtyTextures;
    }
  } else if (subCmd == GX_AURORA_SET_TEX_IMG_PTR) {
    const auto texMapId = reader.read<u8>();
    CHECK(texMapId < MaxTextures, "invalid texture map id {}", texMapId);
    auto& slot = g_gxState.loadedTextures[texMapId];
    const auto newData = reinterpret_cast<const void*>(reader.read<u64>());
    if (slot.data != newData) {
      slot.data = newData;
      slot.texObjId = 0;
      slot.texDataVersion = 0;
      g_gxState.dirty |= DirtyTextures;
    }
  } else if (subCmd == GX_AURORA_LOAD_TLUT) {
    const auto idx = reader.read<u8>();
    CHECK(idx < MaxTluts, "invalid tlut slot {}", idx);
    auto& slot = g_gxState.loadedTluts[idx];
    const auto newData = reinterpret_cast<const void*>(reader.read<u64>());
    const auto newFormat = static_cast<GXTlutFmt>(reader.read<u32>());
    const u16 newNumEntries = reader.read<u16>();
    const u32 newTlutObjId = reader.read<u32>();
    const u32 newTlutDataVersion = reader.read<u32>();
    const u8 newFlags = slot.flags & ~0x80u; // Reset no-cache flag
    if (slot.data != newData || slot.format != newFormat || slot.numEntries != newNumEntries ||
        slot.tlutObjId != newTlutObjId || slot.tlutDataVersion != newTlutDataVersion || slot.flags != newFlags) {
      if (slot.tlutObjId != newTlutObjId || slot.tlutDataVersion != newTlutDataVersion) {
        texture::invalidate_bindings();
      }
      slot.data = newData;
      slot.format = newFormat;
      slot.numEntries = newNumEntries;
      slot.tlutObjId = newTlutObjId;
      slot.tlutDataVersion = newTlutDataVersion;
      slot.flags = newFlags;
      g_gxState.dirty |= DirtyTextures;
    }
  } else if (subCmd == GX2_SET_POLYGON_OFFSET) {
    const f32 frontOffset = reader.read<f32>();
    const f32 frontScale = reader.read<f32>();
    const f32 backOffset = reader.read<f32>();
    const f32 backScale = reader.read<f32>();
    const f32 clamp = reader.read<f32>();
    if (g_gxState.frontOffset != frontOffset || g_gxState.frontScale != frontScale ||
        g_gxState.backOffset != backOffset || g_gxState.backScale != backScale || g_gxState.clamp != clamp) {
      g_gxState.frontOffset = frontOffset;
      g_gxState.frontScale = frontScale;
      g_gxState.backOffset = backOffset;
      g_gxState.backScale = backScale;
      g_gxState.clamp = clamp;
      g_gxState.dirty |= DirtyPipeline;
    }
  } else if (subCmd == GX_AURORA_SET_DRAWS_BLOCKING) {
    const bool blocking = reader.read<u8>() != 0;
    if (g_gxState.drawsBlocking != blocking) {
      g_gxState.drawsBlocking = blocking;
      g_gxState.dirty |= DirtyPipeline;
    }
  } else if (subCmd == GX_AURORA_LOAD_COPY_SRC) {
    const s32 left = reader.read<s32>();
    const s32 top = reader.read<s32>();
    const s32 width = reader.read<s32>();
    const s32 height = reader.read<s32>();
    g_gxState.texCopySrc = {left, top, width, height};
  } else if (subCmd == GX_AURORA_LOAD_COPY_DST) {
    g_gxState.texCopyDstWidth = reader.read<u32>();
    g_gxState.texCopyDstHeight = reader.read<u32>();
    g_gxState.texCopyFmt = static_cast<GXTexFmt>(reader.read<u32>());
    g_gxState.texCopyMipmap = reader.read<u8>() != 0;
    g_gxState.texCopyDstWide = true;
  } else if (subCmd == GX_AURORA_LOAD_COPY_DEST) {
    g_gxState.texCopyDest = reinterpret_cast<const void*>(reader.read<u64>());
  } else if (subCmd == GX_AURORA_LOAD_DISP_COPY_SRC) {
    const s32 left = reader.read<s32>();
    const s32 top = reader.read<s32>();
    const s32 width = reader.read<s32>();
    const s32 height = reader.read<s32>();
    g_gxState.dispCopySrc = {left, top, width, height};
  } else if (subCmd == GX_AURORA_READBACK_CAPTURE_PEEKS) {
    readback::handle_capture_peeks(reader);
  } else if (subCmd == GX_AURORA_READBACK_CAPTURE_COPIES) {
    readback::handle_capture_copies(reader);
  } else if (subCmd == GX_AURORA_BEGIN_OFFSCREEN) {
    const u32 width = reader.read<u32>();
    const u32 height = reader.read<u32>();
    gfx::begin_offscreen(width, height);
  } else if (subCmd == GX_AURORA_BEGIN_SCALED_OFFSCREEN) {
    const u32 width = reader.read<u32>();
    const u32 height = reader.read<u32>();
    gfx::begin_offscreen(width, height, true);
  } else if (subCmd == GX_AURORA_END_OFFSCREEN) {
    gfx::end_offscreen();
  } else if (subCmd == GX_AURORA_DESTROY_TEXOBJ) {
    evict_texture_object(reader.read<u32>());
  } else if (subCmd == GX_AURORA_DESTROY_TLUT) {
    evict_tlut_object(reader.read<u32>());
  } else if (subCmd == GX_AURORA_DESTROY_COPY_TEX) {
    evict_copy_texture(reinterpret_cast<const void*>(reader.read<u64>()));
  } else if (subCmd == GX_AURORA_DRAW_SIZED) {
    const u8 cmd = reader.read<u8>();
    const u32 byteLen = reader.read<u32>();
    const GXVtxFmt fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
    const GXPrimitive prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
    if (byteLen != 0) {
      u32 vtxSize;
      if (g_gxState.lastVtxFmt == fmt) {
        vtxSize = g_gxState.lastVtxSize;
      } else {
        vtxSize = calc_vtx_size(fmt);
      }
      AURORA_ASSERT(vtxSize != 0 && byteLen % vtxSize == 0,
                    "GX_AURORA_DRAW_SIZED: {} bytes is not a whole number of size-{} vertices", byteLen, vtxSize);
      u32 vtxCount = byteLen / vtxSize;
      AURORA_ASSERT(vtxCount <= 0xFFFF, "GX_AURORA_DRAW_SIZED: too many vertices ({})", vtxCount);
      draw_prim(prim, fmt, static_cast<u16>(vtxCount), reader);
    }
  } else if (subCmd == GX_AURORA_DRAW_INDEXED) {
    ZoneScopedN("DRAW_INDEXED");
    const u8 cmd = reader.read<u8>();
    const u16 vtxCount = reader.read<u16>();
    const u32 indexCount = reader.read<u32>();
    const GXVtxFmt fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
    const GXPrimitive prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
    AURORA_ASSERT(prim == GX_TRIANGLES, "GX_AURORA_DRAW_INDEXED: primitive must be GX_TRIANGLES, got {}",
                  static_cast<u32>(prim));
    const size_t idxBytes = static_cast<size_t>(indexCount) * sizeof(u16);
    // Index data is always host-endian; push it to the GPU buffer as-is
    const auto indexData = reader.take(idxBytes);
    const gfx::Range idxRange = gfx::push_indices(indexData.data(), indexData.size(), 4);
    u32 vtxSize;
    if (g_gxState.lastVtxFmt == fmt) {
      vtxSize = g_gxState.lastVtxSize;
    } else {
      vtxSize = calc_vtx_size(fmt);
    }
    const u32 totalVtxBytes = vtxCount * vtxSize;
    const auto vertexData = reader.take(totalVtxBytes);
    const gfx::Range vertRange = gfx::push_verts(vertexData.data(), vertexData.size(), 4);
    if (indexCount != 0) {
      push_gx_draw(prim, fmt, vtxCount, vertRange, idxRange, indexCount);
    }
  } else if (subCmd == GX_AURORA_DEBUG_GROUP_PUSH) {
    auto label = reader.read_string();
    gfx::push_debug_group(std::move(label));
  } else if (subCmd == GX_AURORA_DEBUG_GROUP_POP) {
    pop_debug_group();
  } else if (subCmd == GX_AURORA_DEBUG_MARKER_INSERT) {
    auto label = reader.read_string();
    gfx::insert_debug_marker(std::move(label));
  } else if (subCmd == GX_AURORA_CLEAR) {
    const auto flags = reader.read<u8>();
    const gfx::ClipRect rect{
        .x = static_cast<int32_t>(reader.read<u32>()),
        .y = static_cast<int32_t>(reader.read<u32>()),
        .width = static_cast<int32_t>(reader.read<u32>()),
        .height = static_cast<int32_t>(reader.read<u32>()),
    };
    const GXColor color{reader.read<u8>(), reader.read<u8>(), reader.read<u8>(), reader.read<u8>()};
    const auto z = reader.read<u32>();
    clear_efb(rect, (flags & 0x08) != 0, (flags & 0x01) != 0, (flags & 0x02) != 0, (flags & 0x04) != 0, color, z);
  }

  else {
    Log.error("Unknown Aurora subcommand: {:04X}", subCmd);
  }
}

void clear_draw_cache() noexcept {
  sDrawCache.bindGeneration = 0;
  sDrawCache.uniformRange = {};
  sDrawCache.fogRange = {};
  sDrawCache.hasFogRange = false;
}

} // namespace aurora::gx::fifo
