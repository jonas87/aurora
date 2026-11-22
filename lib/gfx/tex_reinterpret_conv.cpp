#include "tex_reinterpret_conv.hpp"

#include "../internal.hpp"
#include "../webgpu/gpu.hpp"
#include "../webgpu/gpu_prof.hpp"
#include "texture.hpp"

#include <array>
#include <cstring>

namespace aurora::gfx::tex_reinterpret_conv {
static Module Log("aurora::gfx::tex_reinterpret_conv");

using webgpu::g_device;

namespace {

// The hardware's tile layouts: 8-bit formats in 8x4 tiles of 32 bytes, 16-bit formats in 4x4 tiles
// of 32 bytes, both row by row. A copy's byte is found in the copy's layout, a texel's bytes in the
// object's layout. RGB565 copies are kept at eight bits per channel; their bytes truncate to 5, 6
// and 5 bits as the hardware's copy does.
constexpr std::string_view Shader = R"(
struct Params {
    sourceFormat: u32,
    targetFormat: u32,
    sourceWidth: u32,
    sourceHeight: u32,
    targetWidth: u32,
    targetHeight: u32,
    pad0: u32,
    pad1: u32,
};
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: Params;

var<private> positions: array<vec2f, 3> = array(
    vec2f(-1.0, 1.0),
    vec2f(-1.0, -3.0),
    vec2f(3.0, 1.0),
);

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    return vec4f(positions[vi], 0.0, 1.0);
}

fn tiles_across(width: u32, tile: u32) -> u32 {
    return (width + tile - 1u) / tile;
}

// The copy's texel at logical (x, y), read from the copy texture at Aurora's render scale.
fn copy_texel(x: u32, y: u32) -> vec4f {
    let size = vec2u(textureDimensions(src));
    let scale = vec2f(size) / vec2f(f32(params.sourceWidth), f32(params.sourceHeight));
    let p = min(vec2u((vec2f(f32(x), f32(y)) + 0.5) * scale), size - 1u);
    return textureLoad(src, vec2i(p), 0);
}

fn to_byte(v: f32) -> u32 {
    return u32(floor(clamp(v, 0.0, 1.0) * 255.0 + 0.5));
}

// The byte at `offset` of the copy as the hardware writes it.
fn copy_byte(offset: u32) -> u32 {
    let tile = offset / 32u;
    let within = offset % 32u;
    if (params.sourceFormat == 0u) {
        let across = tiles_across(params.sourceWidth, 8u);
        let x = (tile % across) * 8u + within % 8u;
        let y = (tile / across) * 4u + within / 8u;
        return to_byte(copy_texel(x, y).r);
    }
    let across = tiles_across(params.sourceWidth, 4u);
    let x = (tile % across) * 4u + (within / 2u) % 4u;
    let y = (tile / across) * 4u + within / 8u;
    let c = copy_texel(x, y);
    let v = ((to_byte(c.r) >> 3u) << 11u) | ((to_byte(c.g) >> 2u) << 5u) | (to_byte(c.b) >> 3u);
    return select(v & 0xFFu, v >> 8u, within % 2u == 0u);
}

fn expand(v: u32, bits: u32) -> f32 {
    let full = (v << (8u - bits)) | (v >> (2u * bits - 8u));
    return f32(full & 0xFFu) / 255.0;
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let x = u32(pos.x);
    let y = u32(pos.y);
    if (params.targetFormat == 0u) {
        let across = tiles_across(params.targetWidth, 8u);
        let offset = ((y / 4u) * across + x / 8u) * 32u + (y % 4u) * 8u + x % 8u;
        let i = f32(copy_byte(offset)) / 255.0;
        return vec4f(i, i, i, i);
    }
    let across = tiles_across(params.targetWidth, 4u);
    let offset = ((y / 4u) * across + x / 4u) * 32u + ((y % 4u) * 4u + x % 4u) * 2u;
    let v = (copy_byte(offset) << 8u) | copy_byte(offset + 1u);
    if (params.targetFormat == 1u) {
        let i = f32(v & 0xFFu) / 255.0;
        return vec4f(i, i, i, f32(v >> 8u) / 255.0);
    }
    if (params.targetFormat == 2u) {
        return vec4f(expand(v >> 11u, 5u), expand((v >> 5u) & 0x3Fu, 6u), expand(v & 0x1Fu, 5u), 1.0);
    }
    if ((v & 0x8000u) != 0u) {
        return vec4f(expand((v >> 10u) & 0x1Fu, 5u), expand((v >> 5u) & 0x1Fu, 5u), expand(v & 0x1Fu, 5u), 1.0);
    }
    let a = (v >> 12u) & 0x7u;
    return vec4f(f32((v >> 8u) & 0xFu) / 15.0, f32((v >> 4u) & 0xFu) / 15.0, f32(v & 0xFu) / 15.0,
                 f32((a << 5u) | (a << 2u) | (a >> 1u)) / 255.0);
}
)"sv;

struct Params {
  uint32_t source;
  uint32_t target;
  uint32_t sourceWidth;
  uint32_t sourceHeight;
  uint32_t targetWidth;
  uint32_t targetHeight;
  uint32_t pad0;
  uint32_t pad1;
};
static_assert(sizeof(Params) == 32);

wgpu::RenderPipeline g_pipeline;
wgpu::BindGroupLayout g_bindGroupLayout;

} // namespace

bool plan(GXTexFmt copyFormat, GXTexFmt textureFormat, Source& source, Target& target) {
  switch (copyFormat) {
  case GX_TF_I8:
  case GX_TF_Z8:
  case GX_CTF_R8:
  case GX_CTF_G8:
  case GX_CTF_B8:
  case GX_CTF_A8:
    source = Source::Bytes8;
    break;
  case GX_TF_RGB565:
    source = Source::Rgb565;
    break;
  default:
    return false;
  }
  switch (textureFormat) {
  case GX_TF_I8:
    target = Target::I8;
    break;
  case GX_TF_IA8:
    target = Target::Ia8;
    break;
  case GX_TF_RGB565:
    target = Target::Rgb565;
    break;
  case GX_TF_RGB5A3:
    target = Target::Rgb5a3;
    break;
  default:
    return false;
  }
  // The same bytes read the same way need no conversion: the copy texture already holds them.
  return !((source == Source::Bytes8 && target == Target::I8) ||
           (source == Source::Rgb565 && target == Target::Rgb565));
}

void initialize() {
  const wgpu::ShaderSourceWGSL wgslDescriptor{wgpu::ShaderSourceWGSL::Init{
      .code = Shader,
  }};
  const wgpu::ShaderModuleDescriptor shaderDescriptor{
      .nextInChain = &wgslDescriptor,
      .label = "TexReinterpretConv Shader",
  };
  const auto module = g_device.CreateShaderModule(&shaderDescriptor);

  const std::array bindGroupLayoutEntries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture =
              wgpu::TextureBindingLayout{
                  .sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                  .viewDimension = wgpu::TextureViewDimension::e2D,
              },
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Fragment,
          .buffer =
              wgpu::BufferBindingLayout{
                  .type = wgpu::BufferBindingType::Uniform,
                  .minBindingSize = sizeof(Params),
              },
      },
  };
  const wgpu::BindGroupLayoutDescriptor bindGroupLayoutDescriptor{
      .label = "TexReinterpretConv Bind Group Layout",
      .entryCount = bindGroupLayoutEntries.size(),
      .entries = bindGroupLayoutEntries.data(),
  };
  g_bindGroupLayout = g_device.CreateBindGroupLayout(&bindGroupLayoutDescriptor);

  const wgpu::PipelineLayoutDescriptor layoutDescriptor{
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &g_bindGroupLayout,
  };
  const auto pipelineLayout = g_device.CreatePipelineLayout(&layoutDescriptor);

  constexpr std::array colorTargets{wgpu::ColorTargetState{
      .format = wgpu::TextureFormat::RGBA8Unorm,
  }};
  const wgpu::FragmentState fragmentState{
      .module = module,
      .entryPoint = "fs_main",
      .targetCount = colorTargets.size(),
      .targets = colorTargets.data(),
  };
  const wgpu::RenderPipelineDescriptor pipelineDescriptor{
      .label = "TexReinterpretConv",
      .layout = pipelineLayout,
      .vertex =
          wgpu::VertexState{
              .module = module,
              .entryPoint = "vs_main",
          },
      .primitive =
          wgpu::PrimitiveState{
              .topology = wgpu::PrimitiveTopology::TriangleList,
          },
      .fragment = &fragmentState,
  };
  g_pipeline = g_device.CreateRenderPipeline(&pipelineDescriptor);
}

void shutdown() {
  g_pipeline = {};
  g_bindGroupLayout = {};
}

void run(const wgpu::CommandEncoder& cmd, const ConvRequest& req) {
  const Params params{
      .source = static_cast<uint32_t>(req.source),
      .target = static_cast<uint32_t>(req.target),
      .sourceWidth = req.sourceWidth,
      .sourceHeight = req.sourceHeight,
      .targetWidth = req.targetWidth,
      .targetHeight = req.targetHeight,
  };
  const wgpu::BufferDescriptor bufferDescriptor{
      .label = "TexReinterpretConv Params",
      .usage = wgpu::BufferUsage::Uniform,
      .size = sizeof(Params),
      .mappedAtCreation = true,
  };
  const auto buffer = g_device.CreateBuffer(&bufferDescriptor);
  std::memcpy(buffer.GetMappedRange(), &params, sizeof(Params));
  buffer.Unmap();

  const std::array bindGroupEntries{
      wgpu::BindGroupEntry{
          .binding = 0,
          .textureView = req.src->sampleTextureView,
      },
      wgpu::BindGroupEntry{
          .binding = 1,
          .buffer = buffer,
          .size = sizeof(Params),
      },
  };
  const wgpu::BindGroupDescriptor bindGroupDescriptor{
      .layout = g_bindGroupLayout,
      .entryCount = bindGroupEntries.size(),
      .entries = bindGroupEntries.data(),
  };
  const auto bindGroup = g_device.CreateBindGroup(&bindGroupDescriptor);

  const std::array colorAttachments{
      wgpu::RenderPassColorAttachment{
          .view = req.dst->attachmentTextureView,
          .loadOp = wgpu::LoadOp::Clear,
          .storeOp = wgpu::StoreOp::Store,
          .clearValue = {0.0, 0.0, 0.0, 0.0},
      },
  };
  const wgpu::RenderPassDescriptor renderPassDescriptor{
      .label = "TexReinterpretConv Pass",
      .colorAttachmentCount = colorAttachments.size(),
      .colorAttachments = colorAttachments.data(),
      .timestampWrites = webgpu::gpu_prof::pass_writes("Copy reinterpret"),
  };
  const auto pass = cmd.BeginRenderPass(&renderPassDescriptor);
  pass.SetPipeline(g_pipeline);
  pass.SetBindGroup(0, bindGroup);
  pass.Draw(3);
  pass.End();
}

} // namespace aurora::gfx::tex_reinterpret_conv
