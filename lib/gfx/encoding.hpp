#pragma once

#include "frame_packet.hpp"

namespace aurora::gfx {

bool bind_pipeline(PipelineRef ref, const wgpu::RenderPassEncoder& pass);
void bind_pipeline(const wgpu::RenderPipeline& pipeline, const wgpu::RenderPassEncoder& pass);

// What a run of draws of one kind left bound in the render pass, so that each draw sets only what differs from the
// draw before. Forgotten at the start of a pass, after a custom draw and whenever the kind of draw changes.
struct DrawPassState {
  bool indexBufferBound = false;
  bool hasUniformOffset = false;
  uint32_t uniformOffset = 0;
  bool hasTextureBindGroup = false;
  BindGroupRef textureBindGroup = 0;
  bool hasBlendAlpha = false;
  uint32_t blendAlpha = 0;
};
DrawPassState& draw_pass_state() noexcept;

namespace detail {
void encode_op(wgpu::CommandEncoder& encoder, FramePacket& frame, const FrameOp& op);
// Only the op's array cache fills, for an op with nothing else to encode (push_cached_storage has counted on them).
void encode_array_fills(wgpu::CommandEncoder& encoder, const FramePacket& frame, const FrameOp& op);
}

} // namespace aurora::gfx
