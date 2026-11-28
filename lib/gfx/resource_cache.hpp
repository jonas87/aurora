#pragma once

#include "types.hpp"

namespace aurora::gfx {

BindGroupRef bind_group_ref(const WGPUBindGroupDescriptor& descriptor);
// The same with a key the caller computed, which must identify the descriptor as its hash would: for descriptors
// whose layout and entry fields are fixed, a hash of the varying handles alone, far shorter than the entries.
BindGroupRef bind_group_ref(const WGPUBindGroupDescriptor& descriptor, BindGroupRef key);
wgpu::BindGroup find_bind_group(BindGroupRef id);
wgpu::Sampler sampler_ref(const wgpu::SamplerDescriptor& descriptor);
// The cached sampler without a reference of its own, for a bind group's entry (the bind group takes one): the cache
// keeps its samplers until shutdown, and a draw's lookup does not pay a reference count's two calls into Dawn.
WGPUSampler sampler_handle(const wgpu::SamplerDescriptor& descriptor);

namespace detail {
void clear_bind_group_cache();
void expire_cached_bind_groups();
void shutdown_resource_cache();
} // namespace detail

} // namespace aurora::gfx
