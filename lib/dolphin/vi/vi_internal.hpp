#pragma once

#include <dolphin/gx/GXStruct.h>

#include <cstdint>

#include <aurora/math.hpp>

namespace aurora::vi {
void configure(const GXRenderModeObj* rm) noexcept;
Vec2<uint32_t> configured_fb_size() noexcept;
// The lines of the frame buffer the configured mode shows, or 0 before VIConfigure.
uint32_t configured_xfb_lines() noexcept;
// The frame buffer the display shows: the last VISetNextFrameBuffer that VIFlush committed.
const void* displayed_frame_buffer() noexcept;
} // namespace aurora::vi
