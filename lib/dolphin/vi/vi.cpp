#include <dolphin/vi.h>

#include "../../window.hpp"
#include "aurora/math.hpp"
#include "vi_internal.hpp"

#include <atomic>
#include <optional>

namespace aurora::vi {
std::optional<GXRenderModeObj> g_renderMode;

// The frame buffer VISetNextFrameBuffer names, and the one the display shows once VIFlush has
// committed it. The game may set them from its retrace callbacks, off the rendering thread.
std::atomic<void*> g_nextFrameBuffer{nullptr};
std::atomic<void*> g_displayedFrameBuffer{nullptr};

Vec2<uint32_t> render_mode_size() noexcept {
  if (!g_renderMode) {
    return {640, 480};
  }
  return {g_renderMode->fbWidth, g_renderMode->efbHeight};
}

void configure(const GXRenderModeObj* rm) noexcept {
  const auto oldSize = render_mode_size();
  if (rm == nullptr) {
    g_renderMode.reset();
  } else {
    g_renderMode = *rm;
  }
  if (render_mode_size() != oldSize) {
    window::notify_frame_buffer_change();
  }
}

Vec2<uint32_t> configured_fb_size() noexcept { return render_mode_size(); }

uint32_t configured_xfb_lines() noexcept { return g_renderMode ? g_renderMode->xfbHeight : 0; }

const void* displayed_frame_buffer() noexcept { return g_displayedFrameBuffer.load(std::memory_order_acquire); }
} // namespace aurora::vi

extern "C" {
void VIInit() {}
void VIConfigure(const GXRenderModeObj* rm) { aurora::vi::configure(rm); }
void VIConfigurePan(u16 xOrg, u16 yOrg, u16 width, u16 height) {
  (void)xOrg;
  (void)yOrg;
  (void)width;
  (void)height;
}
u32 VIGetTvFormat() { return 0; }
void VIFlush() {
  aurora::vi::g_displayedFrameBuffer.store(aurora::vi::g_nextFrameBuffer.load(std::memory_order_acquire),
                                           std::memory_order_release);
}
void VISetNextFrameBuffer(void* fb) { aurora::vi::g_nextFrameBuffer.store(fb, std::memory_order_release); }
void* VIGetNextFrameBuffer() { return aurora::vi::g_nextFrameBuffer.load(std::memory_order_acquire); }
void* VIGetCurrentFrameBuffer() { return aurora::vi::g_displayedFrameBuffer.load(std::memory_order_acquire); }

void VISetWindowTitle(const char* title) { aurora::window::set_title(title); }
void VISetWindowFullscreen(bool fullscreen) { aurora::window::set_fullscreen(fullscreen); }
bool VIGetWindowFullscreen() { return aurora::window::get_fullscreen(); }
void VISetWindowSize(uint32_t width, uint32_t height) { aurora::window::set_window_size(width, height); }
void VISetWindowPosition(uint32_t x, uint32_t y) { aurora::window::set_window_position(x, y); }
void VICenterWindow() { aurora::window::center_window(); }
void VISetFrameBufferScale(float scale) { aurora::window::set_frame_buffer_scale(scale); }
}
