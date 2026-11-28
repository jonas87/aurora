#pragma once

#include <aurora/aurora.h>

#include <cstddef>
#include <cstdint>

namespace aurora::gfx::present {

// The steady clock's time in nanoseconds (std::chrono::steady_clock), the clock of the pacing's times.
int64_t steady_now() noexcept;

// Present pacing (aurora_set_present_time): the time the next frame's present waits for, which the frame's present
// request takes as it is recorded; 0 presents it as soon as it is ready.
void set_next_present_time(int64_t steadyNs) noexcept;
int64_t take_next_present_time() noexcept;

// Waits until a steady-clock time in nanoseconds: sleeps until a little before it, on Windows on a high-resolution
// waitable timer, then spins to it, as a plain sleep can wake a millisecond or two late.
void wait_until(int64_t steadyNs) noexcept;

// The primary target's presents: each one's requested time, when it was ready and when its present returned, kept
// until aurora_take_present_times takes them (the last 256).
void record_present_time(int64_t target, int64_t ready, int64_t presented) noexcept;
size_t take_present_times(AuroraPresentTime* times, size_t capacity) noexcept;

} // namespace aurora::gfx::present
