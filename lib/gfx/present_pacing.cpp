#include "present_pacing.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <thread>

namespace aurora::gfx::present {
namespace {

using Clock = std::chrono::steady_clock;

// The time the spin covers before the due time.
constexpr int64_t SpinNs = 500'000;
// The present times kept for aurora_take_present_times.
constexpr size_t KeptPresents = 256;

std::atomic<int64_t> g_nextPresentTime{0};
std::mutex g_presentTimesMutex;
std::deque<AuroraPresentTime> g_presentTimes;

void sleep_for_ns(int64_t ns) noexcept {
#ifdef _WIN32
  struct Timer {
    HANDLE handle = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    ~Timer() {
      if (handle != nullptr) {
        CloseHandle(handle);
      }
    }
  };
  thread_local Timer timer;
  LARGE_INTEGER due;
  // A relative due time is negative, in units of 100 nanoseconds.
  due.QuadPart = -std::max<int64_t>(1, ns / 100);
  if (timer.handle != nullptr && SetWaitableTimer(timer.handle, &due, 0, nullptr, nullptr, FALSE)) {
    WaitForSingleObject(timer.handle, INFINITE);
    return;
  }
#endif
  std::this_thread::sleep_for(std::chrono::nanoseconds(ns));
}

} // namespace

int64_t steady_now() noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}

void set_next_present_time(int64_t steadyNs) noexcept { g_nextPresentTime.store(steadyNs, std::memory_order_release); }

int64_t take_next_present_time() noexcept { return g_nextPresentTime.exchange(0, std::memory_order_acq_rel); }

void wait_until(int64_t steadyNs) noexcept {
  const int64_t sleep = steadyNs - SpinNs - steady_now();
  if (sleep > 0) {
    sleep_for_ns(sleep);
  }
  while (steady_now() < steadyNs) {
    std::this_thread::yield();
  }
}

void record_present_time(int64_t target, int64_t ready, int64_t presented) noexcept {
  std::lock_guard lock{g_presentTimesMutex};
  g_presentTimes.push_back({.target = target, .ready = ready, .presented = presented});
  while (g_presentTimes.size() > KeptPresents) {
    g_presentTimes.pop_front();
  }
}

size_t take_present_times(AuroraPresentTime* times, size_t capacity) noexcept {
  std::lock_guard lock{g_presentTimesMutex};
  size_t count = 0;
  while (count < capacity && !g_presentTimes.empty()) {
    times[count++] = g_presentTimes.front();
    g_presentTimes.pop_front();
  }
  return count;
}

} // namespace aurora::gfx::present
