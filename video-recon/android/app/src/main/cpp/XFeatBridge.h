#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <mutex>
#include <vector>

namespace live_xfeat {

struct Candidate {
  float x = 0;
  float y = 0;
  float score = 0;
};

struct Frame {
  double timestamp = std::numeric_limits<double>::quiet_NaN();
  int width = 0;
  int height = 0;
  std::vector<Candidate> candidates;
};

enum class Lookup { Disabled, Missing, Found };

class Store {
public:
  bool submit(double timestamp, int width, int height, const float *triples, size_t count) {
    if (!std::isfinite(timestamp) || width <= 0 || height <= 0 || (!triples && count)) return false;
    Frame frame{timestamp, width, height, {}};
    frame.candidates.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      const Candidate point{triples[3 * i], triples[3 * i + 1], triples[3 * i + 2]};
      if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.score) || point.score <= 0 ||
          point.x < 0 || point.y < 0 || point.x >= width || point.y >= height)
        continue;
      frame.candidates.push_back(point);
    }
    std::stable_sort(frame.candidates.begin(), frame.candidates.end(),
                     [](const Candidate &a, const Candidate &b) { return a.score > b.score; });
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = true;
    frames_.erase(std::remove_if(frames_.begin(), frames_.end(), [&](const Frame &item) {
                    return std::abs(item.timestamp - timestamp) <= 1e-9;
                  }), frames_.end());
    frames_.push_back(std::move(frame));
    while (frames_.size() > 4) frames_.pop_front();
    submitted_++;
    return true;
  }

  void setEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = enabled;
    if (!enabled) frames_.clear();
  }

  Lookup lookup(double timestamp, int width, int height, Frame &out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_) return Lookup::Disabled;
    while (!frames_.empty() && frames_.front().timestamp < timestamp - 1e-7) frames_.pop_front();
    const auto match = std::find_if(frames_.begin(), frames_.end(), [&](const Frame &frame) {
      return std::abs(frame.timestamp - timestamp) <= 1e-9 && frame.width == width && frame.height == height;
    });
    if (match == frames_.end()) return Lookup::Missing;
    out = *match;
    frames_.erase(match);
    consumed_++;
    return Lookup::Found;
  }

  void selected(size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    selected_ += count;
    lastSelected_ = count;
  }

  struct Stats { size_t submitted, consumed, selected, lastSelected; };
  Stats stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {submitted_, consumed_, selected_, lastSelected_};
  }

private:
  mutable std::mutex mutex_;
  bool enabled_ = false;
  std::deque<Frame> frames_;
  size_t submitted_ = 0;
  size_t consumed_ = 0;
  size_t selected_ = 0;
  size_t lastSelected_ = 0;
};

inline thread_local Store *activeStore = nullptr;
inline thread_local double detectionTimestamp = std::numeric_limits<double>::quiet_NaN();

class FeedScope {
public:
  explicit FeedScope(Store *store) : previous_(activeStore) { activeStore = store; }
  ~FeedScope() { activeStore = previous_; detectionTimestamp = std::numeric_limits<double>::quiet_NaN(); }
private:
  Store *previous_;
};

inline void setDetectionTimestamp(double timestamp) { detectionTimestamp = timestamp; }
inline Lookup lookup(int width, int height, Frame &out) {
  return activeStore ? activeStore->lookup(detectionTimestamp, width, height, out) : Lookup::Disabled;
}
inline void selected(size_t count) { if (activeStore) activeStore->selected(count); }

} // namespace live_xfeat
