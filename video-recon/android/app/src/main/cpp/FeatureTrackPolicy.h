#pragma once
#include <cstddef>

// Active oldest-frame tracks need the complete configured clone window to gain
// baseline. Lost tracks are still processed immediately by the normal updater.
inline bool retireOldestFeatureTracks(std::size_t clones, int maximum) {
    return maximum > 0 && clones > static_cast<std::size_t>(maximum);
}
