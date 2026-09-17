// Writes the self-play samples in a simple binary format, read by Python
// (cpp_dataset.py) for training.
//
// Format (little-endian, all float32 except the header):
//   magic       4 bytes "DAMA"
//   version     int32   = 1
//   n_samples   int32
//   in_planes   int32   (7)
//   policy_size int32   (256)
//   then n_samples consecutive records:
//       x     in_planes*64 float32
//       pi    policy_size  float32
//       mask  policy_size  float32
//       z     1            float32
//
// Deliberately trivial: no compression, no dependencies. Training stays in
// PyTorch/Python, so the only contract to honor is "numpy must read it back
// without surprises" -- checked by tests/test_cpp_dataset.py.
#pragma once
#include "selfplay.hpp"
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

namespace dama {

inline bool write_dataset(const std::string& path,
                          const std::vector<GameResult>& games) {
    std::vector<const Sample*> all;
    for (const auto& g : games)
        for (const auto& s : g.samples) all.push_back(&s);

    // ATOMIC write: first to a temporary file, then a rename. So an interrupted
    // process never leaves a truncated dataset that training would try to read.
    std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;

    int32_t n = (int32_t)all.size();
    int32_t in_planes = ENC_IN_PLANES;
    int32_t policy = ENC_POLICY_SIZE;
    int32_t version = 1;
    bool ok = true;
    ok &= (std::fwrite("DAMA", 1, 4, f) == 4);
    ok &= (std::fwrite(&version, sizeof(int32_t), 1, f) == 1);
    ok &= (std::fwrite(&n, sizeof(int32_t), 1, f) == 1);
    ok &= (std::fwrite(&in_planes, sizeof(int32_t), 1, f) == 1);
    ok &= (std::fwrite(&policy, sizeof(int32_t), 1, f) == 1);

    for (const Sample* s : all) {
        ok &= (std::fwrite(s->x.data(), sizeof(float), s->x.size(), f) == s->x.size());
        ok &= (std::fwrite(s->pi.data(), sizeof(float), s->pi.size(), f) == s->pi.size());
        ok &= (std::fwrite(s->mask.data(), sizeof(float), s->mask.size(), f) == s->mask.size());
        ok &= (std::fwrite(&s->z, sizeof(float), 1, f) == 1);
        if (!ok) break;
    }
    std::fclose(f);
    if (!ok) { std::remove(tmp.c_str()); return false; }
    std::remove(path.c_str());               // Windows: rename does not overwrite
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

}  // namespace dama
