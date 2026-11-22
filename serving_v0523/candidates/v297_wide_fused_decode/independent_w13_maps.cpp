// Follow the source loader's TMA coordinates through the source encoder's descriptors.
template <bool Weight> void inspect(uint64_t& matched, uint64_t& half_errors, uint64_t& holes) {
  constexpr unsigned k = 4096, e = 32, inner = 4096, band = TEST_N / 2;
  constexpr unsigned bytes = Weight ? 4 : 1, packed_per_channel = Weight ? 2 : 1;
  CUtensorMap encoded{};
  assert(make_map<Weight>(Tensor{e, k / 16}, inner, &encoded));
  assert(encoded.el == bytes && encoded.d[0] == 16 && encoded.d[1] == inner * packed_per_channel / 16);
  assert(encoded.d[2] == 2 && encoded.d[3] == e * k / 16);
  assert(encoded.b[0] == 16 && encoded.b[1] == band * packed_per_channel / 16 && encoded.b[2] == 2 && encoded.b[3] == 8);
  MuseGateCtx ctx{};
  ctx.params.b = ctx.params.bs = &encoded;
  MusePackedGateLoader<Weight> loader(ctx);
  for (unsigned expert : {0u, 19u, e-1}) {
    for (unsigned kblock : {0u, k / 128 / 2, k / 128 - 2}) {
      for (unsigned tile = 0; tile < inner / band; ++tile) {
        loader.seek(expert, tile, kblock);
        for (unsigned part = 0; part < 2; ++part) {
          int4 dst{}; loader.load(&dst, nullptr);
          assert(last_map == &encoded);
          assert(last_col == tile * encoded.b[1]);
          assert(last_row == expert * (k / 16) + (kblock + part) * 8);
          for (unsigned kr = 0; kr < encoded.b[3]; ++kr) for (unsigned h = 0; h < 2; ++h) {
            for (unsigned j = 0; j < band * packed_per_channel; ++j) {
              const uint64_t src = (last_row + kr) * encoded.s[2] + h * encoded.s[1]
                                 + (last_col + j / 16) * encoded.s[0] + (j % 16) * bytes;
              const uint64_t physical_row = expert * (k / 16) + (kblock + part) * 8 + kr;
              const uint64_t native = ((physical_row * 2 + h) * inner * packed_per_channel
                                      + tile * band * packed_per_channel + j) * bytes;
              assert(src == native);
              const uint64_t local = ((kr * 2 + h) * encoded.b[1] + j / 16) * 16 + j % 16;
              assert(local == (kr * TEST_N + h * band) * packed_per_channel + j);
              ++matched;
              half_errors += src - h * encoded.s[1] != native;
              // A narrowed second dimension omits the latter half of each physical plane.
              holes += j >= 16 * (encoded.b[1] / 2);
            }
          }
        }
      }
    }
  }
}
int main() {
  uint64_t matched = 0, wrong_half = 0, holes = 0;
  inspect<true>(matched, wrong_half, holes);
  inspect<false>(matched, wrong_half, holes);
  assert(wrong_half * 2 == matched && holes * 2 == matched);
  std::printf("%llu %llu %llu\n", (unsigned long long)matched, (unsigned long long)wrong_half,
              (unsigned long long)holes);
}
