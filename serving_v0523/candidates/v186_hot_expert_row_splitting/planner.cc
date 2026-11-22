// Standalone CPU implementation for one observed window; no framework or GPU dependency.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

struct Costs {
  int64_t tile, weight, hidden, intermediate, workspace;
  double owner_ns, helper_ns, link, setup_ns, planning_ns, margin_ns;
};
struct Move {
  int32_t owner, helper, expert, begin, end;
};
using Bytes = uint64_t;

extern "C" int plan_hot_rows(int32_t ranks, int32_t experts, const int32_t* counts,
                             const int32_t* owners, const Bytes* budget, const Costs* config,
                             Move* output, Bytes* extra, double* estimate) {
  if (ranks < 1 || experts < 0 || !config || !budget || !output || !extra || !estimate ||
      (experts > 0 && (!counts || !owners))) return -1;
  const Costs& c = *config;
  for (double cost : {c.owner_ns, c.helper_ns, c.link, c.setup_ns, c.planning_ns, c.margin_ns}) {
    if (!std::isfinite(cost) || cost < 0) return -1;
  }
  if (c.owner_ns <= 0 || c.helper_ns <= 0 || c.link <= 0 || c.tile <= 0 || c.weight <= 0 ||
      c.hidden <= 0 || c.intermediate <= 0 || c.workspace < 0) return -1;
  if (c.tile > INT32_MAX || c.hidden > INT32_MAX || c.intermediate > INT32_MAX) return -1;
  const auto tiles = [&](int64_t n) { return ((n + c.tile - 1) / c.tile) * c.tile; };
  const auto maximum = [](const auto& v) { return *std::max_element(v.begin(), v.end()); };
  std::vector<double> gpu(ranks, 0.0), wire(ranks, 0.0);
  std::vector<Bytes> scratch(ranks, 0);
  std::vector<int32_t> remaining(experts);
  std::vector<bool> sender(ranks, false), helper(ranks, false);
  std::vector<Move> moves;
  double issued = 0.0;
  for (int e = 0; e < experts; ++e) {
    if (counts[e] < 0 || owners[e] < 0 || owners[e] >= ranks) return -1;
    remaining[e] = counts[e];
    int64_t rows = tiles(counts[e]);
    gpu[owners[e]] += rows * c.owner_ns;
    issued += rows;
  }
  const double base = maximum(gpu);
  estimate[0] = estimate[1] = base;
  std::fill_n(extra, ranks, 0);
  if (base - issued / ranks * std::min(c.owner_ns, c.helper_ns) <=
      c.weight / c.link + c.setup_ns + c.planning_ns + c.margin_ns) return 0;
  const Bytes per_donor = 4ull * c.hidden + 32;
  const Bytes per_helper = per_donor + 6ull * c.intermediate;
  const Bytes fixed_helper = Bytes(c.weight) + Bytes(c.workspace);
  int best_size = 0;
  for (int step = 0; step < ranks-1; ++step) {
    Move selected{};
    std::tuple<double, int64_t, int, int, int> best{};
    double donor_ns = 0, helper_ns = 0, selected_wire = 0;
    Bytes donor_bytes = 0, helper_bytes = 0;
    bool found = false;
    for (int e = 0; e < experts; ++e) {
      int owner = owners[e], n = remaining[e];
      if (n == 0 || helper[owner]) continue;
      int64_t first = (n-1) % c.tile + 1;
      for (int r = 0; r < ranks; ++r) {
        if (r == owner || sender[r] || helper[r] || gpu[r] >= gpu[owner] || budget[r] < fixed_helper) continue;
        Bytes available = std::min((budget[owner]-scratch[owner]) / per_donor,
                                   (budget[r]-fixed_helper) / per_helper);
        int64_t limit = std::min<Bytes>(n, available);
        double previous = std::max(gpu[owner] + wire[owner], gpu[r] + wire[r]);
        for (int64_t m = first; m <= limit; m += c.tile) {
          double transport = (double(c.weight) + m * double(per_donor-16)) / c.link + c.setup_ns;
          double od = gpu[owner] - (tiles(n) - tiles(n-m)) * c.owner_ns;
          double hr = gpu[r] + tiles(m)*c.helper_ns;
          double gain = previous - (std::max(od + wire[owner], hr + wire[r]) + transport);
          auto key = std::make_tuple(-gain, m, e, owner, r);
          if (gain > 0 && (!found || key < best)) {
            best = key;
            found = true;
            selected = {owner, r, e, int32_t(n-m), n};
            donor_ns = od; helper_ns = hr; selected_wire = transport;
            donor_bytes = m * per_donor; helper_bytes = fixed_helper + m * per_helper;
          }
        }
      }
    }
    if (!found) break;
    int owner = selected.owner, r = selected.helper;
    moves.push_back(selected);
    remaining[selected.expert] = selected.begin;
    sender[owner] = true; helper[r] = true;
    gpu[owner] = donor_ns; gpu[r] = helper_ns;
    wire[owner] += selected_wire; wire[r] += selected_wire;
    scratch[owner] += donor_bytes; scratch[r] += helper_bytes;
    double finish = maximum(gpu) + maximum(wire) + c.planning_ns;
    if (base - finish > c.margin_ns && finish < estimate[1]) {
      best_size = int(moves.size());
      std::copy(moves.begin(), moves.end(), output);
      std::copy(scratch.begin(), scratch.end(), extra);
      estimate[1] = finish;
    }
  }
  return best_size;
}
