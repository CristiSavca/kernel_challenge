"""CPU reference planner for one already-routed expert window. All ranks need identical inputs."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Costs:
    # Measured/calibrated by an integrator. The planner has no workload or hardware defaults.
    rows_per_tile: int
    owner_ns_per_issued_row: float
    helper_ns_per_issued_row: float
    link_bytes_per_ns: float
    launch_ns_per_move: float
    global_planning_ns: float
    minimum_saving_ns: float
    weight_bytes: int
    hidden: int
    intermediate: int
    helper_workspace_bytes: int

    def __post_init__(self):
        positive = (self.owner_ns_per_issued_row, self.helper_ns_per_issued_row, self.link_bytes_per_ns)
        nonnegative = (self.launch_ns_per_move, self.global_planning_ns, self.minimum_saving_ns)
        if any(not isfinite(x) or x <= 0 for x in positive):
            raise ValueError("time and bandwidth rates must be finite and positive")
        if any(not isfinite(x) or x < 0 for x in nonnegative):
            raise ValueError("overheads must be finite and nonnegative")
        if any(type(x) is not int or x <= 0 for x in
               (self.rows_per_tile, self.weight_bytes, self.hidden, self.intermediate)):
            raise ValueError("tile, weight and activation dimensions must be positive integers")
        if type(self.helper_workspace_bytes) is not int or self.helper_workspace_bytes < 0:
            raise ValueError("helper scratch size must be a nonnegative integer")

    def issued(self, rows):
        tile = self.rows_per_tile
        return ((rows + tile - 1) // tile) * tile

    def transfer_bytes(self, rows):
        # One bank, BF16 input and output, and a 64-bit canonical slot in each direction.
        return self.weight_bytes + rows * (4 * self.hidden + 16)

    def staging_bytes(self, rows, helper):
        # Keep input, output and two copies of route metadata through asynchronous retirement.
        staged = rows * (4 * self.hidden + 32)
        if helper:
            # W13 [n, 2I], activated [n, I], backend-private workspace and received bank.
            staged += self.weight_bytes + self.helper_workspace_bytes + rows * 6 * self.intermediate
        return staged


@dataclass(frozen=True)
class Move:
    owner: int
    helper: int
    expert: int
    begin: int
    end: int

    @property
    def rows(self):
        return self.end - self.begin


@dataclass(frozen=True)
class Plan:
    moves: tuple
    original_ns: float
    projected_ns: float
    extra_bytes_by_rank: tuple


def plan_splits(counts, owners, scratch_bytes, costs):
    """Return disjoint suffixes in stable native per-expert route order.

    A helper accepts at most one bank. A sender never becomes a helper. The global
    makespan model serializes the compute and the slowest endpoint's transfers.
    It assumes disjoint endpoints can progress in parallel. Independent streams and
    a topology/bisection check are prerequisites to using that model on hardware.
    """
    ranks = len(scratch_bytes)
    if ranks < 1 or len(counts) != len(owners):
        raise ValueError("need the complete rank budgets, expert counts and owner map")
    if any(type(x) is not int or x < 0 for x in (*counts, *scratch_bytes)):
        raise ValueError("counts and budgets must be nonnegative integers")
    if any(type(r) is not int or r < 0 or r >= ranks for r in owners):
        raise ValueError("invalid canonical expert owner")

    remaining = list(counts)
    compute = [0.0] * ranks
    for e, rows in enumerate(counts):
        compute[owners[e]] += costs.issued(rows) * costs.owner_ns_per_issued_row
    baseline = max(compute)
    best = Plan((), baseline, baseline, (0,) * ranks)
    if best.original_ns == 0:
        return best
    # Splitting cannot reduce the total number of fixed-width tiles. Even an
    # ideal balance pays for one bank copy before realizing any saving.
    ideal_ns = sum(costs.issued(n) for n in counts) / ranks * min(
        costs.helper_ns_per_issued_row, costs.owner_ns_per_issued_row
    )
    first_transfer_ns = costs.weight_bytes / costs.link_bytes_per_ns + costs.launch_ns_per_move
    if baseline - ideal_ns <= first_transfer_ns + costs.global_planning_ns + costs.minimum_saving_ns:
        return best
    senders, helpers, moves = set(), set(), []
    wire = [0.0] * ranks
    scratch = [0] * ranks
    while len(helpers) < ranks - 1:
        choice = None
        # Prefer the largest reduction in a pair's bottleneck. Inspect all pairs
        # so tied bottlenecks can improve together before the final global check.
        for expert, available in enumerate(remaining):
            owner = owners[expert]
            if not available or owner in helpers:
                continue
            tile = costs.rows_per_tile
            # Within an owner compute plateau, moving more rows can only increase
            # helper work, bytes and memory. Test the first count that frees each tile.
            first = (available - 1) % tile + 1
            for helper in range(ranks):
                if helper == owner or helper in senders or helper in helpers or compute[helper] >= compute[owner]:
                    continue
                previous_pair = max(compute[owner] + wire[owner], compute[helper] + wire[helper])
                for rows in range(first, available + 1, tile):
                    owner_bytes = costs.staging_bytes(rows, False)
                    helper_bytes = costs.staging_bytes(rows, True)
                    if scratch[owner] + owner_bytes > scratch_bytes[owner] or helper_bytes > scratch_bytes[helper]:
                        break
                    transport = costs.transfer_bytes(rows) / costs.link_bytes_per_ns + costs.launch_ns_per_move
                    freed = costs.issued(available) - costs.issued(available - rows)
                    owner_cost = compute[owner] - freed * costs.owner_ns_per_issued_row
                    helper_cost = compute[helper] + costs.issued(rows) * costs.helper_ns_per_issued_row
                    pair_cost = max(owner_cost + wire[owner], helper_cost + wire[helper]) + transport
                    improvement = previous_pair - pair_cost
                    if improvement <= 0:
                        continue
                    key = (-improvement, rows, expert, owner, helper)
                    if choice is None or key < choice[0]:
                        choice = (key, Move(owner, helper, expert, available - rows, available),
                                  owner_cost, helper_cost, transport, owner_bytes, helper_bytes)
        if choice is None:
            break
        _, move, donor_ns, helper_ns, transfer_ns, donor_bytes, helper_bytes = choice
        owner, helper = move.owner, move.helper
        remaining[move.expert] = move.begin
        senders.add(owner)
        helpers.add(helper)
        moves.append(move)
        compute[owner], compute[helper] = donor_ns, helper_ns
        wire[owner] += transfer_ns
        wire[helper] += transfer_ns
        scratch[owner] += donor_bytes
        scratch[helper] += helper_bytes
        finish = max(compute) + max(wire) + costs.global_planning_ns
        if baseline - finish > costs.minimum_saving_ns and finish < best.projected_ns:
            best = Plan(tuple(moves), baseline, finish, tuple(scratch))
    return best
