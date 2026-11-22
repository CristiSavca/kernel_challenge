"""CPU-only proof for an exact, fixed-budget DFLASH selector tree."""

from fractions import Fraction
from heapq import heappop, heappush
from itertools import product
from random import Random


ROOT = ()


def children(lattice, prefix):
    if len(prefix) < len(lattice):
        row = lattice[len(prefix)][prefix[-1] if prefix else 0]
        for candidate, probability in enumerate(row):
            yield prefix + (candidate,), probability


def choose_tree(lattice, budget):
    selected = {}
    frontier = []
    for child, probability in children(lattice, ROOT):
        heappush(frontier, (-probability, len(child), child))
    while frontier and len(selected) < budget:
        score, _, prefix = heappop(frontier)
        mass = -score
        selected[prefix] = mass
        for child, probability in children(lattice, prefix):
            heappush(frontier, (-mass * probability, len(child), child))
    return selected


def all_prefix_masses(lattice):
    masses = {ROOT: Fraction(1)}
    for depth in range(len(lattice)):
        for prefix in tuple(masses):
            if len(prefix) == depth:
                for child, probability in children(lattice, prefix):
                    masses[child] = masses[prefix] * probability
    return masses


def greedy_chain(lattice):
    prefix = ROOT
    mass = Fraction(1)
    selected = {}
    for _ in lattice:
        prefix, probability = max(children(lattice, prefix), key=lambda item: item[1])
        mass *= probability
        selected[prefix] = mass
    return selected


def check_optimality():
    rng = Random(188)
    depths = 3
    prefixes = [p for size in range(1, depths + 1) for p in product(range(2), repeat=size)]
    valid = [[] for _ in range(len(prefixes) + 1)]
    for flags in product((False, True), repeat=len(prefixes)):
        subset = {p for p, include in zip(prefixes, flags) if include}
        if all(p[:-1] == ROOT or p[:-1] in subset for p in subset):
            valid[len(subset)].append(subset)

    def row():
        left = Fraction(rng.randrange(11), 10)
        return left, 1 - left

    checked = 0
    for _ in range(64):
        lattice = [[row()], [row(), row()], [row(), row()]]
        masses = all_prefix_masses(lattice)
        for budget in range(len(prefixes) + 1):
            selected = choose_tree(lattice, budget)
            assert len(selected) == budget
            assert all(p[:-1] == ROOT or p[:-1] in selected for p in selected)
            expected = 1 + sum(selected.values())
            optimum = 1 + max(sum(masses[p] for p in subset) for subset in valid[budget])
            assert expected == optimum
            # Enumerate possible target paths. A wrong edge commits a bonus
            # token and ends the walk; deeper guesses carry no credit.
            observed_expectation = 0
            for target in product(range(2), repeat=depths):
                accepted = 1
                for size in range(1, depths + 1):
                    if target[:size] not in selected:
                        break
                    accepted += 1
                observed_expectation += masses[target] * accepted
            assert expected == observed_expectation
            checked += 1
    return checked


def check_attention_and_commit():
    # Distinct physical output rows, arbitrary sibling order and all valid
    # small ancestor-closed candidate subsets across contexts and windows.
    checks = 0
    lattice = [[(Fraction(3, 5), Fraction(2, 5))]] + [
        [(Fraction(3, 5), Fraction(2, 5))] * 2 for _ in range(6)
    ]
    for budget in (0, 1, 3, 7, 15):
        tree = list(choose_tree(lattice, budget))
        rows = dict(zip([ROOT] + tree[::-1], range(budget + 1)))
        for prefix in [ROOT] + tree:
            ancestors = [prefix[:depth] for depth in range(len(prefix) + 1)]
            for context, window in product((0, 1, 7, 64, 2048, 2055, 131070), (1, 4, 32, 2048)):
                # Reference is the literal linear path ending at this query.
                committed = range(context)
                linear = (tuple(committed[-window:]) + tuple(context + rows[a] for a in ancestors))[-window:]
                # Packed evaluation computes a query-specific position, then
                # clamps the live prefix and uses a branch membership bitset.
                position = context + len(prefix)
                left = max(position - window + 1, 0)
                visible_prefix = tuple(range(min(left, context), context))
                visible_tree = tuple(
                    context + rows[node] for node in sorted(rows, key=len)
                    if node == prefix[:len(node)] and context + len(node) >= left
                )
                assert visible_prefix + visible_tree == linear
                checks += 1
            # Reorder only this request's reserved output slots. The next draft
            # sees the live path first and can index untouched reserve after it.
            accepted = [rows[a] for a in ancestors]
            reslots = accepted + [i for i in range(budget + 1) if i not in accepted]
            assert sorted(reslots) == list(range(budget + 1))
            read = {index: node for node, index in rows.items()}
            assert [read[index] for index in reslots[:len(accepted)]] == ancestors
    return checks


def show_scenarios():
    def lattice_at(root_p, body_p):
        root = (Fraction(root_p), 1 - Fraction(root_p))
        body = (Fraction(body_p), 1 - Fraction(body_p))
        return [[root]] + [[body, body] for _ in range(6)]

    for root_p, body_p in (("0.79", "0.79"), ("0.55", "0.55"), ("0.55", "0.90")):
        lattice = lattice_at(root_p, body_p)
        linear = 1 + sum(greedy_chain(lattice).values())
        equal_rows = 1 + sum(choose_tree(lattice, 7).values())
        double_rows = 1 + sum(choose_tree(lattice, 15).values())
        print(
            f"Hypothetical p(root)={root_p} p(later)={body_p} committed/step "
            f"{float(linear):.4f} linear, {float(equal_rows):.4f} fixed 8 rows, "
            f"{float(double_rows):.4f} fixed 16 rows."
        )


if __name__ == "__main__":
    print(f"Passed {check_optimality():,} exact tree-budget and acceptance checks.")
    print(f"Passed {check_attention_and_commit():,} packed SWA mask checks and path commits.")
    show_scenarios()
