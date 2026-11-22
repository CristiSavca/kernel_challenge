"""Check V99's bounded comparison against two fully verified synthetic proposals."""

import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v70_dflash_selector_tree"))
from cpu_check import Pointer, program


def sign(value):
    return int(value > 0) - int(value < 0)


def walk(tokens, parents, truth):
    node = 0
    path = [0]
    for target in truth:
        for child in range(node + 1, len(tokens)):
            if parents[child] == node and tokens[child] == target:
                node = child
                path.append(node)
                break
        else:
            break
    return path


def exhaustive_comparison():
    counts = Counter()
    for width in (2, 3, 4):
        chain_parents = tuple(range(-1, width - 1))
        for edges in product(*(range(node) for node in range(1, width))):
            parents = (-1, *edges)
            for labels in product(range(2), repeat=width - 1):
                tree = (99, *labels)
                for proposal in product(range(2), repeat=width - 1):
                    chain = (99, *proposal)
                    for truth in product(range(2), repeat=width):
                        chain_reward = len(walk(chain, chain_parents, truth))
                        tree_reward = len(walk(tree, parents, truth))
                        winner = sign(chain_reward - tree_reward)
                        for observed in (chain_reward, tree_reward):
                            seen = truth[:observed]
                            from_chain = len(walk(chain, chain_parents, seen))
                            from_tree = len(walk(tree, parents, seen))
                            assert sign(from_chain - from_tree) == winner
                            counts[width] += 1
    return counts


def make_tree(tokens, parents):
    width = len(tokens)
    depths = np.zeros(width, np.int64)
    masks = np.ones_like(depths)
    for node in range(1, width):
        parent = parents[node]
        depths[node] = depths[parent] + 1
        masks[node] = masks[parent] | (1 << node)
    return np.array([tokens, depths, parents, masks], dtype=np.int64)


def run_kernels(checkout):
    path = checkout / "python/sglang/kernels/ops/speculative/dflash_tree.py"
    pid, (select, accept, update) = program(path, ("_select_chain_or_tree", "_accept_tree", "_update_tree_advantage"))
    rng = np.random.default_rng(8751)
    counts = Counter()
    for width in range(2, 9):
        block = 1 << (width - 1).bit_length()
        chain_parents = np.arange(-1, width - 1, dtype=np.int64)
        for batch in (1, 4, 7):
            for case in range(12):
                original = np.zeros((4, batch, width), np.int64)
                chains = rng.integers(0, 3, size=(batch, width))
                scores = rng.integers(-width, width + 1, size=batch, dtype=np.int32)
                scores[case % batch] = (-width, 0, width)[case % 3]
                true_paths = rng.integers(0, 3, size=(batch, width))
                old_scores = scores.copy()
                for row in range(batch):
                    parents = np.array([-1] + [rng.integers(i) for i in range(1, width)])
                    tokens = rng.integers(0, 3, size=width)
                    tokens[0] = chains[row, 0] = 123 + row
                    if case % 4 == 0:
                        true_paths[row, :-1] = chains[row, 1:]
                    if case % 4 == 1:
                        parents = chain_parents.copy()
                        tokens = chains[row].copy()
                    if case % 4 == 2:
                        parents[1:] = 0
                        tokens[1:] = true_paths[row, 0]
                    original[:, row] = make_tree(tokens, parents)

                chosen = np.full_like(original, -42)
                predictions = np.full((batch, width), -901, dtype=np.int64)
                output = np.empty_like(predictions)
                order = np.empty_like(predictions)
                prefix = rng.integers(0, 131072, size=batch, dtype=np.int64)
                lengths = np.empty(batch, dtype=np.int32)
                bonus = np.empty_like(prefix)
                new_lengths = np.empty_like(prefix)
                new_scores = np.empty_like(scores)
                votes = []
                for row in range(batch):
                    pid[0] = row
                    select(*map(Pointer, (original, chains, scores, chosen)), batch * width, width, block)
                    linear = make_tree(chains[row], chain_parents)
                    expect = linear if scores[row] > 0 else original[:, row]
                    np.testing.assert_array_equal(chosen[:, row], expect)
                    truth = true_paths[row]
                    path_nodes = walk(expect[0], expect[2], truth)
                    tree_reward = len(walk(original[0, row], original[2, row], truth))
                    chain_reward = len(walk(chains[row], chain_parents, truth))
                    vote = sign(chain_reward - tree_reward)
                    votes.append(vote)
                    for depth, node in enumerate(path_nodes):
                        predictions[row, node] = truth[depth]
                    accept(*map(Pointer, (chosen[0], chosen[2], predictions, prefix, lengths,
                                          bonus, output, new_lengths, order)), width, block)
                    observed = len(path_nodes)
                    assert lengths[row] == observed and new_lengths[row] == prefix[row] + observed
                    assert bonus[row] == truth[observed - 1]
                    np.testing.assert_array_equal(output[row, :observed], truth[:observed])
                    np.testing.assert_array_equal(order[row, :observed], path_nodes)
                    np.testing.assert_array_equal(np.sort(order[row]), np.arange(width))
                    output[row, observed:] = 2147483647
                    counts[f"vote_{vote}"] += 1
                    counts["full_budget"] += int(observed == width)
                for row, vote in enumerate(votes):
                    pid[0] = row
                    update(*map(Pointer, (original[0], original[2], chains, output, lengths,
                                          scores, new_scores)), width, block)
                    previous = int(old_scores[row])
                    expected = np.clip(previous - sign(previous) + 2 * vote, -width, width)
                    assert new_scores[row] == expected
                np.testing.assert_array_equal(old_scores, scores)
                counts["batches"] += 1
                counts["requests"] += batch
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    args = parser.parse_args()
    exhaustive = exhaustive_comparison()
    print(json.dumps(dict(exhaustive_by_width=exhaustive, exhaustive_total=sum(exhaustive.values()),
                          production_kernels=run_kernels(args.checkout), device="CPU"), indent=2))
