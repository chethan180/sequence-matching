"""The blossom matcher must return a maximum-weight matching. Checked against exhaustive search on random graphs."""
import random
import time
import unittest
from smp.matching import best_pairs, max_weight_matching


def brute_force(n, edges):
    """Maximum total weight by exhaustive search over the lowest unmatched vertex (exact for small n)."""
    adj = [[] for _ in range(n)]
    for i, j, w in edges:
        adj[i].append((j, w))
        adj[j].append((i, w))
    memo = {}

    def best(mask):
        if mask == 0:
            return 0
        if mask in memo:
            return memo[mask]
        low = (mask & -mask).bit_length() - 1
        rest = mask & ~(1 << low)
        top = best(rest)
        for j, w in adj[low]:
            if rest >> j & 1:
                top = max(top, w + best(rest & ~(1 << j)))
        memo[mask] = top
        return top
    return best((1 << n) - 1)


def check(test, n, edges):
    mate = max_weight_matching(edges)
    weight = {}
    for i, j, w in edges:
        weight[(i, j)] = weight[(j, i)] = max(w, weight.get((i, j), w))
    total = 0
    for v, u in enumerate(mate):
        if u >= 0:
            test.assertEqual(mate[u], v)                 # symmetric
            test.assertIn((v, u), weight)                # only real edges
            if u > v:
                total += weight[(v, u)]
    test.assertEqual(total, brute_force(n, edges))


class MatchingTests(unittest.TestCase):
    def test_small_known_cases(self):
        self.assertEqual(max_weight_matching([]), [])
        self.assertEqual(max_weight_matching([(0, 1, 1)]), [1, 0])
        # the problem statement's example: A-B 90, C-D 5, A-D 65, B-C 65 -> A-D and B-C
        self.assertEqual(sorted(best_pairs([(90, 'A', 'B'), (5, 'C', 'D'), (65, 'A', 'D'), (65, 'B', 'C')])),
                         [['A', 'D'], ['B', 'C']])
        # odd cycle (blossom) with a pendant edge
        check(self, 6, [(0, 1, 5), (1, 2, 5), (2, 0, 5), (2, 3, 4), (3, 4, 6), (4, 5, 1)])

    def test_random_graphs_against_exhaustive_search(self):
        rng = random.Random(0)
        for trial in range(3000):
            n = rng.randint(2, 13)
            p = rng.choice([0.15, 0.3, 0.6, 1.0])
            wmax = rng.choice([1, 3, 10, 1000, 10 ** 12])
            edges = [(i, j, rng.randint(1, wmax)) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
            check(self, n, edges)

    def test_tiered_weights_give_maximum_cardinality_first(self):
        rng = random.Random(1)
        for trial in range(300):
            n = rng.randint(4, 12)
            raw = [(i, j, rng.randint(1, 10 ** 6)) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.35]
            if not raw:
                continue
            big = 10 ** 12
            mate = max_weight_matching([(i, j, big + w) for i, j, w in raw])
            card = sum(1 for v, u in enumerate(mate) if u > v)
            self.assertEqual(card, brute_force(n, [(i, j, 1) for i, j, _ in raw]))

    def test_size_of_a_real_day_is_fast(self):
        rng = random.Random(2)
        n = 200
        edges = [(i, j, rng.randint(1, 10 ** 15)) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.02]
        start = time.perf_counter()
        mate = max_weight_matching(edges)
        self.assertLess(time.perf_counter() - start, 2.0)
        self.assertTrue(any(u >= 0 for u in mate))


if __name__ == '__main__':
    unittest.main()
