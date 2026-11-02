"""Benchmark instances."""

from .instance import SCPRInstance


def small_4x4():
    return SCPRInstance([0, 1, 2, 3], [0, 1, 2, 3],
                        [({0, 1}, {0, 1}), ({1, 2}, {1, 2}), ({2, 3}, {2, 3}), ({0, 3}, {0})],
                        "Small_4x4")


def reduction_demo():
    return SCPRInstance([0, 1, 2, 3, 4], ["r0", "r1", "r2", "r3"],
                        [({4}, {"r2"}), ({0, 1, 2}, {"r0", "r1"}),
                         ({1, 2, 3}, {"r1", "r2"}), ({0, 3}, {"r3"})],
                        "Reduction_Demo")


def thesis_1_5_2():
    """Example 1.5.2 of Babatunde (2025): 16 lights, 8 batteries."""
    return SCPRInstance(list(range(16)), list(range(8)),
                        [(set(range(0, 8)), {0, 1, 2, 3}), (set(range(8, 16)), {2, 4, 6}),
                         (set(range(0, 4)), {0, 5}), ({8, 11, 14}, {7})],
                        "Thesis_1.5.2")


ALL = (small_4x4, reduction_demo, thesis_1_5_2)
