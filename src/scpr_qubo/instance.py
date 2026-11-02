"""SCPR instance (U, R, E), Definition 1.5.3 of Babatunde (2025).

A pair (A_i, R_i) activates when every reason in R_i is selected. Coverage is
at-least-one: an element may be covered by several activated pairs.
"""


class SCPRInstance:
    def __init__(self, U, R, E, name="SCPR"):
        self.U = list(U)
        self.R = list(R)
        self.E = [(set(A), set(Ri)) for A, Ri in E]
        self.name = name

    def __repr__(self):
        return f"SCPR({self.name}): |U|={len(self.U)}, |R|={len(self.R)}, |E|={len(self.E)}"

    def check_coverage(self, S):
        """True iff the pairs activated by S cover U."""
        if not self.U:
            return True
        covered = set()
        for A, Ri in self.E:
            if Ri.issubset(S):
                covered |= A
        return covered == set(self.U)
