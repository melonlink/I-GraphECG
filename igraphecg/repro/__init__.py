"""Reproduction layer: the one place that knows where things are.

Every published script resolves its inputs and outputs through here rather than naming a
directory itself. That is what lets the code archive be a subset selection instead of a
string rewrite: the repository and the archive differ in one small file, `repro.yaml`, and
in nothing else.

The rule this enforces, and the reason it exists: no shipped source file may name a path
outside the roots declared below. Hardcoded output paths -- several of them pointing into
frozen paper-submission directories, one of them an absolute path to the author's machine --
are what made every revision bump break the pipeline and what forced the archive builder to
rewrite path strings, which in turn manufactured dead `paper/...` paths in the deposit.
"""
from igraphecg.repro import lineage
from igraphecg.repro.roots import (data, outputs, paper, processed, repo, runs,
                                   outdir, describe)

__all__ = ["repo", "outputs", "runs", "data", "processed", "paper", "outdir",
           "describe", "lineage"]
