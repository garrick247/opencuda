"""
Dominance computation for OpenCUDA IR.

Uses iterative dataflow (Cooper et al. simple algorithm):
  - dom[entry] = {entry}
  - dom[b]     = {b} ∪ (∩ dom[p] for all predecessors p)
  - Iterate until fixpoint.

Time complexity: O(n² · iterations), typically 2–3 passes on reducible CFGs.
Adequate for the small kernels OpenCUDA handles (typically < 30 blocks).

Public API
----------
compute_dominators(kernel)     → dict[label, frozenset[label]]
dominates(dom, a, b)           → bool
immediate_dominator(dom, labels, b) → Optional[str]
build_dom_tree(dom, labels)    → dict[label, list[label]]
kernel_stats(kernel)           → dict with block_count, reachable_count, loop_count
"""

from __future__ import annotations
from typing import Optional
from ..ir.nodes import Kernel, BasicBlock, BrTerm, CondBrTerm, RetTerm


# ---------------------------------------------------------------------------
# Internal CFG helper (duplicated locally to avoid circular imports)
# ---------------------------------------------------------------------------

def _bb_successors(bb: BasicBlock) -> list[str]:
    t = bb.terminator
    if isinstance(t, BrTerm):
        return [t.target]
    if isinstance(t, CondBrTerm):
        return [t.true_bb, t.false_bb]
    return []


# ---------------------------------------------------------------------------
# Core dominance algorithm
# ---------------------------------------------------------------------------

def compute_dominators(kernel: Kernel) -> dict[str, frozenset[str]]:
    """Return dom[label] = frozenset of all block labels that dominate label.

    A block A dominates block B if every path from the entry to B passes
    through A.  Every block dominates itself.

    Unreachable blocks get dom = {block} (they dominate only themselves).
    """
    if not kernel.blocks:
        return {}

    labels = [bb.label for bb in kernel.blocks]
    entry = labels[0]
    label_set = frozenset(labels)

    # Build predecessor map
    pred_map: dict[str, list[str]] = {lbl: [] for lbl in labels}
    for bb in kernel.blocks:
        for succ in _bb_successors(bb):
            if succ in pred_map:
                pred_map[succ].append(bb.label)

    # Compute reachable set first (BFS from entry).
    # Only reachable predecessors participate in dominator intersection;
    # unreachable predecessors would otherwise poison the dom sets of their
    # reachable successors (making dom[succ] = {succ} because the dead
    # predecessor's dom set shares no elements with the reachable dom sets).
    reachable: set[str] = set()
    work = [entry]
    while work:
        lbl = work.pop()
        if lbl in reachable:
            continue
        reachable.add(lbl)
        bb_map = {bb.label: bb for bb in kernel.blocks}
        if lbl in bb_map:
            for s in _bb_successors(bb_map[lbl]):
                if s in pred_map and s not in reachable:
                    work.append(s)

    # Pessimistic initialisation: every block dominated by all blocks.
    # Entry starts correct.
    dom: dict[str, frozenset[str]] = {lbl: label_set for lbl in labels}
    dom[entry] = frozenset({entry})

    # Iterate to fixpoint in block-list order (approximates RPO).
    # Only intersect over REACHABLE predecessors so that dead blocks do not
    # pollute the dominator sets of their reachable successors.
    changed = True
    while changed:
        changed = False
        for lbl in labels:
            if lbl == entry:
                continue
            # Reachable predecessors only
            reach_preds = [p for p in pred_map[lbl]
                           if p in reachable and p in dom]
            if not reach_preds:
                # Unreachable block: dominator set = just itself
                new_dom = frozenset({lbl})
            else:
                new_dom = dom[reach_preds[0]]
                for p in reach_preds[1:]:
                    new_dom = new_dom & dom[p]
                new_dom = new_dom | frozenset({lbl})
            if new_dom != dom[lbl]:
                dom[lbl] = new_dom
                changed = True

    return dom


def dominates(dom: dict[str, frozenset[str]], a: str, b: str) -> bool:
    """Return True if block ``a`` dominates block ``b``.

    Every block dominates itself.  ``a`` dominates ``b`` iff every path from
    the entry to ``b`` passes through ``a``.
    """
    if a == b:
        return True
    return a in dom.get(b, frozenset())


def immediate_dominator(dom: dict[str, frozenset[str]],
                        labels: list[str],
                        b: str) -> Optional[str]:
    """Return the immediate dominator (idom) of block ``b``, or None for entry.

    The idom of ``b`` is the unique strict dominator of ``b`` that is itself
    dominated by every other strict dominator of ``b`` — i.e. the one closest
    to ``b`` in the dominator tree.

    Algorithm: idom(b) is the strict dominator ``s`` such that
      strict_doms(b) ⊆ dom(s)
    which means s is dominated by all other strict dominators of b (it is the
    deepest / closest one).
    """
    strict_doms = dom.get(b, frozenset()) - {b}
    if not strict_doms:
        return None  # entry block (or isolated block) has no idom

    for candidate in strict_doms:
        if strict_doms <= dom.get(candidate, frozenset()):
            return candidate

    return None  # should not happen for valid CFGs


def build_dom_tree(dom: dict[str, frozenset[str]],
                   labels: list[str]) -> dict[str, list[str]]:
    """Build the dominator tree as parent → children adjacency dict.

    Returns ``children[label]`` = list of blocks whose immediate dominator is
    ``label``.  The entry block is the root (no parent).
    """
    children: dict[str, list[str]] = {lbl: [] for lbl in labels}
    for lbl in labels:
        idom = immediate_dominator(dom, labels, lbl)
        if idom is not None:
            children[idom].append(lbl)
    return children


# ---------------------------------------------------------------------------
# Benchmark / stats helper (Deliverable D)
# ---------------------------------------------------------------------------

def kernel_stats(kernel: Kernel) -> dict:
    """Return a dict of structural metrics for a kernel.

    Keys:
      block_count       — total basic blocks
      reachable_count   — blocks reachable from entry
      loop_count        — number of back edges (natural loops)
      has_branches      — True if any CondBrTerm present
    """
    if not kernel.blocks:
        return {
            'block_count': 0,
            'reachable_count': 0,
            'loop_count': 0,
            'has_branches': False,
        }

    labels = [bb.label for bb in kernel.blocks]
    label_set = set(labels)
    entry = labels[0]
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    # BFS reachability
    reachable: set[str] = set()
    work = [entry]
    while work:
        lbl = work.pop()
        if lbl in reachable:
            continue
        reachable.add(lbl)
        bb_map = {bb.label: bb for bb in kernel.blocks}
        if lbl in bb_map:
            for s in _bb_successors(bb_map[lbl]):
                if s in label_set and s not in reachable:
                    work.append(s)

    # Back edge count (loop count)
    loop_count = 0
    seen_edges: set[tuple[str, str]] = set()
    for bb in kernel.blocks:
        for succ in _bb_successors(bb):
            if succ not in label_to_idx:
                continue
            if label_to_idx[succ] < label_to_idx[bb.label]:
                edge = (bb.label, succ)
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    loop_count += 1

    has_branches = any(
        isinstance(bb.terminator, CondBrTerm) for bb in kernel.blocks
    )

    return {
        'block_count': len(labels),
        'reachable_count': len(reachable),
        'loop_count': loop_count,
        'has_branches': has_branches,
    }
