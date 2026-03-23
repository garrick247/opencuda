"""
test_dominance.py — v0.9 Dominance Infrastructure Tests.

Verifies compute_dominators(), dominates(), immediate_dominator(),
build_dom_tree(), and kernel_stats() across synthetic and real CFGs:

  Group 1: Straight-line CFG — each block dominated by all predecessors
  Group 2: Diamond CFG — join point dominated only by entry + itself
  Group 3: Simple loop — header dominated by entry; body by header
  Group 4: Nested loops — inner header dominated by outer header
  Group 5: Unreachable blocks — isolated block dominates only itself
  Group 6: Multi-exit loop — dominator still correct at exit
  Group 7: dominates() API — symmetry, reflexivity, transitivity
  Group 8: immediate_dominator() — idom queries on known CFGs
  Group 9: build_dom_tree() — children structure on known CFGs
  Group 10: kernel_stats() — block/loop/reachable counts on real kernels
  Group 11: Real kernels — dominance properties from parsed source
"""

import pytest
from pathlib import Path

from opencuda.ir.nodes import (
    BasicBlock, Kernel, KernelParam, Module, Value, Const,
    BrTerm, CondBrTerm, RetTerm, BinInst, CmpInst, BinOp, CmpOp,
)
from opencuda.ir.types import INT32
from opencuda.ir.dominator import (
    compute_dominators, dominates, immediate_dominator,
    build_dom_tree, kernel_stats,
)
from opencuda.frontend.preprocess import preprocess
from opencuda.frontend.parser import parse
from opencuda.ir.optimize import optimize

TESTS_DIR = Path(__file__).parent.parent.parent / 'tests'
ALL_CU_FILES = sorted(f for f in TESTS_DIR.glob('*.cu') if not f.name.startswith('gpu_'))


# ---------------------------------------------------------------------------
# CFG construction helpers
# ---------------------------------------------------------------------------

def _make_kernel(name: str, *block_labels: str) -> Kernel:
    """Create a kernel with empty blocks (no instructions, no terminator)."""
    k = Kernel(name=name, params=[])
    for lbl in block_labels:
        k.blocks.append(BasicBlock(label=lbl))
    return k


def _set_br(kernel: Kernel, src: str, tgt: str) -> None:
    """Set an unconditional branch from src block to tgt block."""
    _get(kernel, src).terminator = BrTerm(target=tgt)


def _set_ret(kernel: Kernel, src: str) -> None:
    _get(kernel, src).terminator = RetTerm()


def _set_condbr(kernel: Kernel, src: str, cond_val: Value,
                true_bb: str, false_bb: str) -> None:
    _get(kernel, src).terminator = CondBrTerm(cond=cond_val,
                                               true_bb=true_bb,
                                               false_bb=false_bb)


def _get(kernel: Kernel, label: str) -> BasicBlock:
    for bb in kernel.blocks:
        if bb.label == label:
            return bb
    raise KeyError(label)


def _dummy_cond(kernel: Kernel) -> Value:
    """Return a dummy predicate Value for CondBrTerm."""
    return kernel.new_value('cond', INT32)


# ---------------------------------------------------------------------------
# Group 1: Straight-line CFG  A → B → C → D
# ---------------------------------------------------------------------------

@pytest.fixture
def straight_kernel():
    """entry_1 → bb_2 → bb_3 → bb_4"""
    k = _make_kernel('straight', 'entry_1', 'bb_2', 'bb_3', 'bb_4')
    _set_br(k, 'entry_1', 'bb_2')
    _set_br(k, 'bb_2', 'bb_3')
    _set_br(k, 'bb_3', 'bb_4')
    _set_ret(k, 'bb_4')
    return k


def test_straight_entry_dominates_all(straight_kernel):
    dom = compute_dominators(straight_kernel)
    for bb in straight_kernel.blocks:
        assert dominates(dom, 'entry_1', bb.label), \
            f'entry_1 should dominate {bb.label}'


def test_straight_each_dominates_successors(straight_kernel):
    dom = compute_dominators(straight_kernel)
    chain = ['entry_1', 'bb_2', 'bb_3', 'bb_4']
    for i, a in enumerate(chain):
        for b in chain[i:]:
            assert dominates(dom, a, b), f'{a} should dominate {b}'


def test_straight_no_backward_dominance(straight_kernel):
    dom = compute_dominators(straight_kernel)
    chain = ['entry_1', 'bb_2', 'bb_3', 'bb_4']
    for i, a in enumerate(chain):
        for b in chain[:i]:
            assert not dominates(dom, a, b), \
                f'{a} should NOT dominate {b} (goes backward)'


def test_straight_dom_sets(straight_kernel):
    dom = compute_dominators(straight_kernel)
    assert dom['entry_1'] == frozenset({'entry_1'})
    assert dom['bb_2']    == frozenset({'entry_1', 'bb_2'})
    assert dom['bb_3']    == frozenset({'entry_1', 'bb_2', 'bb_3'})
    assert dom['bb_4']    == frozenset({'entry_1', 'bb_2', 'bb_3', 'bb_4'})


# ---------------------------------------------------------------------------
# Group 2: Diamond CFG  entry → {A, B} → merge
# ---------------------------------------------------------------------------

@pytest.fixture
def diamond_kernel():
    """entry_1 → (bb_2, bb_3) → bb_4"""
    k = _make_kernel('diamond', 'entry_1', 'bb_2', 'bb_3', 'bb_4')
    cond = _dummy_cond(k)
    _set_condbr(k, 'entry_1', cond, 'bb_2', 'bb_3')
    _set_br(k, 'bb_2', 'bb_4')
    _set_br(k, 'bb_3', 'bb_4')
    _set_ret(k, 'bb_4')
    return k


def test_diamond_entry_dominates_all(diamond_kernel):
    dom = compute_dominators(diamond_kernel)
    for bb in diamond_kernel.blocks:
        assert dominates(dom, 'entry_1', bb.label)


def test_diamond_side_does_not_dominate_merge(diamond_kernel):
    dom = compute_dominators(diamond_kernel)
    # Neither branch arm dominates the merge point
    assert not dominates(dom, 'bb_2', 'bb_4')
    assert not dominates(dom, 'bb_3', 'bb_4')


def test_diamond_side_does_not_dominate_other_side(diamond_kernel):
    dom = compute_dominators(diamond_kernel)
    assert not dominates(dom, 'bb_2', 'bb_3')
    assert not dominates(dom, 'bb_3', 'bb_2')


def test_diamond_merge_dom_set(diamond_kernel):
    dom = compute_dominators(diamond_kernel)
    # merge is only dominated by entry and itself
    assert dom['bb_4'] == frozenset({'entry_1', 'bb_4'})


def test_diamond_branch_dom_sets(diamond_kernel):
    dom = compute_dominators(diamond_kernel)
    assert dom['bb_2'] == frozenset({'entry_1', 'bb_2'})
    assert dom['bb_3'] == frozenset({'entry_1', 'bb_3'})


# ---------------------------------------------------------------------------
# Group 3: Simple loop  entry → header → body → header (back edge)
#          header → exit
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_loop_kernel():
    """
    entry_1 → header_2 → body_3 → header_2 (back edge)
                        ↘ exit_4
    """
    k = _make_kernel('simple_loop', 'entry_1', 'header_2', 'body_3', 'exit_4')
    cond = _dummy_cond(k)
    _set_br(k, 'entry_1', 'header_2')
    _set_condbr(k, 'header_2', cond, 'body_3', 'exit_4')
    _set_br(k, 'body_3', 'header_2')  # back edge
    _set_ret(k, 'exit_4')
    return k


def test_loop_entry_dominates_all(simple_loop_kernel):
    dom = compute_dominators(simple_loop_kernel)
    for bb in simple_loop_kernel.blocks:
        assert dominates(dom, 'entry_1', bb.label)


def test_loop_header_dominates_body(simple_loop_kernel):
    dom = compute_dominators(simple_loop_kernel)
    assert dominates(dom, 'header_2', 'body_3')


def test_loop_header_dominates_exit(simple_loop_kernel):
    dom = compute_dominators(simple_loop_kernel)
    assert dominates(dom, 'header_2', 'exit_4')


def test_loop_body_does_not_dominate_header(simple_loop_kernel):
    dom = compute_dominators(simple_loop_kernel)
    # back edge does not create dominance from body to header
    assert not dominates(dom, 'body_3', 'header_2')


def test_loop_body_does_not_dominate_exit(simple_loop_kernel):
    dom = compute_dominators(simple_loop_kernel)
    assert not dominates(dom, 'body_3', 'exit_4')


def test_loop_header_dom_set(simple_loop_kernel):
    dom = compute_dominators(simple_loop_kernel)
    assert dom['header_2'] == frozenset({'entry_1', 'header_2'})


def test_loop_body_dom_set(simple_loop_kernel):
    dom = compute_dominators(simple_loop_kernel)
    assert dom['body_3'] == frozenset({'entry_1', 'header_2', 'body_3'})


# ---------------------------------------------------------------------------
# Group 4: Nested loops
#   entry → outer_header → inner_header → inner_body → inner_header (back)
#                         ↘ outer_exit     inner_header → outer_body → outer_header (back)
# ---------------------------------------------------------------------------

@pytest.fixture
def nested_loop_kernel():
    """
    entry_1 → outer_hdr_2 → inner_hdr_3 → inner_body_4 → inner_hdr_3
              outer_hdr_2 → outer_exit_5
              inner_hdr_3 → outer_body_6 → outer_hdr_2
    """
    k = _make_kernel('nested_loop',
                     'entry_1', 'outer_hdr_2', 'inner_hdr_3',
                     'inner_body_4', 'outer_body_6', 'outer_exit_5')
    c1 = _dummy_cond(k)
    c2 = _dummy_cond(k)
    c3 = _dummy_cond(k)
    _set_br(k, 'entry_1', 'outer_hdr_2')
    _set_condbr(k, 'outer_hdr_2', c1, 'inner_hdr_3', 'outer_exit_5')
    _set_condbr(k, 'inner_hdr_3', c2, 'inner_body_4', 'outer_body_6')
    _set_br(k, 'inner_body_4', 'inner_hdr_3')  # inner back edge
    _set_br(k, 'outer_body_6', 'outer_hdr_2')  # outer back edge
    _set_ret(k, 'outer_exit_5')
    return k


def test_nested_entry_dominates_all(nested_loop_kernel):
    dom = compute_dominators(nested_loop_kernel)
    for bb in nested_loop_kernel.blocks:
        assert dominates(dom, 'entry_1', bb.label)


def test_nested_outer_hdr_dominates_inner(nested_loop_kernel):
    dom = compute_dominators(nested_loop_kernel)
    assert dominates(dom, 'outer_hdr_2', 'inner_hdr_3')
    assert dominates(dom, 'outer_hdr_2', 'inner_body_4')
    assert dominates(dom, 'outer_hdr_2', 'outer_body_6')


def test_nested_inner_hdr_dominates_inner_body(nested_loop_kernel):
    dom = compute_dominators(nested_loop_kernel)
    assert dominates(dom, 'inner_hdr_3', 'inner_body_4')


def test_nested_inner_hdr_not_dominate_outer_body(nested_loop_kernel):
    dom = compute_dominators(nested_loop_kernel)
    # outer_body_6 has two predecessors: inner_hdr_3 (the condbr false path)
    # So inner_hdr_3 DOES dominate outer_body_6 in this topology
    # But inner_body_4 does NOT dominate outer_body_6
    assert not dominates(dom, 'inner_body_4', 'outer_body_6')


def test_nested_body_not_dominate_headers(nested_loop_kernel):
    dom = compute_dominators(nested_loop_kernel)
    assert not dominates(dom, 'inner_body_4', 'outer_hdr_2')
    assert not dominates(dom, 'inner_body_4', 'inner_hdr_3')
    assert not dominates(dom, 'outer_body_6', 'outer_hdr_2')


# ---------------------------------------------------------------------------
# Group 5: Unreachable block
# ---------------------------------------------------------------------------

@pytest.fixture
def unreachable_kernel():
    """entry_1 → bb_2; bb_orphan_3 (not reachable)"""
    k = _make_kernel('unreachable', 'entry_1', 'bb_2', 'bb_orphan_3')
    _set_br(k, 'entry_1', 'bb_2')
    _set_ret(k, 'bb_2')
    _set_ret(k, 'bb_orphan_3')  # no predecessor → unreachable
    return k


def test_unreachable_self_dominates(unreachable_kernel):
    dom = compute_dominators(unreachable_kernel)
    # Unreachable block dominates itself
    assert dominates(dom, 'bb_orphan_3', 'bb_orphan_3')


def test_unreachable_not_dominated_by_entry(unreachable_kernel):
    dom = compute_dominators(unreachable_kernel)
    # Entry does NOT dominate an unreachable block
    # (no path from entry → orphan, so trivially, but algorithm gives {orphan})
    assert not dominates(dom, 'entry_1', 'bb_orphan_3')


def test_unreachable_not_dominate_reachable(unreachable_kernel):
    dom = compute_dominators(unreachable_kernel)
    assert not dominates(dom, 'bb_orphan_3', 'entry_1')
    assert not dominates(dom, 'bb_orphan_3', 'bb_2')


# ---------------------------------------------------------------------------
# Group 6: Multi-exit loop  (loop with break)
# ---------------------------------------------------------------------------

@pytest.fixture
def break_loop_kernel():
    """
    entry_1 → header_2 → body_3 → check_4 → header_2 (back) or exit_5
                        ↘ early_exit_6
    """
    k = _make_kernel('break_loop',
                     'entry_1', 'header_2', 'body_3',
                     'check_4', 'exit_5', 'early_exit_6')
    c1 = _dummy_cond(k)
    c2 = _dummy_cond(k)
    c3 = _dummy_cond(k)
    _set_br(k, 'entry_1', 'header_2')
    _set_condbr(k, 'header_2', c1, 'body_3', 'exit_5')
    _set_condbr(k, 'body_3', c2, 'check_4', 'early_exit_6')
    _set_condbr(k, 'check_4', c3, 'header_2', 'exit_5')
    _set_ret(k, 'exit_5')
    _set_ret(k, 'early_exit_6')
    return k


def test_break_loop_header_dominates_body(break_loop_kernel):
    dom = compute_dominators(break_loop_kernel)
    assert dominates(dom, 'header_2', 'body_3')
    assert dominates(dom, 'header_2', 'check_4')


def test_break_loop_entry_dominates_all_exits(break_loop_kernel):
    dom = compute_dominators(break_loop_kernel)
    assert dominates(dom, 'entry_1', 'exit_5')
    assert dominates(dom, 'entry_1', 'early_exit_6')


def test_break_loop_body_not_dominate_exit(break_loop_kernel):
    dom = compute_dominators(break_loop_kernel)
    # exit_5 has two predecessors: header_2 and check_4; body does not dominate it
    assert not dominates(dom, 'body_3', 'exit_5')


# ---------------------------------------------------------------------------
# Group 7: dominates() API properties
# ---------------------------------------------------------------------------

def test_dominates_reflexive(straight_kernel):
    dom = compute_dominators(straight_kernel)
    for bb in straight_kernel.blocks:
        assert dominates(dom, bb.label, bb.label), \
            f'{bb.label} must dominate itself'


def test_dominates_antisymmetric(straight_kernel):
    dom = compute_dominators(straight_kernel)
    # For a→b in straight line, if a dom b and b dom a, then a == b
    labels = [bb.label for bb in straight_kernel.blocks]
    for a in labels:
        for b in labels:
            if a != b and dominates(dom, a, b):
                assert not dominates(dom, b, a), \
                    f'antisymmetry violated: {a} and {b} mutually dominate'


def test_dominates_transitive(straight_kernel):
    dom = compute_dominators(straight_kernel)
    labels = [bb.label for bb in straight_kernel.blocks]
    for a in labels:
        for b in labels:
            for c in labels:
                if dominates(dom, a, b) and dominates(dom, b, c):
                    assert dominates(dom, a, c), \
                        f'transitivity violated: {a} dom {b} dom {c} but not {a} dom {c}'


def test_dominates_unknown_label(straight_kernel):
    dom = compute_dominators(straight_kernel)
    # Unknown labels should return False safely
    assert not dominates(dom, 'entry_1', 'nonexistent')
    assert not dominates(dom, 'nonexistent', 'entry_1')


# ---------------------------------------------------------------------------
# Group 8: immediate_dominator() queries
# ---------------------------------------------------------------------------

def test_idom_entry_is_none(straight_kernel):
    dom = compute_dominators(straight_kernel)
    labels = [bb.label for bb in straight_kernel.blocks]
    idom = immediate_dominator(dom, labels, 'entry_1')
    assert idom is None


def test_idom_straight_line(straight_kernel):
    dom = compute_dominators(straight_kernel)
    labels = [bb.label for bb in straight_kernel.blocks]
    assert immediate_dominator(dom, labels, 'bb_2') == 'entry_1'
    assert immediate_dominator(dom, labels, 'bb_3') == 'bb_2'
    assert immediate_dominator(dom, labels, 'bb_4') == 'bb_3'


def test_idom_diamond_merge(diamond_kernel):
    dom = compute_dominators(diamond_kernel)
    labels = [bb.label for bb in diamond_kernel.blocks]
    # merge (bb_4) idom is entry_1 (only shared strict dominator)
    assert immediate_dominator(dom, labels, 'bb_4') == 'entry_1'
    # branch arms' idom is entry_1
    assert immediate_dominator(dom, labels, 'bb_2') == 'entry_1'
    assert immediate_dominator(dom, labels, 'bb_3') == 'entry_1'


def test_idom_loop_header(simple_loop_kernel):
    dom = compute_dominators(simple_loop_kernel)
    labels = [bb.label for bb in simple_loop_kernel.blocks]
    # header's idom is entry_1 (single path in, back edge does not add dominators)
    assert immediate_dominator(dom, labels, 'header_2') == 'entry_1'
    assert immediate_dominator(dom, labels, 'body_3') == 'header_2'
    assert immediate_dominator(dom, labels, 'exit_4') == 'header_2'


# ---------------------------------------------------------------------------
# Group 9: build_dom_tree() structure
# ---------------------------------------------------------------------------

def test_dom_tree_straight_line(straight_kernel):
    dom = compute_dominators(straight_kernel)
    labels = [bb.label for bb in straight_kernel.blocks]
    tree = build_dom_tree(dom, labels)
    # entry_1 → [bb_2], bb_2 → [bb_3], bb_3 → [bb_4], bb_4 → []
    assert tree['entry_1'] == ['bb_2']
    assert tree['bb_2'] == ['bb_3']
    assert tree['bb_3'] == ['bb_4']
    assert tree['bb_4'] == []


def test_dom_tree_diamond(diamond_kernel):
    dom = compute_dominators(diamond_kernel)
    labels = [bb.label for bb in diamond_kernel.blocks]
    tree = build_dom_tree(dom, labels)
    # entry_1 dominates all; both arms and merge are its direct children
    assert set(tree['entry_1']) == {'bb_2', 'bb_3', 'bb_4'}
    assert tree['bb_2'] == []
    assert tree['bb_3'] == []
    assert tree['bb_4'] == []


def test_dom_tree_loop(simple_loop_kernel):
    dom = compute_dominators(simple_loop_kernel)
    labels = [bb.label for bb in simple_loop_kernel.blocks]
    tree = build_dom_tree(dom, labels)
    # entry → header; header → {body, exit}
    assert tree['entry_1'] == ['header_2']
    assert set(tree['header_2']) == {'body_3', 'exit_4'}
    assert tree['body_3'] == []
    assert tree['exit_4'] == []


def test_dom_tree_root_has_no_parent(straight_kernel):
    """Entry block must not appear as a child of any node."""
    dom = compute_dominators(straight_kernel)
    labels = [bb.label for bb in straight_kernel.blocks]
    tree = build_dom_tree(dom, labels)
    for lbl, children in tree.items():
        if lbl != 'entry_1':  # others can have children
            assert 'entry_1' not in children, \
                f'entry_1 appeared as child of {lbl}'


# ---------------------------------------------------------------------------
# Group 10: kernel_stats() on synthetic kernels
# ---------------------------------------------------------------------------

def test_stats_straight_no_loops(straight_kernel):
    s = kernel_stats(straight_kernel)
    assert s['block_count'] == 4
    assert s['reachable_count'] == 4
    assert s['loop_count'] == 0
    assert not s['has_branches']


def test_stats_diamond_no_loops(diamond_kernel):
    s = kernel_stats(diamond_kernel)
    assert s['block_count'] == 4
    assert s['loop_count'] == 0
    assert s['has_branches']


def test_stats_simple_loop(simple_loop_kernel):
    s = kernel_stats(simple_loop_kernel)
    assert s['loop_count'] == 1


def test_stats_nested_loops(nested_loop_kernel):
    s = kernel_stats(nested_loop_kernel)
    assert s['loop_count'] == 2


def test_stats_unreachable(unreachable_kernel):
    s = kernel_stats(unreachable_kernel)
    assert s['block_count'] == 3
    assert s['reachable_count'] == 2  # orphan not reachable


def test_stats_empty_kernel():
    k = Kernel(name='empty', params=[])
    s = kernel_stats(k)
    assert s['block_count'] == 0
    assert s['loop_count'] == 0


# ---------------------------------------------------------------------------
# Group 11: Real kernels — parse and verify dominance properties
# ---------------------------------------------------------------------------

def _parse_opt(path: Path):
    src = path.read_text()
    return optimize(parse(preprocess(src)))


@pytest.mark.parametrize('cu_file', ALL_CU_FILES, ids=lambda p: p.stem)
def test_real_kernel_entry_dominates_all_reachable(cu_file):
    """Entry block must dominate every reachable block in every real kernel."""
    mod = _parse_opt(cu_file)
    for kernel in mod.kernels:
        if not kernel.blocks:
            continue
        dom = compute_dominators(kernel)
        entry = kernel.blocks[0].label
        # BFS reachability
        labels = {bb.label for bb in kernel.blocks}
        reachable = set()
        work = [entry]
        while work:
            lbl = work.pop()
            if lbl in reachable:
                continue
            reachable.add(lbl)
            bb_map = {bb.label: bb for bb in kernel.blocks}
            if lbl in bb_map:
                t = bb_map[lbl].terminator
                from opencuda.ir.nodes import BrTerm, CondBrTerm
                if isinstance(t, BrTerm) and t.target in labels:
                    work.append(t.target)
                elif isinstance(t, CondBrTerm):
                    for tgt in (t.true_bb, t.false_bb):
                        if tgt in labels:
                            work.append(tgt)
        for lbl in reachable:
            assert dominates(dom, entry, lbl), \
                f'{kernel.name}: entry does not dominate {lbl}'


@pytest.mark.parametrize('cu_file', ALL_CU_FILES, ids=lambda p: p.stem)
def test_real_kernel_stats_consistent(cu_file):
    """block_count >= reachable_count, loop_count >= 0."""
    mod = _parse_opt(cu_file)
    for kernel in mod.kernels:
        s = kernel_stats(kernel)
        assert s['block_count'] >= s['reachable_count']
        assert s['loop_count'] >= 0
        assert s['block_count'] >= 0


@pytest.mark.parametrize('cu_file', ALL_CU_FILES, ids=lambda p: p.stem)
def test_real_kernel_idom_chain_leads_to_entry(cu_file):
    """Following idom pointers from any reachable block always reaches entry."""
    mod = _parse_opt(cu_file)
    for kernel in mod.kernels:
        if not kernel.blocks:
            continue
        dom = compute_dominators(kernel)
        labels = [bb.label for bb in kernel.blocks]
        entry = labels[0]
        for bb in kernel.blocks:
            lbl = bb.label
            visited = {lbl}
            cur = lbl
            while True:
                idom = immediate_dominator(dom, labels, cur)
                if idom is None:
                    # Reached entry or isolated block
                    assert cur == entry or cur not in {b.label for b in kernel.blocks[:1]}, \
                        f'{kernel.name}: idom chain from {lbl} ended at {cur} (not entry)'
                    break
                assert idom not in visited, \
                    f'{kernel.name}: idom cycle detected at {idom}'
                visited.add(idom)
                cur = idom
                if cur == entry:
                    break


@pytest.mark.parametrize('cu_file', ALL_CU_FILES, ids=lambda p: p.stem)
def test_real_kernel_dom_tree_covers_all_blocks(cu_file):
    """Every block appears exactly once in the dom tree (as node, not root)."""
    mod = _parse_opt(cu_file)
    for kernel in mod.kernels:
        if not kernel.blocks:
            continue
        dom = compute_dominators(kernel)
        labels = [bb.label for bb in kernel.blocks]
        tree = build_dom_tree(dom, labels)

        # Count how many times each non-root block appears as a child
        child_count: dict[str, int] = {lbl: 0 for lbl in labels}
        for parent_children in tree.values():
            for child in parent_children:
                child_count[child] = child_count.get(child, 0) + 1

        entry = labels[0]
        for lbl in labels:
            if lbl == entry:
                assert child_count[lbl] == 0, \
                    f'{kernel.name}: entry appears as child'
            else:
                assert child_count[lbl] == 1, \
                    f'{kernel.name}: {lbl} appears {child_count[lbl]} times in tree'
