"""Discrimination tree indexing matching C LADR discrimw.c / discrimb.c.

A discrimination tree indexes terms by traversing them in pre-order and
storing paths in a trie. Two variants are provided:

- **DiscrimWild** (imperfect filter): All query variables are wildcards.
  Fast retrieval, but returned terms may not actually match — caller must
  verify with a full match/unification check. Used for forward subsumption
  and demodulation where false positives are cheap to discard.

- **DiscrimBind** (perfect filter): Query variables are distinguished and
  bound during retrieval, producing a substitution. Every returned term is
  a correct generalization. Used when substitutions are needed immediately.

Both variants support:
- Insert: add a (term → object) mapping
- Delete: remove a previously inserted mapping
- Retrieve generalizations: find stored terms that are more general than query

Thread safety: Uses ReadWriteLock from threading_guide — concurrent reads
during inference generation, exclusive writes during index updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from pyladr.core.substitution import Context, Trail
from pyladr.core.term import Term
from pyladr.threading_guide import make_rw_lock


# ── Node types matching C discrim.h ──────────────────────────────────────────

class _NodeType(IntEnum):
    """Discrimination tree node type (C: type field in struct discrim)."""
    DVARIABLE = 0   # variable node
    DRIGID = 1      # rigid symbol node


# ── Discrim Node ─────────────────────────────────────────────────────────────

@dataclass(slots=True)
class DiscrimNode:
    """A node in the discrimination tree (C: struct discrim).

    Internal nodes have children; leaf nodes have a data list.
    Siblings are stored in a sorted list (not a linked list as in C).
    A dict (_children_map) provides O(1) child lookup by (node_type, symbol).
    """

    symbol: int          # varnum for DVARIABLE, symnum for DRIGID
    node_type: _NodeType
    children: list[DiscrimNode] = field(default_factory=list)
    data: list[Any] = field(default_factory=list)
    _children_map: dict[tuple[int, int], DiscrimNode] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0 and len(self.data) > 0


# ── Path encoding ────────────────────────────────────────────────────────────

def _term_to_path(t: Term) -> list[tuple[_NodeType, int]]:
    """Encode a term as a pre-order path of (node_type, symbol) pairs.

    This matches the C traversal in discrim_bind_insert_rec / discrim_wild_insert_rec.
    Variables → (DVARIABLE, varnum_or_0), rigid → (DRIGID, symnum), then recurse args.
    """
    path: list[tuple[_NodeType, int]] = []
    _collect_path(t, path)
    return path


def _term_to_wild_path(t: Term) -> list[tuple[_NodeType, int]]:
    """Encode a term using wild convention: all variables → symbol=0."""
    path: list[tuple[_NodeType, int]] = []
    _collect_wild_path(t, path)
    return path


def _collect_path(t: Term, path: list[tuple[_NodeType, int]]) -> None:
    """Collect bind-variant path (variables distinguished by varnum)."""
    if t.is_variable:
        path.append((_NodeType.DVARIABLE, t.varnum))
    else:
        path.append((_NodeType.DRIGID, t.symnum))
        for a in t.args:
            _collect_path(a, path)


def _collect_wild_path(t: Term, path: list[tuple[_NodeType, int]]) -> None:
    """Collect wild-variant path (all variables → single wildcard)."""
    if t.is_variable:
        path.append((_NodeType.DVARIABLE, 0))
    else:
        path.append((_NodeType.DRIGID, t.symnum))
        for a in t.args:
            _collect_wild_path(a, path)


# ── Find or create child node ────────────────────────────────────────────────

def _find_child(parent: DiscrimNode, node_type: _NodeType, symbol: int) -> DiscrimNode | None:
    """Find a child node by type and symbol. O(1) via dict lookup."""
    return parent._children_map.get((node_type, symbol))


def _find_or_create_child(
    parent: DiscrimNode, node_type: _NodeType, symbol: int
) -> DiscrimNode:
    """Find or create a child node, maintaining sorted order.

    Sort order: DVARIABLE nodes first (sorted by symbol), then DRIGID (sorted by symbol).
    This matches C where variables always precede rigid symbols in sibling lists.
    Uses O(1) dict lookup for existing children; falls back to sorted insertion.
    """
    key = (node_type, symbol)
    existing = parent._children_map.get(key)
    if existing is not None:
        return existing

    new_node = DiscrimNode(symbol=symbol, node_type=node_type)
    parent._children_map[key] = new_node

    # Maintain sorted order in children list (DVARIABLE < DRIGID, then by symbol)
    children = parent.children
    lo, hi = 0, len(children)
    while lo < hi:
        mid = (lo + hi) >> 1
        c = children[mid]
        if (c.node_type, c.symbol) < key:
            lo = mid + 1
        else:
            hi = mid
    children.insert(lo, new_node)
    return new_node


# ══════════════════════════════════════════════════════════════════════════════
# Wild Discrimination Tree (imperfect filter)
# ══════════════════════════════════════════════════════════════════════════════


class DiscrimWild:
    """Wild discrimination tree matching C discrimw.c.

    All variables in inserted terms are treated as a single wildcard.
    Retrieval finds potential generalizations (imperfect: may have false positives).
    Thread-safe via ReadWriteLock.
    """

    __slots__ = ("_root", "_lock", "_size")

    def __init__(self) -> None:
        self._root = DiscrimNode(symbol=-1, node_type=_NodeType.DRIGID)
        self._lock = make_rw_lock()
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def insert(self, t: Term, obj: Any) -> None:
        """Insert a term→object mapping into the tree.

        Matches C discrim_wild_insert(). Traverses the term and tree
        simultaneously in a single pass (no intermediate path list).
        """
        with self._lock.write_lock():
            node = self._root
            # Inline pre-order traversal with direct tree navigation
            work = [t]
            while work:
                term = work.pop()
                if term.is_variable:
                    node = _find_or_create_child(node, _NodeType.DVARIABLE, 0)
                else:
                    node = _find_or_create_child(node, _NodeType.DRIGID, term.symnum)
                    # Push args right-to-left so left arg is processed first
                    for i in range(term.arity - 1, -1, -1):
                        work.append(term.args[i])
            node.data.append(obj)
            self._size += 1

    def delete(self, t: Term, obj: Any) -> bool:
        """Remove a term→object mapping. Returns True if found and removed.

        Matches C discrim_wild_delete(). Traverses the term and tree
        simultaneously in a single pass (no intermediate path list).
        """
        with self._lock.write_lock():
            # Navigate to leaf, recording (parent, child_key) for pruning
            node = self._root
            ancestors: list[tuple[DiscrimNode, tuple[int, int]]] = []
            work = [t]
            while work:
                term = work.pop()
                if term.is_variable:
                    ntype, sym = _NodeType.DVARIABLE, 0
                else:
                    ntype, sym = _NodeType.DRIGID, term.symnum
                    for i in range(term.arity - 1, -1, -1):
                        work.append(term.args[i])
                child = _find_child(node, ntype, sym)
                if child is None:
                    return False
                ancestors.append((node, (ntype, sym)))
                node = child

            # Remove object from leaf data
            try:
                node.data.remove(obj)
            except ValueError:
                return False

            self._size -= 1

            # Prune empty nodes bottom-up
            if not node.data and not node.children:
                for parent, key in reversed(ancestors):
                    child = parent._children_map.pop(key)
                    parent.children.remove(child)
                    if parent.data or parent.children:
                        break

            return True

    def retrieve_generalizations(self, query: Term) -> list[Any]:
        """Retrieve all stored objects whose terms generalize the query.

        This is an imperfect filter — some results may not actually be
        generalizations. The caller should verify with match().

        Matches C discrim_wild_retrieve_first/next() combined.
        """
        results: list[Any] = []
        with self._lock.read_lock():
            self._retrieve_gen_rec(query, self._root, results)
        return results

    def _retrieve_gen_rec(self, query: Term, node: DiscrimNode, results: list[Any]) -> None:
        """Recursive generalization retrieval matching C wild retrieval.

        At each position, a stored wildcard matches any query subterm,
        and a stored rigid symbol must match the query's rigid symbol exactly.
        Uses O(1) dict lookup for child nodes.
        """
        # Try wildcard child — O(1) lookup
        wild_child = node._children_map.get((_NodeType.DVARIABLE, 0))
        if wild_child is not None:
            if wild_child.data:
                results.extend(wild_child.data)

        if query.is_variable:
            return

        # Try rigid match — O(1) lookup
        rigid_child = node._children_map.get((_NodeType.DRIGID, query.symnum))
        if rigid_child is not None:
            if query.arity == 0:
                if rigid_child.data:
                    results.extend(rigid_child.data)
            else:
                self._retrieve_args_rec(query.args, 0, rigid_child, results)

    def _retrieve_args_rec(
        self,
        args: tuple[Term, ...],
        arg_idx: int,
        node: DiscrimNode,
        results: list[Any],
    ) -> None:
        """Recursively match arguments of a complex term. O(1) child lookup."""
        if arg_idx >= len(args):
            if node.data:
                results.extend(node.data)
            return

        arg = args[arg_idx]

        # Try wildcard — O(1) lookup
        wild_child = node._children_map.get((_NodeType.DVARIABLE, 0))
        if wild_child is not None:
            self._retrieve_args_rec(args, arg_idx + 1, wild_child, results)

        if arg.is_variable:
            return

        # Try rigid match — O(1) lookup
        rigid_child = node._children_map.get((_NodeType.DRIGID, arg.symnum))
        if rigid_child is not None:
            if arg.arity == 0:
                self._retrieve_args_rec(args, arg_idx + 1, rigid_child, results)
            else:
                self._retrieve_nested_args(arg.args, 0, args, arg_idx + 1, rigid_child, results)

    def _retrieve_nested_args(
        self,
        inner_args: tuple[Term, ...],
        inner_idx: int,
        outer_args: tuple[Term, ...],
        outer_idx: int,
        node: DiscrimNode,
        results: list[Any],
    ) -> None:
        """Handle nested argument matching for complex arguments. O(1) child lookup."""
        if inner_idx >= len(inner_args):
            self._retrieve_args_rec(outer_args, outer_idx, node, results)
            return

        inner_arg = inner_args[inner_idx]

        # Try wildcard — O(1) lookup
        wild_child = node._children_map.get((_NodeType.DVARIABLE, 0))
        if wild_child is not None:
            self._retrieve_nested_args(
                inner_args, inner_idx + 1, outer_args, outer_idx, wild_child, results
            )

        if inner_arg.is_variable:
            return

        # Try rigid match — O(1) lookup
        rigid_child = node._children_map.get((_NodeType.DRIGID, inner_arg.symnum))
        if rigid_child is not None:
            if inner_arg.arity == 0:
                self._retrieve_nested_args(
                    inner_args, inner_idx + 1, outer_args, outer_idx, rigid_child, results
                )
            else:
                # Further nesting
                self._retrieve_nested_args(
                    inner_arg.args, 0,
                    inner_args, inner_idx + 1,
                    rigid_child, results,
                )

    def retrieve_generalizations_flat(self, query: Term) -> list[Any]:
        """Retrieve generalizations using flat stack (matches C algorithm exactly).

        This is the primary retrieval method — more robust than the recursive version.
        """
        results: list[Any] = []
        with self._lock.read_lock():
            # Flatten the query term to a pre-order list
            flat_query = list(query.subterms())
            self._flat_retrieve(flat_query, 0, self._root, results)
        return results

    def _flat_retrieve(
        self,
        flat_query: list[Term],
        pos: int,
        node: DiscrimNode,
        results: list[Any],
    ) -> None:
        """Flat-stack generalization retrieval (iterative).

        Walks the query in pre-order (flat_query) and the tree simultaneously.
        At each position, tries wildcard (skip entire subterm) and rigid match.
        Uses an explicit stack to avoid RecursionError on large terms.
        Uses O(1) dict lookup for child nodes instead of linear scan.
        """
        work: list[tuple[int, DiscrimNode]] = [(pos, node)]

        while work:
            pos, node = work.pop()

            if pos >= len(flat_query):
                if node.data:
                    results.extend(node.data)
                continue

            query_subterm = flat_query[pos]

            # Try wildcard child: DVARIABLE(0) matches entire subterm — O(1) lookup
            wild_child = node._children_map.get((_NodeType.DVARIABLE, 0))
            if wild_child is not None:
                skip = query_subterm.symbol_count
                work.append((pos + skip, wild_child))

            if query_subterm.is_variable:
                continue

            # Try rigid match: O(1) lookup by symbol
            rigid_child = node._children_map.get((_NodeType.DRIGID, query_subterm.symnum))
            if rigid_child is not None:
                work.append((pos + 1, rigid_child))

    def retrieve_unifiables_flat(self, query: Term) -> list[Any]:
        """Retrieve all stored objects whose terms could unify with the query.

        If the query is ground, uses generalization retrieval (efficient).
        If the query contains variables, returns all stored objects since any
        indexed term could potentially unify with a variable. The caller must
        still verify with unify() — this is an imperfect filter.
        """
        if query.is_ground:
            return self.retrieve_generalizations_flat(query)
        # Non-ground query: any stored term might unify. Return all.
        results: list[Any] = []
        with self._lock.read_lock():
            self._collect_all_data(self._root, results)
        return results

    def _collect_all_data(self, node: DiscrimNode, results: list[Any]) -> None:
        """Collect all data from all nodes in the tree."""
        work: list[DiscrimNode] = [node]
        while work:
            n = work.pop()
            if n.data:
                results.extend(n.data)
            for child in n.children:
                work.append(child)


# ══════════════════════════════════════════════════════════════════════════════
# Bind Discrimination Tree (perfect filter)
# ══════════════════════════════════════════════════════════════════════════════


class DiscrimBind:
    """Bind discrimination tree matching C discrimb.c.

    Variables in stored terms are distinguished by their variable number.
    During retrieval, tree variables are bound to query subterms, producing
    a substitution. Every result is guaranteed to be a generalization.
    Thread-safe via ReadWriteLock.
    """

    __slots__ = ("_root", "_lock", "_size")

    def __init__(self) -> None:
        self._root = DiscrimNode(symbol=-1, node_type=_NodeType.DRIGID)
        self._lock = make_rw_lock()
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def insert(self, t: Term, obj: Any) -> None:
        """Insert a term→object mapping. Variables distinguished by varnum.

        Matches C discrim_bind_insert(). Single-pass traversal.
        """
        with self._lock.write_lock():
            node = self._root
            work = [t]
            while work:
                term = work.pop()
                if term.is_variable:
                    node = _find_or_create_child(node, _NodeType.DVARIABLE, term.varnum)
                else:
                    node = _find_or_create_child(node, _NodeType.DRIGID, term.symnum)
                    for i in range(term.arity - 1, -1, -1):
                        work.append(term.args[i])
            node.data.append(obj)
            self._size += 1

    def delete(self, t: Term, obj: Any) -> bool:
        """Remove a term→object mapping. Returns True if found and removed.

        Matches C discrim_bind_delete(). Single-pass traversal.
        """
        with self._lock.write_lock():
            node = self._root
            ancestors: list[tuple[DiscrimNode, tuple[int, int]]] = []
            work = [t]
            while work:
                term = work.pop()
                if term.is_variable:
                    ntype, sym = _NodeType.DVARIABLE, term.varnum
                else:
                    ntype, sym = _NodeType.DRIGID, term.symnum
                    for i in range(term.arity - 1, -1, -1):
                        work.append(term.args[i])
                child = _find_child(node, ntype, sym)
                if child is None:
                    return False
                ancestors.append((node, (ntype, sym)))
                node = child

            try:
                node.data.remove(obj)
            except ValueError:
                return False

            self._size -= 1

            if not node.data and not node.children:
                for parent, key in reversed(ancestors):
                    child = parent._children_map.pop(key)
                    parent.children.remove(child)
                    if parent.data or parent.children:
                        break

            return True

    def retrieve_generalizations(
        self, query: Term, subst: Context | None = None
    ) -> list[tuple[Any, Context]]:
        """Retrieve all generalizations with their substitutions.

        Returns list of (object, context) pairs where each context maps
        tree variables to query subterms.

        Matches C discrim_bind_retrieve_first/next().
        """
        results: list[tuple[Any, Context]] = []
        with self._lock.read_lock():
            flat_query = list(query.subterms())
            if subst is None:
                subst = Context()
            self._bind_retrieve(flat_query, 0, self._root, subst, results)
        return results

    def _bind_retrieve(
        self,
        flat_query: list[Term],
        pos: int,
        node: DiscrimNode,
        subst: Context,
        results: list[tuple[Any, Context]],
    ) -> None:
        """Flat-stack bind retrieval with variable binding.

        Like the wild version but tracks variable bindings. When a tree
        variable is encountered:
        - If already bound: check that bound term matches query subterm
        - If unbound: bind it to the query subterm
        """
        if pos >= len(flat_query):
            if node.data:
                for obj in node.data:
                    # Create a copy of the substitution for each result
                    result_subst = Context()
                    for i in range(len(subst.terms)):
                        result_subst.terms[i] = subst.terms[i]
                        result_subst.contexts[i] = subst.contexts[i]
                    results.append((obj, result_subst))
            return

        query_subterm = flat_query[pos]
        skip = query_subterm.symbol_count

        # Try DVARIABLE children — iterate only variable children via the children list
        # (bind trees may have multiple variable nodes with different varnums)
        for child in node.children:
            if child.node_type != _NodeType.DVARIABLE:
                break  # sorted order: all DVARIABLE nodes come first
            varnum = child.symbol
            if subst.is_bound(varnum):
                bound_term = subst.terms[varnum]
                if bound_term is not None and bound_term == query_subterm:
                    self._bind_retrieve(flat_query, pos + skip, child, subst, results)
            else:
                subst.bind(varnum, query_subterm, None)
                self._bind_retrieve(flat_query, pos + skip, child, subst, results)
                subst.unbind(varnum)

        if query_subterm.is_variable:
            return

        # Try DRIGID child — O(1) lookup
        rigid_child = node._children_map.get((_NodeType.DRIGID, query_subterm.symnum))
        if rigid_child is not None:
            self._bind_retrieve(flat_query, pos + 1, rigid_child, subst, results)


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Index wrapper (matching C mindex.c)
# ══════════════════════════════════════════════════════════════════════════════


class IndexType(IntEnum):
    """Index type selection matching C Mindextype."""
    LINEAR = 0
    DISCRIM_WILD = 1
    DISCRIM_BIND = 2


class Mindex:
    """Multi-index interface matching C mindex.c.

    Provides a uniform interface over different index backends.
    Currently supports LINEAR (brute force) and DISCRIM_WILD/DISCRIM_BIND.
    """

    __slots__ = ("_index_type", "_discrim_wild", "_discrim_bind", "_linear", "_lock")

    def __init__(self, index_type: IndexType = IndexType.DISCRIM_WILD) -> None:
        self._index_type = index_type
        self._discrim_wild: DiscrimWild | None = None
        self._discrim_bind: DiscrimBind | None = None
        self._linear: list[tuple[Term, Any]] | None = None
        self._lock = make_rw_lock()

        if index_type == IndexType.DISCRIM_WILD:
            self._discrim_wild = DiscrimWild()
        elif index_type == IndexType.DISCRIM_BIND:
            self._discrim_bind = DiscrimBind()
        elif index_type == IndexType.LINEAR:
            self._linear = []

    @property
    def index_type(self) -> IndexType:
        return self._index_type

    @property
    def size(self) -> int:
        if self._discrim_wild is not None:
            return self._discrim_wild.size
        if self._discrim_bind is not None:
            return self._discrim_bind.size
        if self._linear is not None:
            return len(self._linear)
        return 0

    def insert(self, t: Term, obj: Any) -> None:
        """Insert a term→object mapping."""
        if self._discrim_wild is not None:
            self._discrim_wild.insert(t, obj)
        elif self._discrim_bind is not None:
            self._discrim_bind.insert(t, obj)
        elif self._linear is not None:
            with self._lock.write_lock():
                self._linear.append((t, obj))

    def delete(self, t: Term, obj: Any) -> bool:
        """Remove a term→object mapping."""
        if self._discrim_wild is not None:
            return self._discrim_wild.delete(t, obj)
        if self._discrim_bind is not None:
            return self._discrim_bind.delete(t, obj)
        if self._linear is not None:
            with self._lock.write_lock():
                try:
                    self._linear.remove((t, obj))
                    return True
                except ValueError:
                    return False
        return False

    def retrieve_generalizations(self, query: Term) -> list[Any]:
        """Retrieve objects whose indexed terms generalize the query."""
        if self._discrim_wild is not None:
            return self._discrim_wild.retrieve_generalizations_flat(query)
        if self._discrim_bind is not None:
            return [obj for obj, _ in self._discrim_bind.retrieve_generalizations(query)]
        if self._linear is not None:
            # Linear: return all (caller must match)
            with self._lock.read_lock():
                return [obj for _, obj in self._linear]
        return []

    def retrieve_unifiables(self, query: Term) -> list[Any]:
        """Retrieve objects whose indexed terms could unify with the query.

        For ground queries, equivalent to retrieve_generalizations (efficient).
        For non-ground queries, returns all stored objects as candidates.
        The caller must verify with unify().
        """
        if self._discrim_wild is not None:
            return self._discrim_wild.retrieve_unifiables_flat(query)
        if self._discrim_bind is not None:
            # DiscrimBind only supports generalization; fall back to all for non-ground
            if query.is_ground:
                return [obj for obj, _ in self._discrim_bind.retrieve_generalizations(query)]
            # Non-ground: return all entries
            return [obj for obj, _ in self._discrim_bind.retrieve_generalizations(query)]
        if self._linear is not None:
            with self._lock.read_lock():
                return [obj for _, obj in self._linear]
        return []

    def retrieve_generalizations_with_subst(
        self, query: Term
    ) -> list[tuple[Any, Context]]:
        """Retrieve generalizations with substitutions (bind variant only)."""
        if self._discrim_bind is not None:
            return self._discrim_bind.retrieve_generalizations(query)
        raise TypeError(
            f"retrieve_generalizations_with_subst requires DISCRIM_BIND, "
            f"got {self._index_type.name}"
        )
