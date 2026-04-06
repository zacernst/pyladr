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
from pyladr.threading_guide import ReadWriteLock


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
    """

    symbol: int          # varnum for DVARIABLE, symnum for DRIGID
    node_type: _NodeType
    children: list[DiscrimNode] = field(default_factory=list)
    data: list[Any] = field(default_factory=list)

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
    """Find a child node by type and symbol (linear scan, sorted by type then symbol)."""
    for child in parent.children:
        if child.node_type == node_type and child.symbol == symbol:
            return child
    return None


def _find_or_create_child(
    parent: DiscrimNode, node_type: _NodeType, symbol: int
) -> DiscrimNode:
    """Find or create a child node, maintaining sorted order.

    Sort order: DVARIABLE nodes first (sorted by symbol), then DRIGID (sorted by symbol).
    This matches C where variables always precede rigid symbols in sibling lists.
    """
    # Search for existing
    for i, child in enumerate(parent.children):
        if child.node_type == node_type and child.symbol == symbol:
            return child
        # Insertion point: maintain sort (DVARIABLE < DRIGID, then by symbol)
        if (child.node_type, child.symbol) > (node_type, symbol):
            new_node = DiscrimNode(symbol=symbol, node_type=node_type)
            parent.children.insert(i, new_node)
            return new_node
    # Append at end
    new_node = DiscrimNode(symbol=symbol, node_type=node_type)
    parent.children.append(new_node)
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
        self._lock = ReadWriteLock()
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def insert(self, t: Term, obj: Any) -> None:
        """Insert a term→object mapping into the tree.

        Matches C discrim_wild_insert().
        """
        path = _term_to_wild_path(t)
        with self._lock.write_lock():
            node = self._root
            for ntype, sym in path:
                node = _find_or_create_child(node, ntype, sym)
            node.data.append(obj)
            self._size += 1

    def delete(self, t: Term, obj: Any) -> bool:
        """Remove a term→object mapping. Returns True if found and removed.

        Matches C discrim_wild_delete().
        """
        path = _term_to_wild_path(t)
        with self._lock.write_lock():
            # Navigate to leaf
            node = self._root
            ancestors: list[tuple[DiscrimNode, int]] = []  # (parent, child_index)
            for ntype, sym in path:
                child = _find_child(node, ntype, sym)
                if child is None:
                    return False
                idx = node.children.index(child)
                ancestors.append((node, idx))
                node = child

            # Remove object from leaf data
            try:
                node.data.remove(obj)
            except ValueError:
                return False

            self._size -= 1

            # Prune empty nodes bottom-up
            if not node.data and not node.children:
                for parent, idx in reversed(ancestors):
                    parent.children.pop(idx)
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
        """
        # Try wildcard children (DVARIABLE nodes match any query term)
        for child in node.children:
            if child.node_type == _NodeType.DVARIABLE:
                # Wildcard matches entire query subterm — skip to next position
                if child.data:
                    results.extend(child.data)
                # If child has further children, this is an internal wildcard
                # in a longer path — shouldn't happen in a well-formed tree
                # since wildcard always matches exactly one subterm position
                break  # only one wildcard node (symbol=0)

        if query.is_variable:
            # Query variable: only wildcards match (already handled above)
            return

        # Try rigid match: stored symbol must equal query symbol
        for child in node.children:
            if child.node_type == _NodeType.DRIGID and child.symbol == query.symnum:
                if query.arity == 0:
                    # Constant: reached leaf position
                    if child.data:
                        results.extend(child.data)
                else:
                    # Complex: recursively match arguments
                    self._retrieve_args_rec(query.args, 0, child, results)
                break  # symbols are unique at each level

    def _retrieve_args_rec(
        self,
        args: tuple[Term, ...],
        arg_idx: int,
        node: DiscrimNode,
        results: list[Any],
    ) -> None:
        """Recursively match arguments of a complex term."""
        if arg_idx >= len(args):
            # All arguments matched — collect data at this node
            if node.data:
                results.extend(node.data)
            return

        arg = args[arg_idx]

        # Try wildcard children
        for child in node.children:
            if child.node_type == _NodeType.DVARIABLE:
                self._retrieve_args_rec(args, arg_idx + 1, child, results)
                break

        if arg.is_variable:
            # Query arg is variable: only wildcards match (handled above)
            return

        # Try rigid match on this argument
        for child in node.children:
            if child.node_type == _NodeType.DRIGID and child.symbol == arg.symnum:
                if arg.arity == 0:
                    # Constant argument
                    self._retrieve_args_rec(args, arg_idx + 1, child, results)
                else:
                    # Complex argument: recurse into its args first,
                    # then continue with remaining top-level args
                    self._retrieve_nested_args(arg.args, 0, args, arg_idx + 1, child, results)
                break

    def _retrieve_nested_args(
        self,
        inner_args: tuple[Term, ...],
        inner_idx: int,
        outer_args: tuple[Term, ...],
        outer_idx: int,
        node: DiscrimNode,
        results: list[Any],
    ) -> None:
        """Handle nested argument matching for complex arguments."""
        if inner_idx >= len(inner_args):
            # Done with inner args, continue with outer args
            self._retrieve_args_rec(outer_args, outer_idx, node, results)
            return

        inner_arg = inner_args[inner_idx]

        # Try wildcard
        for child in node.children:
            if child.node_type == _NodeType.DVARIABLE:
                self._retrieve_nested_args(
                    inner_args, inner_idx + 1, outer_args, outer_idx, child, results
                )
                break

        if inner_arg.is_variable:
            return

        # Try rigid match
        for child in node.children:
            if child.node_type == _NodeType.DRIGID and child.symbol == inner_arg.symnum:
                if inner_arg.arity == 0:
                    self._retrieve_nested_args(
                        inner_args, inner_idx + 1, outer_args, outer_idx, child, results
                    )
                else:
                    # Further nesting
                    self._retrieve_nested_args(
                        inner_arg.args, 0,
                        # After inner_arg's args, continue with rest of inner_args
                        # This is handled by the pre-order traversal
                        inner_args, inner_idx + 1,
                        child, results,
                    )
                    # FIXME: This doesn't correctly handle deeply nested terms.
                    # The C version uses an explicit flat stack. Let's use that approach.
                break

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
        """Flat-stack generalization retrieval.

        Walks the query in pre-order (flat_query) and the tree simultaneously.
        At each position, tries wildcard (skip entire subterm) and rigid match.
        """
        if pos >= len(flat_query):
            # Consumed all query subterms — collect results
            if node.data:
                results.extend(node.data)
            return

        query_subterm = flat_query[pos]

        # Try wildcard children: DVARIABLE matches entire subterm at this position
        for child in node.children:
            if child.node_type == _NodeType.DVARIABLE:
                # Skip the entire subterm (all its descendant subterms)
                skip = query_subterm.symbol_count
                self._flat_retrieve(flat_query, pos + skip, child, results)
                break  # only one DVARIABLE node (symbol=0)

        if query_subterm.is_variable:
            # Query is variable: only wildcards match (already handled)
            return

        # Try rigid match: stored symbol must equal query symbol
        for child in node.children:
            if child.node_type == _NodeType.DRIGID and child.symbol == query_subterm.symnum:
                # Matched: advance to next subterm in pre-order
                self._flat_retrieve(flat_query, pos + 1, child, results)
                break


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
        self._lock = ReadWriteLock()
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def insert(self, t: Term, obj: Any) -> None:
        """Insert a term→object mapping. Variables distinguished by varnum.

        Matches C discrim_bind_insert().
        """
        path = _term_to_path(t)
        with self._lock.write_lock():
            node = self._root
            for ntype, sym in path:
                node = _find_or_create_child(node, ntype, sym)
            node.data.append(obj)
            self._size += 1

    def delete(self, t: Term, obj: Any) -> bool:
        """Remove a term→object mapping. Returns True if found and removed.

        Matches C discrim_bind_delete().
        """
        path = _term_to_path(t)
        with self._lock.write_lock():
            node = self._root
            ancestors: list[tuple[DiscrimNode, int]] = []
            for ntype, sym in path:
                child = _find_child(node, ntype, sym)
                if child is None:
                    return False
                idx = node.children.index(child)
                ancestors.append((node, idx))
                node = child

            try:
                node.data.remove(obj)
            except ValueError:
                return False

            self._size -= 1

            if not node.data and not node.children:
                for parent, idx in reversed(ancestors):
                    parent.children.pop(idx)
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

        # Try DVARIABLE children (tree variables)
        for child in node.children:
            if child.node_type == _NodeType.DVARIABLE:
                varnum = child.symbol
                if subst.is_bound(varnum):
                    # Already bound: check match
                    bound_term = subst.terms[varnum]
                    if bound_term is not None and bound_term.term_ident(query_subterm):
                        self._bind_retrieve(flat_query, pos + skip, child, subst, results)
                else:
                    # Unbound: bind and continue
                    subst.bind(varnum, query_subterm, None)
                    self._bind_retrieve(flat_query, pos + skip, child, subst, results)
                    subst.unbind(varnum)

        if query_subterm.is_variable:
            # Query variable: only tree variables can match (handled above)
            return

        # Try DRIGID children
        for child in node.children:
            if child.node_type == _NodeType.DRIGID and child.symbol == query_subterm.symnum:
                self._bind_retrieve(flat_query, pos + 1, child, subst, results)
                break


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
        self._lock = ReadWriteLock()

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
