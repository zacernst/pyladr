"""AC (Associative-Commutative) normal form computation matching C accanon.c.

AC normal form properties:
- Right-associated: f(a, f(b, f(c, d)))
- Arguments sorted in canonical order
- Flattened: no nested AC terms with the same symbol

The canonical ordering is: CONSTANT < COMPLEX < VARIABLE.
Within constants: by symbol number.
Within complex: lexicographic by arguments.
Within variables: by variable number.
"""

from __future__ import annotations

from pyladr.core.term import Term, get_rigid_term, get_variable_term

MAX_ACM_ARGS = 2500  # Matching C MAX_ACM_ARGS


def flatten_ac(t: Term, ac_symnum: int) -> list[Term]:
    """Flatten an AC term into a list of arguments.

    Recursively collects all arguments under nested applications of the
    same AC symbol. Matching C flatten().

    Args:
        t: Term to flatten.
        ac_symnum: The private_symbol of the AC operator.

    Returns:
        List of flattened arguments (leaves under the AC operator).
    """
    if t.private_symbol != ac_symnum or t.arity != 2:
        return [t]

    result: list[Term] = []
    result.extend(flatten_ac(t.args[0], ac_symnum))
    result.extend(flatten_ac(t.args[1], ac_symnum))
    return result


def right_associate(ac_symnum: int, arity: int, args: list[Term]) -> Term:
    """Build a right-associated term from a list of arguments.

    f(a, b, c, d) → f(a, f(b, f(c, d)))

    Matching C right_associate().

    Args:
        ac_symnum: The private_symbol of the AC operator.
        arity: Arity of the AC operator (always 2).
        args: List of arguments to right-associate.

    Returns:
        Right-associated term.
    """
    if len(args) == 0:
        raise ValueError("Cannot right-associate empty argument list")
    if len(args) == 1:
        return args[0]

    # Build from right: f(a_{n-1}, a_n), then f(a_{n-2}, ...), etc.
    result = args[-1]
    for i in range(len(args) - 2, -1, -1):
        result = Term(
            private_symbol=ac_symnum,
            arity=arity,
            args=(args[i], result),
        )
    return result


def term_compare_ncv(t1: Term, t2: Term) -> int:
    """Canonical ordering: NAME < COMPLEX < VARIABLE.

    Matching C compare_ncv() used for AC canonicalization.

    Returns:
        -1 if t1 < t2, 0 if equal, 1 if t1 > t2.
    """
    # Category: constant (0), complex (1), variable (2)
    def category(t: Term) -> int:
        if t.is_variable:
            return 2
        if t.is_constant:
            return 0
        return 1

    c1, c2 = category(t1), category(t2)
    if c1 != c2:
        return -1 if c1 < c2 else 1

    if t1.is_variable:
        if t1.varnum != t2.varnum:
            return -1 if t1.varnum < t2.varnum else 1
        return 0

    if t1.is_constant:
        if t1.symnum != t2.symnum:
            return -1 if t1.symnum < t2.symnum else 1
        return 0

    # Complex: compare by symbol, then lexicographic by arguments
    if t1.symnum != t2.symnum:
        return -1 if t1.symnum < t2.symnum else 1
    if t1.arity != t2.arity:
        return -1 if t1.arity < t2.arity else 1
    for a1, a2 in zip(t1.args, t2.args, strict=True):
        cmp = term_compare_ncv(a1, a2)
        if cmp != 0:
            return cmp
    return 0


def ac_canonical(t: Term, is_ac: callable) -> Term:
    """Compute AC canonical form of a term.

    Matching C ac_canonical(). Recursively canonicalizes all subterms,
    then if the top symbol is AC: flatten, sort, right-associate.

    Args:
        t: Term to canonicalize.
        is_ac: Function(symnum) -> bool that checks if a symbol is AC.

    Returns:
        Canonicalized term.
    """
    if t.is_variable:
        return t

    if t.is_constant:
        return t

    # Recursively canonicalize arguments first
    new_args = tuple(ac_canonical(a, is_ac) for a in t.args)

    if t.arity == 2 and is_ac(t.symnum):
        # Flatten, sort, right-associate
        temp = Term(private_symbol=t.private_symbol, arity=t.arity, args=new_args)
        flat_args = flatten_ac(temp, t.private_symbol)
        flat_args.sort(key=_sort_key)
        return right_associate(t.private_symbol, t.arity, flat_args)

    return Term(private_symbol=t.private_symbol, arity=t.arity, args=new_args)


def _sort_key(t: Term):
    """Sort key for AC canonical ordering: NAME < COMPLEX < VARIABLE."""
    if t.is_variable:
        return (2, t.varnum, ())
    if t.is_constant:
        return (0, t.symnum, ())
    return (1, t.symnum, tuple(_sort_key(a) for a in t.args))


def flatten_with_multiplicities(
    t: Term,
    ac_symnum: int,
) -> list[tuple[Term, int]]:
    """Flatten and collapse multiplicities.

    Matching C ac_mult_context() pattern. Returns (term, multiplicity) pairs
    for unique flattened arguments.
    """
    flat = flatten_ac(t, ac_symnum)
    flat.sort(key=_sort_key)

    result: list[tuple[Term, int]] = []
    i = 0
    while i < len(flat):
        count = 1
        while i + count < len(flat) and flat[i].term_ident(flat[i + count]):
            count += 1
        result.append((flat[i], count))
        i += count
    return result
