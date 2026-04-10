"""LADR syntax parser matching C parse.c behavior.

Implements a recursive-descent parser with precedence climbing for
operator-heavy mathematical formulas. Produces Term/Clause/Literal
objects identical to those the C implementation would produce.

Key C functions replicated:
- read_term / sread_term → parse_term
- terms_to_term → _parse_expr (precedence climbing)
- declare_standard_parse_types → _STANDARD_OPS
- tokenize → uses tokenizer module
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import (
    EQ_SYM,
    FALSE_SYM,
    NEQ_SYM,
    NOT_SYM,
    OR_SYM,
    TRUE_SYM,
    ParseType,
    SymbolTable,
)
from pyladr.core.term import Term, build_binary_term, build_unary_term, get_rigid_term, get_variable_term

from .tokenizer import Token, TokenType, strip_comments, tokenize

if TYPE_CHECKING:
    pass


# ── Parse errors ────────────────────────────────────────────────────────────


class ParseError(Exception):
    """Error during LADR parsing, with position information."""

    def __init__(self, message: str, pos: int = -1) -> None:
        self.pos = pos
        super().__init__(message)


# ── Operator table (matching C declare_standard_parse_types) ────────────────


@dataclass(frozen=True, slots=True)
class OpInfo:
    """Operator notation and precedence."""

    parse_type: ParseType
    precedence: int


# Standard operators from C declare_standard_parse_types()
# Precedence values and associativity match exactly.
_STANDARD_OPS: dict[str, OpInfo] = {
    ",": OpInfo(ParseType.INFIX_RIGHT, 999),
    "#": OpInfo(ParseType.INFIX_RIGHT, 810),
    "<->": OpInfo(ParseType.INFIX, 800),
    "->": OpInfo(ParseType.INFIX, 800),
    "<-": OpInfo(ParseType.INFIX, 800),
    "|": OpInfo(ParseType.INFIX_RIGHT, 790),
    "||": OpInfo(ParseType.INFIX_RIGHT, 790),
    "&": OpInfo(ParseType.INFIX_RIGHT, 780),
    "&&": OpInfo(ParseType.INFIX_RIGHT, 780),
    "=": OpInfo(ParseType.INFIX, 700),
    "!=": OpInfo(ParseType.INFIX, 700),
    "==": OpInfo(ParseType.INFIX, 700),
    "!==": OpInfo(ParseType.INFIX, 700),
    "<": OpInfo(ParseType.INFIX, 700),
    "<=": OpInfo(ParseType.INFIX, 700),
    ">": OpInfo(ParseType.INFIX, 700),
    ">=": OpInfo(ParseType.INFIX, 700),
    "@<": OpInfo(ParseType.INFIX, 700),
    "@<=": OpInfo(ParseType.INFIX, 700),
    "@>": OpInfo(ParseType.INFIX, 700),
    "@>=": OpInfo(ParseType.INFIX, 700),
    "+": OpInfo(ParseType.INFIX, 500),
    "*": OpInfo(ParseType.INFIX, 500),
    "@": OpInfo(ParseType.INFIX, 500),
    "/": OpInfo(ParseType.INFIX, 500),
    "\\": OpInfo(ParseType.INFIX, 500),
    "^": OpInfo(ParseType.INFIX, 500),
    "v": OpInfo(ParseType.INFIX, 500),
    NOT_SYM: OpInfo(ParseType.PREFIX, 350),
    "'": OpInfo(ParseType.POSTFIX, 300),
}

QUANTIFIER_PRECEDENCE = 750

MAX_PRECEDENCE = 1000


# ── Variable detection (matching C variable_name) ───────────────────────────


def is_variable_name(name: str) -> bool:
    """C variable_name() with STANDARD style: u-z are variables.

    The C default (not PROLOG_STYLE, not INTEGER_STYLE) considers
    names starting with u, v, w, x, y, z as variables.
    """
    return len(name) > 0 and "u" <= name[0] <= "z"


# ── Parser ──────────────────────────────────────────────────────────────────


class LADRParser:
    """LADR syntax parser matching C parse.c behavior.

    Usage:
        table = SymbolTable()
        parser = LADRParser(table)
        term = parser.parse_term("f(x, a) = a.")
        clauses = parser.parse_input(input_text)
    """

    def __init__(self, symbol_table: SymbolTable | None = None) -> None:
        self.symbols = symbol_table if symbol_table is not None else SymbolTable()
        self.ops: dict[str, OpInfo] = dict(_STANDARD_OPS)

    # ── Public API ──────────────────────────────────────────────────────

    def parse_term_from_string(self, s: str) -> Term:
        """Parse a single term from a string (C parse_term_from_string).

        The string should be a single term WITHOUT a trailing period.
        """
        tokens = tokenize(s)
        pos = 0
        term, pos = self._parse_expr(tokens, pos, MAX_PRECEDENCE)
        if pos < len(tokens):
            raise ParseError(
                f"Unexpected token after term: {tokens[pos]}",
                tokens[pos].pos,
            )
        return self._set_variables(term)

    def parse_term(self, s: str) -> Term:
        """Parse a single term from string, possibly with trailing period."""
        s = strip_comments(s).strip()
        if s.endswith("."):
            s = s[:-1].strip()
        return self.parse_term_from_string(s)

    def parse_clause_from_string(self, s: str) -> Clause:
        """Parse a clause from a string (C parse_clause_from_string).

        A clause is a term at the top level. If the term is a disjunction
        (using '|'), it becomes multiple literals.
        """
        term = self.parse_term(s)
        return self._term_to_clause(term)

    def parse_input(self, text: str) -> ParsedInput:
        """Parse a complete LADR input file.

        Recognizes:
        - formulas(NAME). ... end_of_list.
        - op(PREC, TYPE, SYM).  (operator declarations)
        - % comments
        """
        text = strip_comments(text)
        result = ParsedInput()
        statements = self._split_statements(text)

        i = 0
        while i < len(statements):
            stmt = statements[i].strip()
            if not stmt:
                i += 1
                continue

            # Check for formulas(NAME) block
            if stmt.startswith("formulas(") or stmt.startswith("clauses("):
                list_name = self._extract_list_name(stmt)
                i += 1
                clauses: list[Clause] = []
                while i < len(statements):
                    inner = statements[i].strip()
                    if inner == "end_of_list":
                        break
                    if inner:
                        clause = self.parse_clause_from_string(inner + ".")
                        clauses.append(clause)
                    i += 1
                if list_name == "goals":
                    result.goals.extend(clauses)
                elif list_name in ("sos", "assumptions"):
                    result.sos.extend(clauses)
                elif list_name == "usable":
                    result.usable.extend(clauses)
                elif list_name == "hints":
                    result.hints.extend(clauses)
                elif list_name == "demodulators":
                    result.demodulators.extend(clauses)
                else:
                    result.sos.extend(clauses)
                i += 1
                continue

            # Check for op() declaration
            if stmt.startswith("op("):
                self._parse_op_declaration(stmt)
                i += 1
                continue

            # Parse set/clear/assign directives
            if stmt.startswith("set(") and stmt.endswith(")"):
                flag_name = stmt[4:-1].strip()
                if flag_name:
                    result.flags[flag_name] = True
                i += 1
                continue
            if stmt.startswith("clear(") and stmt.endswith(")"):
                flag_name = stmt[6:-1].strip()
                if flag_name:
                    result.flags[flag_name] = False
                i += 1
                continue
            if stmt.startswith("assign(") and stmt.endswith(")"):
                inner = stmt[7:-1].strip()
                parts = inner.split(",", 1)
                if len(parts) == 2:
                    name = parts[0].strip()
                    try:
                        val: int | float = int(parts[1].strip())
                    except ValueError:
                        try:
                            val = float(parts[1].strip())
                        except ValueError:
                            val = 0
                    result.assigns[name] = val
                i += 1
                continue

            i += 1

        return result

    # ── Precedence climbing parser ──────────────────────────────────────

    def _parse_expr(
        self, tokens: list[Token], pos: int, max_prec: int
    ) -> tuple[Term, int]:
        """Parse an expression with precedence climbing (C terms_to_term).

        Args:
            tokens: Token list.
            pos: Current position.
            max_prec: Maximum allowed precedence for this subexpression.

        Returns:
            (Term, new_pos)
        """
        if pos >= len(tokens):
            raise ParseError("Unexpected end of input")

        # ── Prefix operators ────────────────────────────────────────────
        tok = tokens[pos]
        op = self._get_op(tok)

        if op is not None and op.parse_type in (ParseType.PREFIX, ParseType.PREFIX_PAREN):
            if op.precedence > max_prec:
                raise ParseError(
                    f"Prefix operator '{tok.value}' precedence {op.precedence} "
                    f"exceeds max {max_prec}",
                    tok.pos,
                )
            # For PREFIX_PAREN: arg prec = prec-1; for PREFIX: arg prec = prec
            arg_prec = op.precedence - 1 if op.parse_type == ParseType.PREFIX_PAREN else op.precedence
            sym = self.symbols.str_to_sn(tok.value, 1)
            pos += 1
            arg, pos = self._parse_expr(tokens, pos, arg_prec)
            left = build_unary_term(sym, arg)
        # ── Quantifiers (all/exists) ────────────────────────────────────
        elif tok.value in ("all", "exists") and tok.type == TokenType.ORDINARY:
            if QUANTIFIER_PRECEDENCE > max_prec:
                raise ParseError(
                    f"Quantifier '{tok.value}' precedence exceeds max",
                    tok.pos,
                )
            quant_name = tok.value
            pos += 1
            if pos >= len(tokens) or tokens[pos].type != TokenType.ORDINARY:
                raise ParseError("Expected variable after quantifier", tok.pos)
            var_name = tokens[pos].value
            var_sym = self.symbols.str_to_sn(var_name, 0)
            var_term = get_rigid_term(var_sym, 0)
            pos += 1
            # Body of quantifier
            body, pos = self._parse_expr(tokens, pos, QUANTIFIER_PRECEDENCE)
            quant_sym = self.symbols.str_to_sn("$quantified", 3)
            quant_name_sym = self.symbols.str_to_sn(quant_name, 0)
            quant_name_term = get_rigid_term(quant_name_sym, 0)
            left = get_rigid_term(quant_sym, 3, (quant_name_term, var_term, body))
        # ── Parenthesized expression ────────────────────────────────────
        elif tok.type == TokenType.PUNC and tok.value == "(":
            pos += 1
            left, pos = self._parse_expr(tokens, pos, MAX_PRECEDENCE)
            if pos >= len(tokens) or tokens[pos].value != ")":
                raise ParseError("Expected ')'", tok.pos)
            pos += 1
        # ── List notation [a, b, c] ────────────────────────────────────
        elif tok.type == TokenType.PUNC and tok.value == "[":
            left, pos = self._parse_list(tokens, pos)
        # ── Atom/constant/variable ──────────────────────────────────────
        elif tok.type in (TokenType.ORDINARY, TokenType.STRING):
            name = tok.value
            pos += 1
            # Check for function application: f(...)
            if pos < len(tokens) and tokens[pos].type == TokenType.PUNC and tokens[pos].value == "(":
                pos += 1  # skip (
                args: list[Term] = []
                if pos < len(tokens) and not (
                    tokens[pos].type == TokenType.PUNC and tokens[pos].value == ")"
                ):
                    arg, pos = self._parse_expr(tokens, pos, 998)  # below comma prec
                    args.append(arg)
                    while pos < len(tokens) and tokens[pos].value == ",":
                        pos += 1
                        arg, pos = self._parse_expr(tokens, pos, 998)  # below comma prec
                        args.append(arg)
                if pos >= len(tokens) or tokens[pos].value != ")":
                    raise ParseError(f"Expected ')' after arguments of '{name}'", tok.pos)
                pos += 1
                sym = self.symbols.str_to_sn(name, len(args))
                left = get_rigid_term(sym, len(args), tuple(args))
            else:
                sym = self.symbols.str_to_sn(name, 0)
                left = get_rigid_term(sym, 0)
        elif tok.type == TokenType.SPECIAL:
            # Special token that isn't a recognized prefix op — treat as atom
            name = tok.value
            pos += 1
            sym = self.symbols.str_to_sn(name, 0)
            left = get_rigid_term(sym, 0)
        else:
            raise ParseError(f"Unexpected token: {tok}", tok.pos)

        # ── Postfix and infix operators (left-to-right) ─────────────────
        while pos < len(tokens):
            tok = tokens[pos]
            op = self._get_op(tok)
            if op is None:
                break

            # Postfix
            if op.parse_type in (ParseType.POSTFIX, ParseType.POSTFIX_PAREN):
                if op.precedence > max_prec:
                    break
                sym = self.symbols.str_to_sn(tok.value, 1)
                left = build_unary_term(sym, left)
                pos += 1
                continue

            # Infix
            if op.parse_type in (
                ParseType.INFIX,
                ParseType.INFIX_LEFT,
                ParseType.INFIX_RIGHT,
            ):
                if op.precedence > max_prec:
                    break
                # Determine right operand precedence
                if op.parse_type == ParseType.INFIX_RIGHT:
                    right_prec = op.precedence
                elif op.parse_type == ParseType.INFIX_LEFT:
                    right_prec = op.precedence - 1
                else:  # INFIX (non-associative)
                    right_prec = op.precedence - 1

                sym = self.symbols.str_to_sn(tok.value, 2)
                pos += 1
                right, pos = self._parse_expr(tokens, pos, right_prec)
                left = build_binary_term(sym, left, right)
                continue

            break

        return left, pos

    def _parse_list(
        self, tokens: list[Token], pos: int
    ) -> tuple[Term, int]:
        """Parse list notation [a, b, c] → $cons(a, $cons(b, $cons(c, $nil)))."""
        pos += 1  # skip [
        nil_sym = self.symbols.str_to_sn("$nil", 0)
        cons_sym = self.symbols.str_to_sn("$cons", 2)

        if pos < len(tokens) and tokens[pos].value == "]":
            pos += 1
            return get_rigid_term(nil_sym, 0), pos

        elements: list[Term] = []
        tail: Term | None = None

        elem, pos = self._parse_expr(tokens, pos, 998)
        elements.append(elem)
        while pos < len(tokens) and tokens[pos].value == ",":
            pos += 1
            elem, pos = self._parse_expr(tokens, pos, 998)
            elements.append(elem)

        # Check for cons notation [x:y]
        if pos < len(tokens) and tokens[pos].value == ":":
            pos += 1
            tail, pos = self._parse_expr(tokens, pos, 998)

        if pos >= len(tokens) or tokens[pos].value != "]":
            raise ParseError("Expected ']'")
        pos += 1

        # Build cons list from right
        result = tail if tail is not None else get_rigid_term(nil_sym, 0)
        for elem in reversed(elements):
            result = build_binary_term(cons_sym, elem, result)

        return result, pos

    # ── Helper methods ──────────────────────────────────────────────────

    def _get_op(self, tok: Token) -> OpInfo | None:
        """Look up operator info for a token."""
        if tok.type == TokenType.PUNC:
            # Punctuation chars like '.' and ',' may be operators
            return self.ops.get(tok.value)
        if tok.type in (TokenType.ORDINARY, TokenType.SPECIAL):
            return self.ops.get(tok.value)
        return None

    def _set_variables(self, term: Term) -> Term:
        """Convert constants to variables based on name (C clause_set_variables).

        In LADR standard style, names starting with u-z are variables.
        This replaces rigid terms with matching names by variable terms.
        """
        return self._set_vars_recurse(term, {})

    def _set_vars_recurse(
        self, term: Term, var_map: dict[int, int]
    ) -> Term:
        """Recursively convert variable-named constants to variable terms."""
        if term.is_variable:
            return term

        if term.is_constant:
            name = self.symbols.sn_to_str(term.symnum)
            if is_variable_name(name):
                if term.symnum not in var_map:
                    var_map[term.symnum] = len(var_map)
                return get_variable_term(var_map[term.symnum])
            return term

        # Complex term: recurse into args
        new_args = tuple(self._set_vars_recurse(a, var_map) for a in term.args)
        if new_args == term.args:
            return term
        return Term(private_symbol=term.private_symbol, arity=term.arity, args=new_args)

    def _term_to_clause(self, term: Term) -> Clause:
        """Convert a parsed term to a Clause (C term_to_clause).

        Handles disjunction (|) at top level → multiple literals.
        Handles negation (-) → negative literals.
        """
        literals = self._term_to_literals(term)
        return Clause(literals=tuple(literals))

    def _term_to_literals(self, term: Term) -> list[Literal]:
        """Flatten a disjunctive term into a list of literals."""
        # Check if top-level is disjunction
        if term.is_complex and term.arity == 2:
            name = self.symbols.sn_to_str(term.symnum)
            if name == OR_SYM or name == "||":
                left = self._term_to_literals(term.arg(0))
                right = self._term_to_literals(term.arg(1))
                return left + right

        # Check for negation
        if term.is_complex and term.arity == 1:
            name = self.symbols.sn_to_str(term.symnum)
            if name == NOT_SYM:
                return [Literal(sign=False, atom=term.arg(0))]

        # Check for $F (false)
        if term.is_constant:
            name = self.symbols.sn_to_str(term.symnum)
            if name == FALSE_SYM:
                return []  # empty clause

        return [Literal(sign=True, atom=term)]

    def _split_statements(self, text: str) -> list[str]:
        """Split input text into period-terminated statements.

        Respects parenthesized groups and quoted strings.
        """
        statements: list[str] = []
        current: list[str] = []
        depth = 0
        in_quote = False
        i = 0
        n = len(text)

        while i < n:
            c = text[i]
            if in_quote:
                current.append(c)
                if c == QUOTE_CHAR:
                    in_quote = False
            elif c == QUOTE_CHAR:
                current.append(c)
                in_quote = True
            elif c == "(":
                current.append(c)
                depth += 1
            elif c == ")":
                current.append(c)
                depth = max(0, depth - 1)
            elif c == "." and depth == 0:
                stmt = "".join(current).strip()
                if stmt:
                    statements.append(stmt)
                current = []
            else:
                current.append(c)
            i += 1

        # Handle trailing content without period
        remaining = "".join(current).strip()
        if remaining:
            statements.append(remaining)

        return statements

    def _extract_list_name(self, stmt: str) -> str:
        """Extract name from 'formulas(NAME)' or 'clauses(NAME)'."""
        # Find content between first ( and last )
        start = stmt.index("(") + 1
        end = stmt.rindex(")")
        return stmt[start:end].strip()

    def _parse_op_declaration(self, stmt: str) -> None:
        """Parse op(PREC, TYPE, SYM) declaration."""
        # Strip "op(" prefix and ")" suffix
        inner = stmt[3:-1] if stmt.endswith(")") else stmt[3:]
        parts = [p.strip() for p in inner.split(",", 2)]
        if len(parts) != 3:
            return  # silently ignore malformed op declarations

        try:
            prec = int(parts[0])
        except ValueError:
            return

        type_map = {
            "infix": ParseType.INFIX,
            "infix_left": ParseType.INFIX_LEFT,
            "infix_right": ParseType.INFIX_RIGHT,
            "prefix": ParseType.PREFIX,
            "prefix_paren": ParseType.PREFIX_PAREN,
            "postfix": ParseType.POSTFIX,
            "postfix_paren": ParseType.POSTFIX_PAREN,
            "ordinary": ParseType.ORDINARY,
        }

        ptype = type_map.get(parts[1])
        if ptype is None:
            return

        sym_name = parts[2].strip()
        if ptype == ParseType.ORDINARY:
            self.ops.pop(sym_name, None)
        else:
            self.ops[sym_name] = OpInfo(parse_type=ptype, precedence=prec)


# ── Parsed input structure ──────────────────────────────────────────────────


@dataclass
class ParsedInput:
    """Result of parsing a complete LADR input file."""

    sos: list[Clause] = None  # type: ignore[assignment]
    goals: list[Clause] = None  # type: ignore[assignment]
    usable: list[Clause] = None  # type: ignore[assignment]
    hints: list[Clause] = None  # type: ignore[assignment]
    demodulators: list[Clause] = None  # type: ignore[assignment]
    flags: dict[str, bool] = None  # type: ignore[assignment]
    assigns: dict[str, int | float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.sos is None:
            self.sos = []
        if self.goals is None:
            self.goals = []
        if self.usable is None:
            self.usable = []
        if self.hints is None:
            self.hints = []
        if self.demodulators is None:
            self.demodulators = []
        if self.flags is None:
            self.flags = {}
        if self.assigns is None:
            self.assigns = {}

    @property
    def all_clauses(self) -> list[Clause]:
        return self.sos + self.goals + self.usable + self.hints + self.demodulators


# ── Convenience functions ───────────────────────────────────────────────────

QUOTE_CHAR = '"'


def parse_term(s: str, symbol_table: SymbolTable | None = None) -> Term:
    """Parse a single LADR term from string."""
    parser = LADRParser(symbol_table)
    return parser.parse_term(s)


def parse_clause(s: str, symbol_table: SymbolTable | None = None) -> Clause:
    """Parse a single LADR clause from string."""
    parser = LADRParser(symbol_table)
    return parser.parse_clause_from_string(s)


def parse_input(text: str, symbol_table: SymbolTable | None = None) -> ParsedInput:
    """Parse a complete LADR input file."""
    parser = LADRParser(symbol_table)
    return parser.parse_input(text)
