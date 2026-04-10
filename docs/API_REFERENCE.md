# PyLADR API Reference

Complete reference for all public modules, classes, and functions.

## Package Overview

```
pyladr/
├── core/        # Fundamental data structures
├── inference/   # Inference rules
├── search/      # Search algorithms
├── parsing/     # LADR format parser
├── ordering/    # Term orderings
├── indexing/    # Term indexing
├── mace4/       # Finite model finder
├── parallel/    # Parallel inference engine
├── ml/          # Machine learning components
└── apps/        # Command-line applications
```

---

## `pyladr.core` — Core Data Structures

All core types are re-exported from `pyladr.core` for convenience.

### `Term` (frozen dataclass)

**Module:** `pyladr.core.term`

Immutable first-order term matching C LADR `struct term`. Terms are hashable and suitable as dictionary keys.

```python
@dataclass(frozen=True, slots=True)
class Term:
    private_symbol: int    # >= 0 for variables, < 0 for rigid symbols
    arity: int = 0
    args: tuple[Term, ...] = ()
    container: object = None  # Back-reference (not in hash/eq)
    term_id: int = 0          # Unique ID for indexing (not in hash/eq)
```

**Term types** (determined by `private_symbol` and `arity`):

| Type | Condition | Example |
|------|-----------|---------|
| `VARIABLE` | `private_symbol >= 0` | Variable `x` |
| `CONSTANT` | `private_symbol < 0, arity == 0` | Constant `e` |
| `COMPLEX` | `private_symbol < 0, arity > 0` | `f(x,y)` |

**Key properties:**

- `term_type` → `TermType` — Classification as VARIABLE/CONSTANT/COMPLEX
- `is_variable` / `is_constant` / `is_complex` → `bool`
- `is_ground` → `bool` — True if no variables occur in the term (cached)
- `symbol_count` → `int` — Number of symbols in the term (cached)

**Factory functions:**

```python
get_variable_term(varnum: int) -> Term
get_rigid_term(symnum: int, arity: int, args: tuple[Term, ...] = ()) -> Term
build_binary_term(symnum: int, left: Term, right: Term) -> Term
build_unary_term(symnum: int, arg: Term) -> Term
copy_term(t: Term) -> Term
```

**Constants:**

- `MAX_VARS = 100` — Max distinct variables per term
- `MAX_VNUM = 5000` — Max variable ID
- `MAX_ARITY = 255` — Max function arity

### `Clause` (dataclass)

**Module:** `pyladr.core.clause`

A disjunction of `Literal`s with metadata, matching C `struct topform`.

```python
@dataclass(slots=True)
class Clause:
    literals: tuple[Literal, ...]
    justification: tuple[Justification, ...] = ()
    id: int = 0
    weight: float = 0.0
    initial: bool = False
```

**Key properties:**

- `is_empty` → `bool` — True if no literals (empty clause = contradiction)
- `is_unit` → `bool` — True if exactly one literal
- `is_positive` / `is_negative` → `bool` — All literals same sign
- `literal_count` → `int`

### `Literal` (frozen dataclass)

```python
@dataclass(frozen=True, slots=True)
class Literal:
    sign: bool   # True = positive, False = negative
    atom: Term   # The atomic formula
```

**Properties:** `is_positive`, `is_negative`, `is_equality`, `is_neg_equality`

### `Justification` (frozen dataclass)

Proof tracking — each clause has a sequence of justification steps.

```python
@dataclass(frozen=True, slots=True)
class Justification:
    just_type: JustType       # INPUT, BINARY_RES, PARA, DEMOD, etc.
    clause_id: int = 0        # For COPY, DENY, etc.
    clause_ids: tuple[int, ...] = ()  # For BINARY_RES, HYPER_RES
    para: ParaJust | None = None      # For PARA
    demod_steps: tuple[...] = ()      # For DEMOD
```

**`JustType` enum:** `INPUT`, `GOAL`, `DENY`, `CLAUSIFY`, `COPY`, `BINARY_RES`, `HYPER_RES`, `UR_RES`, `FACTOR`, `PARA`, `DEMOD`, `UNIT_DEL`, `FLIP`, `BACK_DEMOD`, `BACK_UNIT_DEL`, `NEW_SYMBOL`, `EXPAND_DEF`, `FOLD_DEF`, `RENUMBER`, `PROPOSITIONAL`, `INSTANTIATE`, `IVY`

### `SymbolTable`

**Module:** `pyladr.core.symbol`

Manages symbols (function and predicate names) and their metadata.

```python
class SymbolTable:
    def lookup(name: str) -> int | None
    def insert(name: str, arity: int, sym_type: SymbolType) -> int
    def name(symnum: int) -> str
    def arity(symnum: int) -> int
    def sym_type(symnum: int) -> SymbolType
```

**Related types:**

- `SymbolType` — `FUNCTION` or `PREDICATE`
- `ParseType` — `PREFIX`, `INFIX`, `INFIX_LEFT`, `INFIX_RIGHT`, `POSTFIX`, etc.
- `UnifTheory` — Unification theory (`NONE`, `AC`)
- `VariableStyle` — Variable naming convention

### Unification & Substitution

**Module:** `pyladr.core.substitution`

```python
# Unification — returns True if terms unify, binding variables in contexts
unify(t1: Term, c1: Context, t2: Term, c2: Context, trail: Trail) -> bool

# Matching — returns True if pattern matches target (one-way unification)
match(pattern: Term, c_pat: Context, target: Term, trail: Trail) -> bool

# Variant check — True if two terms are alphabetic variants
variant(t1: Term, c1: Context, t2: Term, c2: Context) -> bool

# Apply a substitution context to a term
apply_substitution(t: Term, c: Context) -> Term

# Occur check — True if variable occurs in term
occur_check(varnum: int, ctx: Context, t: Term, ctx2: Context) -> bool
```

**`Context`** — Binds variables to terms during unification.
**`Trail`** — Records bindings for backtracking.

---

## `pyladr.inference` — Inference Rules

All rules are re-exported from `pyladr.inference`.

### Resolution

**Module:** `pyladr.inference.resolution`

```python
# Resolve two clauses on complementary literals
binary_resolve(c1: Clause, i1: int, c2: Clause, i2: int) -> Clause | None

# Generate all resolvents between two clauses
all_binary_resolvents(c1: Clause, c2: Clause) -> list[Clause]

# Factor a clause (unify two literals of the same sign)
factor(c: Clause) -> list[Clause]

# Check if clause is a tautology
is_tautology(c: Clause) -> bool

# Merge duplicate literals
merge_literals(c: Clause) -> Clause

# Standardize variables apart
renumber_variables(c: Clause) -> Clause
```

### Hyper-Resolution

**Module:** `pyladr.inference.hyper_resolution`

```python
# Multi-premise resolution (resolve all negative literals at once)
hyper_resolve(nucleus: Clause, satellites: list[Clause]) -> Clause | None

# Generate all hyper-resolvents from a given clause and usable set
all_hyper_resolvents(given: Clause, usable: list[Clause]) -> list[Clause]
```

### Paramodulation

**Module:** `pyladr.inference.paramodulation`

```python
# Substitute equals for equals at a position
paramodulate(from_c: Clause, from_lit: int, from_side: int,
             into_c: Clause, into_lit: int, into_pos: tuple[int, ...]) -> Clause | None

# Generate all paramodulants between two clauses
para_from_into(from_c: Clause, into_c: Clause,
               symbol_table: SymbolTable) -> list[Clause]

# Orient equalities using term ordering
orient_equalities(c: Clause, symbol_table: SymbolTable) -> None

# Equality predicates
is_eq_atom(t: Term) -> bool
pos_eq(lit: Literal) -> bool
neg_eq(lit: Literal) -> bool
```

### Demodulation

**Module:** `pyladr.inference.demodulation`

```python
# Simplify a term using demodulator equations
demodulate_term(t: Term, index: DemodulatorIndex) -> Term

# Simplify all terms in a clause
demodulate_clause(c: Clause, index: DemodulatorIndex) -> Clause

# Find clauses that can be back-demodulated by a new demodulator
back_demodulatable(demod: Clause, clauses: list[Clause],
                   index: DemodulatorIndex) -> list[Clause]

# Check if a clause qualifies as a demodulator
demodulator_type(c: Clause) -> DemodType
```

**`DemodType` enum:** `FORWARD`, `BACK`, `UNIT`

**`DemodulatorIndex`** — Index structure for efficient demodulator lookup.

### Subsumption

**Module:** `pyladr.inference.subsumption`

```python
# Check if c1 subsumes c2 (c1 is more general)
subsumes(c1: Clause, c2: Clause) -> bool

# Forward subsumption: is the new clause subsumed by any existing clause?
forward_subsume(new_clause: Clause, clause_lists: list[list[Clause]]) -> bool

# Backward subsumption: remove existing clauses subsumed by the new clause
back_subsume(new_clause: Clause, clause_list: list[Clause]) -> list[Clause]
```

---

## `pyladr.search` — Search Algorithms

### `GivenClauseSearch`

**Module:** `pyladr.search.given_clause`

The main search engine implementing the Otter/Prover9 given-clause algorithm.

```python
class GivenClauseSearch:
    def __init__(
        self,
        options: SearchOptions | None = None,
        selection: GivenSelection | None = None,
        symbol_table: SymbolTable | None = None,
    ) -> None: ...

    def run(
        self,
        usable: list[Clause] | None = None,
        sos: list[Clause] | None = None,
    ) -> SearchResult: ...

    @property
    def state(self) -> SearchState: ...

    @property
    def stats(self) -> SearchStatistics: ...

    def set_back_subsumption_callback(
        self, callback: Callable[[Clause, Clause], None]
    ) -> None: ...

    def set_forward_subsumption_callback(
        self, callback: Callable[[Clause, Clause], None]
    ) -> None: ...
```

### `SearchOptions`

```python
@dataclass(slots=True)
class SearchOptions:
    # Inference rules
    binary_resolution: bool = True
    hyper_resolution: bool = False
    paramodulation: bool = False
    factoring: bool = True
    para_into_vars: bool = False

    # Limits (-1 = no limit)
    max_given: int = -1
    max_kept: int = -1
    max_seconds: float = -1.0
    max_generated: int = -1
    max_proofs: int = 1

    # Demodulation
    demodulation: bool = False
    back_demod: bool = False
    demod_step_limit: int = 1000

    # Weight / SOS limits
    max_weight: float = -1.0
    sos_limit: int = -1

    # Simplification
    check_tautology: bool = True
    merge_lits: bool = True

    # Parallelization
    parallel: ParallelSearchConfig | None = None

    # Machine Learning
    online_learning: bool = False
    ml_weight: float | None = None
    embedding_dim: int = 32

    # Goal-directed selection
    goal_directed: bool = False
    goal_proximity_weight: float = 0.3
    embedding_evolution_rate: float = 0.01
    learn_from_back_subsumption: bool = False
    learn_from_forward_subsumption: bool = False

    # Output
    print_given: bool = True
    print_kept: bool = False
    quiet: bool = False
```

### `SearchResult` / `Proof`

```python
@dataclass(frozen=True, slots=True)
class SearchResult:
    exit_code: ExitCode
    proofs: tuple[Proof, ...]
    stats: SearchStatistics

@dataclass(frozen=True, slots=True)
class Proof:
    empty_clause: Clause
    clauses: tuple[Clause, ...]
```

### `ExitCode`

```python
class ExitCode(IntEnum):
    MAX_PROOFS_EXIT = 1     # Proof found
    SOS_EMPTY_EXIT = 2      # Search exhausted
    MAX_GIVEN_EXIT = 3      # Given clause limit
    MAX_KEPT_EXIT = 4       # Kept clause limit
    MAX_SECONDS_EXIT = 5    # Time limit
    MAX_GENERATED_EXIT = 6  # Generated clause limit
    FATAL_EXIT = 7          # Fatal error
```

### `SearchStatistics`

**Module:** `pyladr.search.statistics`

Tracks clause counts and timing during search.

- `given` — Number of given clauses selected
- `generated` — Total inferences generated
- `kept` — Clauses retained after filtering
- `deleted` — Clauses removed by subsumption/weight
- `elapsed_seconds()` — Wall-clock time since search start

### `GivenSelection`

**Module:** `pyladr.search.selection`

Manages given clause selection strategy (age vs. weight ratio).

```python
class GivenSelection:
    def select_given(sos: ClauseList, given_count: int) -> tuple[Clause | None, str]: ...
    def add_clause_to_selectors(c: Clause) -> None: ...
```

---

## `pyladr.parsing` — LADR Parser

### `LADRParser`

**Module:** `pyladr.parsing.ladr_parser`

Recursive-descent parser with precedence climbing for LADR syntax.

```python
class LADRParser:
    def __init__(self, symbol_table: SymbolTable | None = None) -> None: ...

    def parse_input(self, text: str) -> ParsedInput: ...
    def parse_term(self, text: str) -> Term: ...
    def parse_clause(self, text: str) -> Clause: ...

    @property
    def symbol_table(self) -> SymbolTable: ...
```

### `ParsedInput`

```python
@dataclass
class ParsedInput:
    sos: list[Clause]         # Set of Support clauses
    usable: list[Clause]      # Usable clauses
    goals: list[Clause]       # Goal clauses (not yet negated)
    goals_denied: list[Clause]  # Negated goal clauses
    demodulators: list[Clause]  # Explicit demodulators
    settings: dict[str, SettingType]  # Parsed settings
```

### `ParseError`

```python
class ParseError(Exception):
    pos: int  # Character position of error (-1 if unknown)
```

---

## `pyladr.ordering` — Term Orderings

### Dispatch

**Module:** `pyladr.ordering.termorder`

```python
def term_greater(alpha: Term, beta: Term, lex_order_vars: bool,
                 st: SymbolTable) -> bool: ...

def term_order(alpha: Term, beta: Term, st: SymbolTable) -> Ordertype: ...

def assign_order_method(method: OrderMethod) -> None: ...
def get_order_method() -> OrderMethod: ...
```

### `Ordertype` enum

`GREATER_THAN`, `LESS_THAN`, `SAME_AS`, `NOT_COMPARABLE`

### `OrderMethod` enum

`KBO` (Knuth-Bendix Ordering), `LRPO` (Lexicographic Recursive Path Ordering)

### Individual Orderings

```python
# pyladr.ordering.kbo
def kbo(alpha: Term, beta: Term, lex_order_vars: bool,
        st: SymbolTable) -> bool: ...

# pyladr.ordering.lrpo
def lrpo(alpha: Term, beta: Term, lex_order_vars: bool,
         st: SymbolTable) -> bool: ...
```

---

## `pyladr.indexing` — Term Indexing

### `Mindex`

**Module:** `pyladr.indexing.discrimination_tree`

Unified index interface matching C `mindex.c`. Wraps either `DiscrimWild` (fast, imperfect) or `DiscrimBind` (slower, exact).

```python
class Mindex:
    def __init__(self, index_type: IndexType = IndexType.WILD) -> None: ...
    def insert(self, t: Term, data: object) -> None: ...
    def delete(self, t: Term, data: object) -> None: ...
    def retrieve_generalizations(self, t: Term) -> Iterator: ...
    def retrieve_instances(self, t: Term) -> Iterator: ...
    def retrieve_unifiables(self, t: Term) -> Iterator: ...
```

**`IndexType` enum:** `WILD` (imperfect, fast), `BIND` (perfect, with substitutions)

### `FeatureIndex`

**Module:** `pyladr.indexing.feature_index`

Feature vector indexing for subsumption prefiltering.

### `LiteralIndex`

**Module:** `pyladr.indexing.literal_index`

Positive/negative literal pair of `Mindex` instances, matching C `lindex.c`.

---

## `pyladr.mace4` — Finite Model Finder

### `ModelSearcher`

**Module:** `pyladr.mace4.search`

Backtracking search with constraint propagation for finding finite models.

```python
class ModelSearcher:
    def __init__(
        self,
        clauses: list[Clause],
        symbol_table: SymbolTable,
        options: SearchOptions | None = None,
    ) -> None: ...

    def search(self) -> ModelResult: ...
```

### `SearchOptions` (Mace4)

```python
@dataclass(slots=True)
class SearchOptions:
    start_size: int = 2       # Minimum domain size
    end_size: int = 10        # Maximum domain size
    max_models: int = 1       # Stop after N models
    max_seconds: float = 60.0 # Time limit
    increment: int = 1        # Domain size step
    print_models: bool = True
```

### `FiniteModel` / `ModelResult`

```python
@dataclass
class FiniteModel:
    domain_size: int
    cells: dict[int, Cell]
    symbols: dict[str, SymbolInfo]

@dataclass
class ModelResult:
    models: list[FiniteModel]
    domain_sizes_tried: list[int]
    exhausted: bool
```

---

## `pyladr.parallel` — Parallel Inference Engine

### `ParallelInferenceEngine`

**Module:** `pyladr.parallel.inference_engine`

Thread pool for parallel inference generation. Requires Python 3.14+ free-threading for actual parallelism; falls back to sequential on GIL Python.

```python
class ParallelInferenceEngine:
    def __init__(self, config: ParallelSearchConfig | None = None) -> None: ...
    def generate_inferences(
        self, given: Clause, usable: list[Clause], ...
    ) -> list[Clause]: ...
```

### `ParallelSearchConfig`

```python
@dataclass(frozen=True, slots=True)
class ParallelSearchConfig:
    enabled: bool = True
    max_workers: int | None = None  # None = cpu_count
    min_usable_for_parallel: int = 50
    chunk_size: int = 25
    parallel_back_subsumption: bool = True
    min_clauses_for_parallel_back: int = 100
```

### Thread Safety Model

- **Usable list:** Read-only snapshot during inference generation
- **Context/Trail:** Thread-local per worker
- **Index updates:** Sequential after parallel generation
- **Statistics:** Atomic counters
- **Detection:** `FREE_THREADING_AVAILABLE` (from `pyladr.threading_guide`)

---

## Command-Line Applications

All tools are installed as entry points via `pyproject.toml`.

### Main Prover

| Command | Description |
|---------|-------------|
| `pyprover9` | Main theorem prover |
| `pyprover9-mace4` | Combined prover + model finder |

### Proof & Clause Manipulation

| Command | Description |
|---------|-------------|
| `pyprooftrans` | Proof transformation and analysis |
| `pyrenamer` | Variable renaming / standardization |
| `pyclausefilter` | Filter clauses by criteria |
| `pyclausetester` | Test clauses against conditions |
| `pyrewriter` / `pyrewriter2` | Term rewriting |

### Model & Interpretation Tools

| Command | Description |
|---------|-------------|
| `pyinterpformat` | Format interpretations for display |
| `pyinterpfilter` | Filter interpretations |
| `pyisofilter` / `pyisofilter0` / `pyisofilter2` | Isomorphism filtering |
| `pydprofiles` | Discriminator profiles |

### Algebraic Structure Tools

| Command | Description |
|---------|-------------|
| `pymirror-flip` | Lattice duality (order-dual) |
| `pyperm3` | 3-element permutations |
| `pylatfilter` | Lattice property filtering |
| `pyolfilter` | Ortholattice filtering |
| `pyupper-covers` | Compute upper covers |

### Analysis & Extraction

| Command | Description |
|---------|-------------|
| `pyget-givens` | Extract given clauses from output |
| `pyget-interps` | Extract interpretations |
| `pyget-kept` | Extract kept clauses |
| `pysigtest` | Signature testing |
| `pyidfilter` | Identity filtering |

### Conversion & Miscellaneous

| Command | Description |
|---------|-------------|
| `pyladr-to-tptp` | Convert LADR format to TPTP |
| `pyattack` | Substructure search |
| `pylooper` | Iterative search |
| `pyminiscope` | Miniscoping quantifiers |
| `pyunfast` | Unfast operation |
| `pycomplex` | Complexity analysis |
| `pydirectproof` | Direct proof extraction |
| `pygen-trc-defs` | Generate transitive closure definitions |
