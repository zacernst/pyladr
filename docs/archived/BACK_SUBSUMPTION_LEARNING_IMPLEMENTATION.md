# Back-Subsumption Learning Implementation

## Overview

Successfully implemented a command-line option for back-subsumption feedback to the online ML algorithm. When a formula back-subsumes other formulas, this indicates that its structure is particularly useful for generalization, making it valuable positive feedback for the machine learning system.

## Implementation Details

### 1. Command-Line Interface
- **New argument**: `--learn-from-back-subsumption`
- **Location**: Goal-directed selection argument group in `prover9.py`
- **Usage**: `python3 -m pyladr.cli --online-learning --learn-from-back-subsumption -f input.in`
- **Description**: "Enable positive feedback to online ML when clauses back-subsume others (indicates structurally useful clause patterns)"

### 2. SearchOptions Integration
- **File**: `pyladr/search/given_clause.py`
- **New field**: `learn_from_back_subsumption: bool = False`
- **Purpose**: Controls whether back-subsumption events trigger ML feedback

### 3. Callback Mechanism in GivenClauseSearch
- **New slot**: `_back_subsumption_callback` in `__slots__`
- **Initialization**: Set to `None` in constructor
- **Setter method**: `set_back_subsumption_callback(callback)`
- **Integration point**: Modified `_limbo_process()` method at back-subsumption detection

### 4. Back-Subsumption Detection Integration
**Location**: `GivenClauseSearch._limbo_process()` method around line 745

```python
if nc <= victim.num_literals and subsumes(c, victim):
    self._state.stats.back_subsumed += 1
    # ... existing back-subsumption handling ...

    # NEW: Notify online learning if enabled
    if (self._opts.learn_from_back_subsumption and
        self._back_subsumption_callback is not None):
        self._back_subsumption_callback(c, victim)
```

### 5. Online Learning Integration
**File**: `pyladr/search/online_integration.py`

#### New Method: `on_back_subsumption()`
```python
def on_back_subsumption(self, subsuming_clause: Clause, subsumed_clause: Clause) -> None:
    """Called when a clause back-subsumes another clause."""
    if not self._enabled or self._manager is None:
        return

    # Record as positive outcome (twice for extra weight)
    outcome = InferenceOutcome(
        given_clause=subsuming_clause,
        partner_clause=None,
        child_clause=subsuming_clause,  # The successful clause
        outcome=OutcomeType.KEPT,       # Positive signal
        timestamp=time.monotonic(),
        given_count=self._progress_tracker._given_count,
    )

    self._manager.record_outcome(outcome)
    self._manager.record_outcome(outcome)  # Double positive weight
    self._stats.experiences_collected += 2
```

#### Callback Registration
In `_OnlineLearningGivenClauseSearch.__new__()`:
```python
# Set up back-subsumption callback for online learning
if hasattr(instance, 'set_back_subsumption_callback'):
    instance.set_back_subsumption_callback(integration.on_back_subsumption)
```

## How It Works

### Normal Search Flow
1. User runs: `python3 -m pyladr.cli --online-learning --learn-from-back-subsumption -f input.in`
2. `SearchOptions.learn_from_back_subsumption = True` is set
3. When online learning is enabled, `OnlineSearchIntegration` registers its callback
4. During proof search in `_limbo_process()`:
   - When `subsumes(c, victim)` returns `True` (back-subsumption detected)
   - Statistics counter incremented: `stats.back_subsumed += 1`
   - If learning enabled, callback invoked: `callback(subsuming_clause, victim)`
5. Online learning receives positive signal for the subsuming clause
6. ML model learns that this clause structure is valuable for generalization

### Key Benefits
- **Positive Reinforcement**: Clauses that back-subsume others get boosted in the ML model
- **Generalization Learning**: Identifies structurally useful patterns for finding general clauses
- **Double Weight**: Back-subsumption outcomes are recorded twice for extra emphasis
- **Non-Intrusive**: Only affects online learning when explicitly enabled
- **Backward Compatible**: No impact when flag is disabled

### Technical Notes
- Back-subsumption events are relatively rare but highly informative
- They occur when a newly derived clause is more general than existing clauses
- The feature integrates seamlessly with the existing online learning infrastructure
- Logging available when `log_integration_events=True` in OnlineIntegrationConfig

## Testing
- ✅ CLI argument parsing works correctly
- ✅ SearchOptions integration functional
- ✅ Callback mechanism properly registered
- ✅ OnlineSearchIntegration method implemented
- ✅ Detection point properly integrated
- ✅ Complete workflow tested

The implementation is ready for production use. Back-subsumption events will provide valuable positive feedback to the online learning system when they occur during theorem proving.