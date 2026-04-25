# ML Weight Command-Line Control

## New Feature: --ml-weight Option

You can now control ML influence directly from the command line without editing source code!

## Usage Examples

### Fixed ML Influence
```bash
# Light ML influence (10% ML, 90% traditional)
python3 -m pyladr.apps.prover9 --online-learning --ml-weight 0.1 -f problem.in

# Moderate ML influence (30% ML, 70% traditional)
python3 -m pyladr.apps.prover9 --online-learning --ml-weight 0.3 -f problem.in

# Balanced blend (50% ML, 50% traditional)
python3 -m pyladr.apps.prover9 --online-learning --ml-weight 0.5 -f problem.in

# Heavy ML influence (80% ML, 20% traditional)
python3 -m pyladr.apps.prover9 --online-learning --ml-weight 0.8 -f problem.in

# Pure ML (experimental - 100% ML guidance)
python3 -m pyladr.apps.prover9 --online-learning --ml-weight 1.0 -f problem.in

# Pure traditional (0% ML - for comparison)
python3 -m pyladr.apps.prover9 --online-learning --ml-weight 0.0 -f problem.in
```

### Default Adaptive Behavior
```bash
# Uses adaptive weight (0.1 → 0.5 as system learns)
python3 -m pyladr.apps.prover9 --online-learning -f problem.in
```

## What You'll See

With `--ml-weight 0.6`:
```
🧠 Online learning enabled - using real-time contrastive learning
🎛️ Using fixed ML weight: 0.60 (command line)
given #8 (T+ML,wt=10): 17: P(i(i(n(x),x),i(n(x),y))).
```

Without `--ml-weight` (adaptive):
```
🧠 Online learning enabled - using real-time contrastive learning
🎛️ Using adaptive ML weight: 0.1 → 0.5 (default)
```

## Parameter Range

- **0.0**: Pure traditional selection (no ML)
- **0.1-0.3**: Conservative ML guidance
- **0.4-0.6**: Balanced ML/traditional blend
- **0.7-0.9**: Aggressive ML influence
- **1.0**: Pure ML selection (experimental)

## Key Benefits

✅ No source code editing required
✅ Easy experimentation with different influence levels
✅ Perfect for tuning ML aggressiveness per problem type
✅ Maintains backward compatibility (default behavior unchanged)