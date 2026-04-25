# Online Learning Logging System

## Overview

Enhanced logging has been added to PyLADR's online learning system to provide real-time visibility into the learning process during theorem proving.

## Logging Features

### 1. **Initialization Logging** 📚
Shows learning system configuration at startup:
```
📚 Online learning initialized: buffer capacity 5000, min update threshold 50 (adaptive 0.1→0.5)
```

### 2. **Search Startup** 🚀
Confirms that online learning hooks are active:
```
🚀 Starting proof search with online contrastive learning hooks active
```

### 3. **Experience Collection Progress** 💾
Shows when experiences are collected and added to the learning buffer:
```
💾 Experience #50: KEPT clause from given 12 → buffer: 50/5000
💾 Experience #100: KEPT clause from given 23 → buffer: 100/5000
```

### 4. **Learning Trigger Evaluation** 📈
Shows when the system evaluates whether to trigger model updates:
```
📈 Learning eval: 15 new experiences, 65 total → waiting (given #20)
📈 Learning eval: 32 new experiences, 82 total → TRIGGERING (given #30)
```

### 5. **Model Update Results** 🧠
Shows the outcome of model training attempts:

**Successful Updates:**
```
🧠 Model update #1 ✅ accepted (0.12s) | ML weight: 0.10→0.15 | Buffer: 75 experiences
```

**Failed Updates (Rollbacks):**
```
🧠 Model update #2 ❌ rolled back (0.08s) | ML weight: 0.15→0.10 | Cooldown: 5
```

### 6. **ML Weight Configuration** 🎛️
Shows the ML influence setting at startup:
```
🎛️ Using fixed ML weight: 0.50 (command line)
🎛️ Using adaptive ML weight: 0.1 → 0.5 (default)
```

### 7. **Selection Behavior** 🎯
ML-guided clause selections are already visible in the standard output:
```
given #8 (T+ML,wt=10): 17: P(i(i(n(x),x),i(n(x),y))).
given #10 (T+ML,wt=4): 27: P(i(x,x)).
```

## Usage

### Enable Logging
All logging is enabled by default when using `--online-learning`. No additional flags needed.

### Example Output
```bash
python3 -m pyladr.apps.prover9 --online-learning --ml-weight 0.4 -f problem.in -max_given 20
```

```
🧠 Online learning enabled - using real-time contrastive learning
🎛️ Using fixed ML weight: 0.40 (command line)
📚 Online learning initialized: buffer capacity 5000, min update threshold 50
🚀 Starting proof search with online contrastive learning hooks active
given #1 (T,wt=7): 3: P(i(i(n(x),x),x)).
given #2 (T,wt=7): 4: P(i(x,i(n(x),y))).
💾 Experience #50: KEPT clause from given 15 → buffer: 50/5000
📈 Learning eval: 25 new experiences, 75 total → TRIGGERING (given #18)
🧠 Model update #1 ✅ accepted (0.15s) | ML weight: 0.40→0.40 | Buffer: 75 experiences
given #19 (T+ML,wt=8): 45: P(i(i(x,y),i(x,y))).
```

## Logging Frequency

- **Initialization**: Once at startup
- **Experience collection**: Every 50 experiences
- **Learning evaluation**: Every 10 given clauses (when experiences > 0)
- **Model updates**: Every update attempt
- **ML selections**: Every ML-guided clause selection

## Benefits

1. **Transparency**: See exactly what the learning system is doing
2. **Debugging**: Identify why learning may not be triggering
3. **Performance**: Monitor learning frequency and success rates
4. **Verification**: Confirm ML is actually affecting clause selection
5. **Tuning**: Understand when to adjust parameters like `--ml-weight`

## Advanced Logging

For even more detailed logging, set the integration config:
```python
config = OnlineIntegrationConfig(
    log_integration_events=True  # Enables detailed debug logging
)
```

This adds additional technical logging to the standard Python logger at INFO level.