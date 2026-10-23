# Results and Methodology — UAV Strategy Engine

This document records the implemented method and the **actual measured
results** from this machine. It is report-support material, not the final
3-5 page submission.

- Machine: Ubuntu 24.04, Python 3.12.3, NumPy 1.26.4, Matplotlib 3.6.3
- Project root: `uav-strategy-engine/`
- Date of runs: 2026-10-23
- Every number below comes from a generated file under `results/`.

---

## 1. Reproducibility commands

```bash
cd uav-strategy-engine

# Default GA (population 60, 50 generations, seed 42)
python3 main.py --mode ga --quiet

# Sensitivity-guided tuned GA (population 100, 50 generations)
python3 main.py --mode ga --population 100 --generations 50 --quiet

# Equal-budget random baseline (3000 evaluations)
python3 main.py --mode random --quiet

# Fair comparison, 3 seeds, 3000 evaluations per method per seed
python3 main.py --mode compare --seeds 3 --quiet

# One-factor-at-a-time sensitivity, 2 seeds, 30 generations
python3 main.py --mode sensitivity --quiet
```

Final run directories:

| Run | Directory |
|---|---|
| GA baseline | `results/run_20261023-234357_ga_seed42/` |
| GA tuned | `results/run_20261023-234420_ga_seed42/` |
| Random baseline | `results/run_20261023-234511_random_seed42/` |
| GA vs random | `results/run_20261023-234528_compare_seed42/` |
| Sensitivity | `results/run_20261023-233431_sensitivity_seed42/` |

---

## 2. Environment instance

The two GA demonstration runs and the seed-42 random baseline use the same
deterministic environment:

| Property | Value |
|---|---|
| Grid | 50 x 50 cells |
| Obstacle cells | 467 |
| No-fly cells | 217 |
| Hazard mean / max | 0.3654 / 1.0 |
| Targets | 14 |
| Target priorities | 0.516 - 0.975 |
| Drone starts | `(2, 11)`, `(23, 29)`, `(42, 5)` |
| Drones | 3 |
| Waypoints per drone | 6 |
| Battery capacity / drone | 260.0 |
| Max flight range / drone | 110.0 cells |
| Time reference | 120.0 |

Target coordinates (same in every run):

| ID | (x, y) | Priority | ID | (x, y) | Priority |
|---|---|---|---|---|---|
| T0 | (12, 12) | 0.975 | T7 | (10, 44) | 0.807 |
| T1 | (23, 14) | 0.706 | T8 | (17, 17) | 0.616 |
| T2 | (26, 19) | 0.834 | T9 | (30, 16) | 0.833 |
| T3 | (12, 47) | 0.728 | T10 | (22, 27) | 0.676 |
| T4 | (14, 45) | 0.877 | T11 | (16, 33) | 0.943 |
| T5 | (12, 35) | 0.516 | T12 | (15, 19) | 0.910 |
| T6 | (43, 0) | 0.572 | T13 | (37, 23) | 0.746 |

These values are simulation inputs, not real UAV specifications.

---

## 3. Method

### 3.1 Chromosome

A chromosome contains one route for each of the three drones. Each route has
six waypoints. A waypoint is:

```
(x, y, speed_mode, sensor_mode)
```

with `speed_mode in {0, 1, 2}` and `sensor_mode in {0, 1, 2}`. The launch cell is
fixed and precedes waypoint 0. Route length is fixed so crossover and mutation
always produce structurally compatible candidates.

### 3.2 Simulation

Every straight leg is sampled at 0.5-cell resolution. A leg is feasible only if
every sample is inside the grid and outside obstacles and no-fly zones. The
evaluator additionally checks per-drone range and battery limits. Feasibility is
computed from the trajectory, independently of the scalar fitness.

Costs:

```
movement_energy = distance * 1.0 * (speed / cruise_speed)^2
sensor_energy   = leg_time * sensor_energy_rate
energy          = movement_energy + sensor_energy
duration        = sum(leg_distance / leg_speed)
mission_time    = max(duration over drones)
```

Drones run in parallel, which is why mission time is the maximum rather than the
sum.

### 3.3 Metrics

```
coverage       = priority-weighted fraction of detected targets
detection_rate = detected_targets / total_targets
normalized_energy = total_energy / (3 * 260)
normalized_time   = mission_time / 120
normalized_risk   = hazard_exposure / total_distance
```

Coverage uses a union across drones, so duplicate detections do not inflate it.

### 3.4 Fitness

```
raw = 100 * coverage
    +  50 * detection_rate
    -  20 * normalized_energy
    -  15 * normalized_time
    -  30 * normalized_risk
```

Infeasible candidates receive:

```
penalty = 1000 + 50 * violation_count + 25 * sum(min(magnitude, 10))
fitness = raw - penalty
```

Selection and elitism use the tuple `(feasible, fitness)`, so infeasible
candidates cannot beat feasible ones.

### 3.5 Genetic algorithm

| Component | Implementation |
|---|---|
| Initialisation | Ray-based routes; 60% target-directed with jitter; rest random |
| Selection | Tournament, size 3 by default |
| Crossover | One-point suffix exchange for the same drone; per-route probability 0.5 |
| Mutation | Per-waypoint probability 0.05; 60% move, 20% speed, 20% sensor |
| Repair | Snap, local nudge, safe route regeneration |
| Elitism | 2 feasible candidates per generation |
| Cache | chromosome-signature evaluation cache (exact repeat candidates) |
| Stop | Generation limit (50); optional stagnation limit |

The cache counts both requests and unique evaluations. In the tuned run, 5000
requests produced 4135 unique evaluations and 865 cache hits. Feasibility rate is
reported over unique evaluations, matching the random-search definition.

---

## 4. GA baseline versus random search

### 4.1 Same seed, same budget (seed 42, 3000 evaluations)

| Metric | GA (60 x 50) | Random search | GA difference |
|---|---:|---:|---:|
| Best fitness | **122.145** | 105.643 | +16.502 |
| Priority-weighted coverage | **94.67%** | 88.10% | +6.57 pp |
| Detection rate | **92.86%** | 85.71% | +7.14 pp |
| Total energy | **173.08** | 311.03 | -137.94 |
| Mission time | **50.18** | 53.97 | -3.79 |
| Hazard exposure | **37.86** | 55.41 | -17.55 |
| Feasibility rate | **99.83%** | 98.87% | +0.97 pp |
| Runtime | 20.38 s | 15.95 s | +4.43 s |

The GA wins every mission-quality metric at the same evaluation budget. It uses
more wall-clock time because selection, crossover, mutation, and repair add
overhead per evaluation.

### 4.2 Multi-seed comparison (3 seeds, 3000 evaluations per method per seed)

Per-seed results (`comparison.csv`):

| Method | Seed | Fitness | Coverage | Detection | Energy | Time | Hazard | Feasible |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| GA | 42 | 122.145 | 94.67% | 92.86% | 173.08 | 50.18 | 37.86 | yes |
| Random | 42 | 105.643 | 88.10% | 85.71% | 311.03 | 53.97 | 55.41 | yes |
| GA | 43 | 123.933 | 100.00% | 100.00% | 346.96 | 85.43 | 49.02 | yes |
| Random | 43 | 73.880 | 67.48% | 71.43% | 355.85 | 98.34 | 56.60 | yes |
| GA | 44 | 111.210 | 83.73% | 85.71% | 239.53 | 51.19 | 14.72 | yes |
| Random | 44 | 89.265 | 76.08% | 78.57% | 401.95 | 81.15 | 43.53 | yes |

Mean +/- standard deviation over seeds (`comparison_summary.csv`):

| Metric | GA | Random search |
|---|---:|---:|
| Best fitness | **119.096 +/- 6.888** | 89.596 +/- 15.884 |
| Coverage | **92.80%** | 77.22% |
| Detection rate | **92.86%** | 78.57% |
| Total energy | **253.19** | 356.28 |
| Mission time | **62.27** | 77.82 |
| Hazard exposure | **33.87** | 51.85 |
| Feasibility rate | **98.77%** | 96.39% |
| Runtime | 21.63 s | 16.97 s |
| Evaluations | 3000 | 3000 |

The GA won all three seed-level comparisons. Random search had higher variance
(fitness SD 15.88 vs 6.89), consistent with selection maintaining a stronger
population.

---

## 5. Sensitivity-guided tuning

Sensitivity settings: 30 generations, seeds 42 and 43, one factor changed at a
time. Each row is the mean of two runs. The full 18-run table is in
`sensitivity_results.csv`; means are in `sensitivity_summary.csv`.

### 5.1 Population size

| Population | Mean fitness | SD | Coverage | Energy | Time | Runtime | Evaluations/run |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 123.200 | 4.536 | 97.34% | 210.69 | 59.44 | 10.22 s | 1500 |
| 100 | 127.030 | 0.509 | 100.00% | 231.05 | 57.60 | 24.06 s | 3000 |
| 200 | **127.915** | 0.334 | 100.00% | 208.09 | 55.53 | 51.48 s | 6000 |

Increasing population from 50 to 100 removed all missed targets in both seeds
and sharply reduced variance. Population 200 gained only +0.885 mean fitness
over 100 while doubling the evaluation budget, so 100 is the better
cost/quality choice.

### 5.2 Mutation rate

| Mutation rate | Mean fitness | SD | Coverage | Energy | Time | Hazard |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 123.937 | 5.868 | 97.34% | 215.83 | 52.02 | 46.51 |
| 0.05 | **125.299** | 6.450 | 97.34% | 207.13 | 50.13 | 39.02 |
| 0.10 | 120.732 | 0.573 | 94.67% | 212.95 | 50.22 | 37.79 |

`0.05` gives the best mean fitness and the lowest energy/time among the tested
values. `0.10` is more stable across seeds but finds lower-quality coverage.
The large SD at `0.05` is caused by seed-to-seed variation, not instability of
the operator itself.

### 5.3 Tournament size

| Tournament size | Mean fitness | SD | Coverage | Energy | Time | Hazard |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 122.109 | 2.861 | 100.00% | 356.00 | 74.10 | 65.28 |
| 3 | **125.299** | 6.450 | 97.34% | 207.13 | 50.13 | 39.02 |
| 5 | 124.701 | 4.190 | 97.34% | 189.44 | 53.69 | 42.74 |

Weak pressure (`k=2`) reaches full coverage but pays for it with much higher
energy, time, and hazard exposure. `k=3` gives the best overall fitness; `k=5`
uses less energy but slightly more time and risk. This is a useful
multi-objective trade-off rather than a single "best" setting.

### 5.4 Tuned configuration

The sensitivity results support population 100, mutation 0.05, tournament 3.
A 50-generation run with that configuration produced:

| Metric | GA tuned (100 x 50) | GA baseline (60 x 50) |
|---|---:|---:|
| Best fitness | **128.617** | 122.145 |
| Priority-weighted coverage | **100.00%** | 94.67% |
| Detection rate | **100.00%** | 92.86% |
| Total energy | 216.04 | 173.08 |
| Normalized energy | 0.27697 | 0.22190 |
| Mission time | 49.88 | 50.18 |
| Normalized time | 0.41568 | 0.41815 |
| Hazard exposure | 46.71 | 37.86 |
| Normalized risk | 0.32027 | 0.27490 |
| Feasible | yes | yes |
| Evaluations / unique | 5000 / 4135 | 3000 / 2410 |
| Cache hits | 865 | 590 |
| Runtime | 38.27 s | 20.38 s |

The tuned run detects all 14 targets. The trade-off is higher energy use and
higher hazard exposure: covering T6 in the top-right corner and the remaining
targets requires longer, riskier routes. This is exactly the behavior the
weighted fitness is meant to expose.

Per-drone detail, tuned run:

| Drone | Start | Distance | Duration | Energy | Hazard | Targets |
|---|---:|---:|---:|---:|---:|---|
| 0 | (2, 11) | 47.38 | 49.88 | 56.43 | 9.96 | T0, T8, T12 |
| 1 | (23, 29) | 39.80 | 49.53 | 47.66 | 17.33 | T3, T4, T5, T7, T10, T11 |
| 2 | (42, 5) | 58.65 | 48.58 | 111.94 | 19.41 | T1, T2, T6, T9, T13 |

---

## 6. Convergence

GA baseline (`fitness_history.csv`):

| Generation | Best feasible | Mean feasible | Best coverage | Feasible count |
|---:|---:|---:|---:|---:|
| 0 | 82.724 | 27.099 | 72.21% | 59 / 60 |
| 49 | 122.145 | 116.659 | 94.67% | 60 / 60 |

GA tuned:

| Generation | Best feasible | Mean feasible | Best coverage | Feasible count |
|---:|---:|---:|---:|---:|
| 0 | 82.724 | 25.551 | 72.21% | 99 / 100 |
| 49 | 128.617 | 126.234 | 100.00% | 100 / 100 |

Both runs show the expected rapid early improvement followed by slow
fine-tuning. The tuned population has a much higher mean fitness at generation
49 (126.23), showing that most of the population converged near the best
solution rather than relying on a single lucky elite.

---

## 7. Automated test evidence

Command:

```bash
python3 -m pytest tests/ -q
```

Result: **44 passed**.

| Test module | Passing tests | Focus |
|---|---:|---|
| `tests/test_environment.py` | 5 | determinism, dimensions, target/start validity, impossible layouts |
| `tests/test_chromosome.py` | 8 | structure, copy isolation, serialisation, validation |
| `tests/test_constraints.py` | 7 | obstacles/no-fly on waypoints and segments, bounds, range, battery |
| `tests/test_evaluator.py` | 12 | coverage, duplicate detection, energy/time/risk, deterministic metrics |
| `tests/test_genetic_operators.py` | 7 | crossover, mutation, parent immutability, tournament preference |
| `tests/test_optimizer.py` | 5 | history length, elitism monotonicity, seeding, no-feasible runs |

No test reaches the network or depends on the display; Matplotlib uses Agg.

---

## 8. Interpretation

- The GA clearly outperforms equal-budget random search in this environment:
  higher fitness, coverage, and detection, with lower energy, time, and hazard
  exposure on average.
- Selection plus elitism is doing real work; random search still finds feasible
  routes because it shares the same target-directed initialisation and repair.
- Covering the last target is expensive. The tuned 100% solution spends more
  energy and accepts more hazard than the 94.7% baseline.
- Population size was the most influential tested parameter. Mutation rate and
  tournament size changed the quality/cost balance but not as strongly.
- Two seeds per sensitivity setting are enough to expose trends, not enough for
  statistical significance. The reported SDs should be treated as descriptive.
- Wall-clock runtime is not the comparison metric; both methods receive the same
  evaluation budget, and the GA spends extra time on genetic operators.

---

## 9. Limitations

- All units and dynamics are simulated; this is not a flight-control model.
- Deterministic sensing and straight-leg flight omit wind, sensing uncertainty,
  and real collision avoidance.
- The chromosome has a fixed number of waypoints; variable-length routes are not
  explored.
- The environment generator is tuned for the default 50 x 50 layout. Very small
  custom grids can fail to place all requested targets and raise a clear error.
- Sensitivity is one-factor-at-a-time; interactions between parameters are not
  measured unless `--full-factorial` is used.
- Results are seed-dependent. The multi-seed comparison reduces this risk, but
  three seeds are still a small sample.
- The fitness weights are a design choice. Different weights will produce
  different "best" strategies; the evaluator reports component metrics so the
  trade-off remains visible.

---

## 10. Report-ready mapping

| Assignment item | Evidence |
|---|---|
| Chromosome design | Section 3.1; `uav/chromosome.py` |
| Fitness function | Section 3.4; `simulation/evaluator.py` |
| Crossover and mutation | Section 3.5; `genetic_algorithm/` |
| Constraint handling | Sections 3.2 and 3.5; `simulation/constraints.py`, `repair.py` |
| Convergence plot | `results/.../convergence.png`; Section 6 |
| Mission map | `results/.../mission_map.png` |
| Coverage heatmap | `results/.../coverage_heatmap.png` |
| Random-search comparison | Section 4; `comparison.csv`, `comparison.png` |
| Sensitivity analysis | Section 5; `sensitivity_results.csv`, `sensitivity.png` |
| Testing | Section 7; `tests/` |
| Reproducibility | Section 1; `run_config.json`, `best_solution.json` |

---

## 11. Caveats on the numbers

- Fitness values are only comparable within the same weights and environment.
- Coverage is priority-weighted; it is not the fraction of grid cells visited.
- The left panel of `coverage_heatmap.png` shows sampled route cells, not the
  sensor footprint. Target detection is shown separately on the right.
- Runtime varies slightly between runs on the same machine due to OS scheduling;
  fitness and route metrics are deterministic for a fixed seed and config.
