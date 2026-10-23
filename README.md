# UAV Disaster Surveillance Strategy Engine

A genetic-algorithm (GA) strategy engine for a fleet of three simulated UAVs
surveying a 50 x 50 disaster grid. The project is a complete, runnable
implementation of the AI assignment: custom chromosome encoding, a
multi-objective fitness function, spatial genetic operators, hard-constraint
repair, a random-search baseline, hyperparameter sensitivity experiments, and
all required visualisations.

> **Report:** the 3-page technical report is included at
> [`report/Tayyab_Mehmood_03-134242-092_Assignment2_Report.pdf`](report/Tayyab_Mehmood_03-134242-092_Assignment2_Report.pdf).
> Its HTML/CSS source and figures are in `report/`.

---

## 1. Project overview

The fleet must balance competing goals:

**Maximise**
- priority-weighted coverage of disaster targets;
- target detection rate.

**Minimise**
- battery / energy expenditure;
- mission duration;
- exposure to hazard intensity (wind / radiation pockets).

**Hard constraints**
- stay inside the grid;
- never enter obstacles or no-fly zones;
- respect each drone's maximum flight range and battery capacity;
- keep every chromosome structurally valid.

The GA evolves complete fleet strategies. Each candidate is one chromosome that
encodes every drone's waypoints, speed modes, and sensor power modes. A separate
evaluator flies the routes offline, computes all metrics, and reports constraint
violations independently of the scalar fitness.

---

## 2. Assignment objectives

| Assignment requirement | Implementation |
|---|---|
| Custom chromosome (waypoints, speed, sensor) | `uav/chromosome.py` |
| Multi-objective fitness | `simulation/evaluator.py` |
| Custom crossover and mutation | `genetic_algorithm/crossover.py`, `mutation.py` |
| Constraint handling (penalty + repair) | `simulation/repair.py`, evaluator violations |
| Object-oriented Python | classes for environment, drone, chromosome, evaluator, GA |
| Modular repository + requirements + README | this repository |
| Convergence plot | `visualization/convergence.py` |
| Route map and coverage heatmap | `visualization/mission_map.py`, `coverage.py` |
| Sensitivity analysis | `experiments/sensitivity_analysis.py` |
| GA vs random search | `experiments/random_search.py`, `--mode compare` |
| Automated tests | `tests/` (44 tests) |

---

## 3. Key AI concepts

- **Genetic algorithms**: population-based stochastic search.
- **Representation**: fixed-length, multi-segment, hybrid integer chromosome.
- **Multi-objective scalarisation**: weighted sum with hard-constraint penalties.
- **Tournament selection** and selection pressure.
- **Spatial crossover**: one-point suffix exchange on compatible drone routes.
- **Mutation**: local coordinate jitter, directed relocation rays, categorical
  speed/sensor changes.
- **Elitism** and optional fitness-stagnation termination.
- **Constraint repair**: snapping, local waypoint nudging, safe route
  regeneration.
- **Random-search baseline** under an equal evaluation budget.
- **One-factor-at-a-time and full-factorial** hyperparameter experiments.

---

## 4. System architecture

```
uav-strategy-engine/
├── config.py                     # dataclass configuration objects
├── main.py                       # CLI entry point
├── environment/
│   ├── grid.py                   # deterministic 50 x 50 map generation
│   └── targets.py                # weighted target model
├── uav/
│   ├── drone.py                  # drone parameters + speed/sensor mode tables
│   └── chromosome.py             # Waypoint, DroneRoute, Chromosome
├── simulation/
│   ├── constraints.py            # dense segment sampling, line-of-sight checks
│   ├── repair.py                 # waypoint and segment repair + regeneration
│   └── evaluator.py              # mission simulation and fitness
├── genetic_algorithm/
│   ├── population.py             # diverse valid initialisation
│   ├── selection.py              # tournament selection
│   ├── crossover.py              # route-suffix one-point crossover
│   ├── mutation.py               # spatial/categorical mutation
│   └── optimizer.py              # GA loop, elitism, caching, history
├── experiments/
│   ├── random_search.py          # baseline with equal budget
│   └── sensitivity_analysis.py   # population / mutation / tournament sweeps
├── visualization/
│   ├── mission_map.py
│   ├── convergence.py
│   ├── coverage.py
│   ├── comparison.py
│   └── sensitivity.py
├── tests/
├── results/                      # generated runs and evidence artifacts
└── docs/
```

Data flow:

```
config -> environment -> drones -> chromosome population
       -> evaluator (metrics + violations)
       -> GA operators + repair -> next generation
       -> best chromosome -> JSON / CSV / PNG artifacts
```

---

## 5. Coordinate convention

- Grid size: `50 x 50`, `x = column`, `y = row`.
- Valid cells: `0 <= x < 50`, `0 <= y < 50`.
- NumPy arrays are indexed `array[y, x]`.
- `(0, 0)` is the top-left cell in every plot.
- Waypoint coordinates are integers.

---

## 6. Chromosome representation

```
Chromosome
└── routes[d]                  # one DroneRoute per drone
    └── waypoints[w]           # fixed count (default 6)
        ├── x : int
        ├── y : int
        ├── speed : {0, 1, 2}
        └── sensor : {0, 1, 2}
```

A leg from the previous point to waypoint `w` uses waypoint `w`'s speed and
sensor settings. The drone's launch cell is fixed and is not part of the
chromosome. Speed/sensor categories:

| speed | meaning | cells / time unit |
|---|---|---|
| 0 | slow | 0.70 |
| 1 | cruise | 1.00 |
| 2 | fast | 1.50 |

| sensor | meaning | detection radius | energy / time |
|---|---|---|---|
| 0 | off | 0.0 | 0.00 |
| 1 | medium | 1.6 cells | 0.35 |
| 2 | high | 3.2 cells | 0.85 |

**Why fixed route length?** It keeps crossover and mutation compatible (all
chromosomes have the same shape), keeps the search space bounded, and makes
elitism and statistics straightforward. Variable-length routes are a possible
extension, not a requirement here.

**Why these genes?** They directly represent the assignment's three decision
dimensions: deployment/trajectory (x, y), speed adjustment, and sensor power.
The design is easy to repair: coordinates can be nudged or regenerated and
categorical values always have a valid replacement.

Waypoints are immutable frozen dataclasses. `Chromosome.copy()` creates new
route lists, so parents and offspring never share mutable structures.

---

## 7. Environment model

- **Obstacles**: circular blobs painted into a boolean mask.
- **No-fly zones**: separate boolean mask, also a hard constraint.
- **Hazard map**: sum of Gaussian hazard pockets clipped to `[0, 1]`.
- **Targets**: 14 weighted disaster targets, priority in `[0.5, 1.0]`.
- **Drone starts**: three spread launch cells.
- All placement is deterministic given the seed. Targets and starts are sampled
  from the largest 4-connected free region. Generation retries if a seed
  produces an unusable layout and raises `EnvironmentGenerationError` rather
  than returning a broken map.

---

## 8. Mission evaluation and fitness

For each drone the evaluator starts at its launch cell, samples every straight
leg densely (0.5-cell resolution), and checks the *entire trajectory*, not just
waypoints.

### Metrics

1. **Priority-weighted coverage**

   `coverage = sum(priority of detected targets) / sum(priority of all targets)`

   Detection is a union across the fleet: repeated visits or multiple drones
   detecting the same target count once.

2. **Detection rate**

   `detection_rate = number of detected targets / total targets`

   Detection is deterministic (no probabilistic sensing), so candidates are
   compared fairly.

3. **Energy**

   ```
   movement = distance * 1.0 * (speed / cruise_speed)^2
   sensor   = time * sensor_energy_rate
   energy   = movement + sensor
   normalized_energy = total_energy / total_fleet_battery_capacity
   ```

4. **Flight time**

   `duration_d = sum(leg_distance / leg_speed)` per drone.
   Drones fly in parallel, so `mission_time = max(duration_d)`.
   `normalized_time = mission_time / time_reference` (default 120 time units).

5. **Hazard exposure**

   For each leg, `mean(hazard intensity of sampled cells) * leg_distance`.
   `normalized_risk = hazard_exposure / total_distance` (mean hazard along the
   fleet's paths, in `[0, 1]`).

### Fitness

```
raw = 100 * coverage
    +  50 * detection_rate
    -  20 * normalized_energy
    -  15 * normalized_time
    -  30 * normalized_risk
```

If any hard constraint is violated:

```
penalty = 1000 + 50 * violation_count + 25 * sum(min(magnitude, 10))
fitness = raw - penalty
```

Feasibility is also stored separately (`feasible: bool`), and violations are
returned as structured records (`kind`, `drone_id`, `detail`, `magnitude`).
Tournament selection and elitism compare `(feasible, fitness)`, so an infeasible
candidate can never outrank a feasible one.

These weights are **configurable starting weights**, not assignment-mandated
values. Different weights change the preferred trade-off.

---

## 9. Genetic algorithm

- **Initialisation**: random ray-based routes; 60% of rays are target-directed
  with jitter, the rest random. This yields diverse routes with clear legs.
- **Selection**: tournament selection, configurable size `k`. Larger `k` raises
  pressure; the population-wide best wins any tournament with probability
  `k / N`.
- **Crossover**: one-point suffix exchange for the same drone in two parents.
  Offspring route lists are independent; waypoints are immutable. The cut may
  create a blocked leg, which repair fixes.
- **Mutation**: per-waypoint probability. 60% move (local jitter or a directed
  ray), 20% speed change, 20% sensor change. Modes always change to a different
  valid mode; parents are never modified.
- **Repair**: snap out-of-bounds/forbidden waypoints to the nearest free cell,
  nudge waypoints until adjacent legs are clear, or regenerate the route. The
  evaluator re-checks feasibility independently.
- **Elitism**: top feasible candidates are copied unchanged. If a generation has
  no feasible candidate, elitism is skipped explicitly and logged.
- **Termination**: generation limit, optional stagnation limit, or
  `no_feasible_solution` when no feasible candidate exists at the end.
- **Caching**: evaluations are cached by the full chromosome signature; mutation
  and crossover operate on copies, so stale results cannot be reused.
- **History**: each generation records best/mean feasible fitness, best
  coverage/detection/energy/time/risk, feasible count, and best penalised
  fitness. Missing feasible values are stored as empty CSV fields / `null` JSON.

---

## 10. Constraint handling

Hard constraints are checked along every sampled leg:

- out-of-bounds cells;
- no-fly-zone cells;
- obstacle cells;
- per-drone maximum range;
- per-drone battery capacity;
- structural validity (route count, waypoint count, coordinate types, modes).

Repair is a helper, not a substitute for validation. After repair, the evaluator
still scans the trajectory. No infeasible chromosome is reported as feasible.

---

## 11. Installation

Requires Python 3.10+ (tested with Python 3.12.3 on Ubuntu 24.04).

```bash
cd uav-strategy-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt`:

```
numpy>=1.24
matplotlib>=3.6
pytest>=7.0
```

All figures use the non-interactive Agg backend and are saved to disk.

---

## 12. Running the simulation

Default GA demonstration (population 60, 50 generations, seed 42):

```bash
python main.py
```

Equivalent explicit command:

```bash
python main.py --mode ga --seed 42 --population 60 --generations 50 \
  --mutation-rate 0.05 --tournament-size 3 --waypoints 6 --drones 3
```

Sensitivity-guided tuned run (population 100, seed 42):

```bash
python main.py --mode ga --population 100 --generations 50 --quiet
```

Other modes:

```bash
python main.py --mode random       # equal-budget random search
python main.py --mode compare --seeds 3
python main.py --mode sensitivity
```

Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--seed` | 42 | environment + GA seed |
| `--population` | 60 | GA population size |
| `--generations` | 50 | GA generation limit |
| `--mutation-rate` | 0.05 | per-waypoint mutation probability |
| `--tournament-size` | 3 | selection pressure |
| `--crossover-rate` | 0.9 | probability of crossover per pair |
| `--elitism` | 2 | feasible elites copied per generation |
| `--stagnation-limit` | none | optional early stop |
| `--waypoints` | 6 | fixed waypoints per drone |
| `--drones` | 3 | fleet size |
| `--targets` | 14 | number of disaster targets |
| `--width`, `--height` | 50, 50 | grid size |
| `--output-dir` | `results` | run output root |
| `--seeds` | 3 | number of seeds in compare mode |
| `--sensitivity-generations` | 30 | generations per sensitivity run |
| `--sensitivity-seeds` | 2 | seeds per sensitivity setting |
| `--full-factorial` | off | run all sensitivity combinations |
| `--no-cache` | off | disable evaluation caching |
| `--quiet` | off | suppress progress output |

Each run creates a new timestamped folder such as
`results/run_20261023-234420_ga_seed42/`; previous runs are never overwritten.

---

## 13. Tests

```bash
python3 -m pytest -q
```

Result: **44 passed** on Python 3.12.3. Test files cover:

- environment determinism, dimensions, target validity, impossible maps;
- chromosome structure, copying, serialisation, invalid values;
- no-fly waypoint and segment handling, repair, bounds, range/battery;
- coverage, duplicate detection, energy/time/risk, deterministic evaluation;
- crossover structure and independence, mutation probabilities and parent
  immutability, tournament feasibility preference;
- GA history length, elitism monotonicity, reproducibility, no-feasible runs,
  single-generation runs.

---

## 14. Output artifacts

A GA run writes:

| File | Content |
|---|---|
| `run_config.json` | CLI args, full configuration, simulation mode tables |
| `best_solution.json` | best chromosome, all metrics, violations, per-drone detail, metadata |
| `fitness_history.csv` | per-generation best/mean feasible fitness and mission metrics |
| `mission_map.png` | environment, targets, starts, evolved routes |
| `mission_map_initial.png` | generation-0 best routes (evolution comparison) |
| `convergence.png` | best vs mean feasible fitness |
| `coverage_heatmap.png` | route visitation density + detected/missed targets |

A random-search run writes `best_solution.json`, `random_search_history.csv`,
`mission_map.png`, `coverage_heatmap.png`, and `random_search_progress.png`.

Compare mode writes `comparison.csv`, `comparison_summary.csv`,
`comparison.png`, per-seed GA/random subfolders, and `best_ga_solution.json`.

Sensitivity mode writes `sensitivity_results.csv` (one row per run) and
`sensitivity_summary.csv` (mean/std per setting), plus `sensitivity.png` for
one-factor runs.

`best_solution.json` / `run_config.json` include: seed, algorithm parameters,
fitness weights, environment configuration and summary, termination reason,
evaluation budget/counts, feasibility, and the complete chromosome.

---

## 15. Reproducibility

- All random choices use `numpy.random.default_rng(seed)`.
- The environment is generated from the run seed.
- GA runs with the same seed and configuration produce the same best chromosome
  signature and fitness (tested).
- Compare mode uses seeds `seed, seed+1, ...` and the same environment per seed
  for GA and random search.
- Sensitivity runs share one environment instance and vary only the configured
  parameter plus the seed list.
- CSV/JSON outputs are written with full-precision values rounded to 6 decimal
  places for CSV readability.

---

## 16. Simulation assumptions and limitations

- All units are **normalised simulation units**, not real UAV specifications.
- Flight is an offline straight-leg model; there is no real-time control,
  collision avoidance, wind dynamics, or probabilistic sensing.
- Obstacles are treated as no-fly for the UAV (blocked terrain is assumed to
  also block low-altitude flight).
- Drones execute in parallel; mission time is the slowest drone's duration.
- Targets are detected deterministically when a sampled route point lies within
  the active sensor radius.
- Route generation is line-of-sight based; a route whose segment is blocked is
  repaired or regenerated, not flown around with a separate path planner.
- Fixed route length (default 6) is a deliberate simplification.
- The GA is a stochastic heuristic and does not guarantee a globally optimal
  solution. Sensitivity results with two seeds are indicative, not definitive.
- Random search uses the same generator and repair rules, so it is a strong
  baseline; the GA is not guaranteed to win on every seed.

---

## 17. Measured results

Actual outputs from this machine (seed 42, 50 x 50 map):

| Run | Fitness | Coverage | Detection | Energy | Time | Hazard |
|---|---:|---:|---:|---:|---:|---:|
| GA 60 x 50 | 122.145 | 94.67% | 92.86% | 173.08 | 50.18 | 37.86 |
| GA 100 x 50 (tuned) | **128.617** | **100.00%** | **100.00%** | 216.04 | 49.88 | 46.71 |
| Random, 3000 evals | 105.643 | 88.10% | 85.71% | 311.03 | 53.97 | 55.41 |

Three-seed comparison at 3000 evaluations per method: GA mean fitness
119.096 +/- 6.888 and coverage 92.80%, versus random-search mean fitness
89.596 +/- 15.884 and coverage 77.22%. Full tables, per-seed values, sensitivity
outputs, and report-ready notes are in
`docs/RESULTS_AND_METHODOLOGY.md` and `results/`.
