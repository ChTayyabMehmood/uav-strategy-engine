# UAV Disaster Surveillance Strategy Engine

A genetic-algorithm (GA) strategy engine for a fleet of three simulated UAVs
surveying a 50 x 50 disaster grid. The project implements the complete AI
assignment: evolved **deployment positions**, waypoint trajectories, speed and
sensor schedules, a multi-objective fitness function, custom spatial genetic
operators, constraint repair, a random-search baseline, sensitivity experiments,
and all required visualisations.

**Report:** the 4-page technical report is at
[`report/Tayyab_Mehmood_03-134242-092_Assignment2_Report.pdf`](report/Tayyab_Mehmood_03-134242-092_Assignment2_Report.pdf).
Its HTML/CSS source and figures are in `report/`.

**Repository:** <https://github.com/ChTayyabMehmood/uav-strategy-engine>

---

## 1. Assignment compliance

| Requirement | Where it is implemented |
| --- | --- |
| Optimal deployment positions | `Chromosome.deployments`, `genetic_algorithm/population.py`, mutation and crossover |
| Sensor activation schedules | Sensor mode on every waypoint/leg in `uav/chromosome.py`; evaluated in `simulation/evaluator.py` |
| Waypoint trajectories | `DroneRoute` and `Waypoint` genes |
| Hybrid chromosome encoding | `uav/chromosome.py` (integer coordinates + discrete speed/sensor genes) |
| Multi-objective fitness | `simulation/evaluator.py` (coverage, detection, energy, time, hazard) |
| Custom crossover and mutation | `genetic_algorithm/crossover.py`, `genetic_algorithm/mutation.py` |
| Avoid premature convergence | Elitism, spatial operators, random immigrants on stagnation |
| Hard constraints (range, no-fly) | `simulation/constraints.py`, `simulation/repair.py`, evaluator penalties |
| Object-oriented implementation | `GridEnvironment`, `Drone`, `Chromosome`, `MissionEvaluator`, `GeneticAlgorithm` |
| Population init, selection, termination | `genetic_algorithm/population.py`, `selection.py`, `optimizer.py` |
| GitHub repository + requirements + README | this repository |
| Convergence plot (best vs average) | `visualization/convergence.py` |
| Evolved path maps | `visualization/mission_map.py` |
| Coverage heatmaps across generations | `visualization/coverage.py` (`plot_coverage_evolution`) |
| Sensitivity analysis | `experiments/sensitivity_analysis.py` |
| Comparison with random search | `experiments/random_search.py`, `--mode compare` |
| 3-5 page technical report | `report/` (4 pages) |
| Automated tests | `tests/` (47 tests) |

---

## 2. Project structure

```
uav-strategy-engine/
├── config.py                     # environment, fleet, fitness and GA settings
├── main.py                       # CLI entry point
├── requirements.txt
├── environment/
│   ├── grid.py                   # deterministic hazard grid + priority map
│   └── targets.py                # targets and priority-zone construction
├── uav/
│   ├── drone.py                  # drone limits and speed/sensor mode tables
│   └── chromosome.py             # deployment genes + Waypoint/DroneRoute/Chromosome
├── simulation/
│   ├── constraints.py            # dense segment sampling and line checks
│   ├── repair.py                 # waypoint and segment repair
│   └── evaluator.py              # mission simulation, coverage mask and fitness
├── genetic_algorithm/
│   ├── population.py             # deployment sampling + valid initialisation
│   ├── selection.py              # tournament selection
│   ├── crossover.py              # deployment/route block-suffix crossover
│   ├── mutation.py               # spatial, speed, sensor and deployment mutation
│   └── optimizer.py              # GA loop, elitism, immigrants, caching, history
├── experiments/
│   ├── random_search.py          # equal-budget baseline
│   └── sensitivity_analysis.py   # population / mutation / tournament sweeps
├── visualization/
│   ├── mission_map.py            # evolved paths and deployment cells
│   ├── convergence.py            # best vs average fitness
│   ├── coverage.py               # coverage maps + coverage evolution
│   ├── comparison.py             # GA vs random search
│   └── sensitivity.py            # sensitivity plots
├── tests/                        # 47 pytest tests
├── results/                      # final run artifacts (GA, random, compare, sensitivity)
└── report/                       # report source, figures and PDF
```

Data flow:

```
config -> environment -> priority map -> initial population
       -> evaluator (coverage mask, metrics, violations)
       -> selection / crossover / mutation / repair
       -> next generation -> best chromosome -> JSON / CSV / PNG
```

---

## 3. Installation

Requires Python 3.10+ (tested on Python 3.12, Ubuntu 24.04).

```bash
cd uav-strategy-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` contains `numpy`, `matplotlib` and `pytest`. All figures use
the non-interactive Agg backend and are written to disk.

---

## 4. Running

Default GA run (population 60, 50 generations, mutation 0.20, seed 42):

```bash
python main.py
```

Explicit equivalent:

```bash
python main.py --mode ga --seed 42 --population 60 --generations 50 \
  --mutation-rate 0.2 --tournament-size 3 --waypoints 6 --drones 3
```

Other modes:

```bash
python main.py --mode random                  # equal-budget random search
python main.py --mode compare --seeds 3       # GA vs random, seeds 42-44
python main.py --mode sensitivity             # population/mutation/tournament
python main.py --mode sensitivity --full-factorial
```

Tuned GA run used in the report:

```bash
python main.py --mode ga --population 200 --generations 50 \
  --tournament-size 5 --output-dir results --quiet
```

Useful options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--seed` | 42 | environment + GA seed |
| `--population` | 60 | GA population size |
| `--generations` | 50 | generation limit |
| `--mutation-rate` | 0.20 | per-waypoint mutation probability |
| `--tournament-size` | 3 | selection pressure |
| `--crossover-rate` | 0.9 | crossover probability per pair |
| `--elitism` | 2 | feasible elites copied per generation |
| `--stagnation-limit` | none | optional early stop |
| `--waypoints` | 6 | waypoints per drone |
| `--drones` | 3 | fleet size |
| `--targets` | 14 | number of targets |
| `--width`, `--height` | 50, 50 | grid size |
| `--output-dir` | `results` | output root |
| `--seeds` | 3 | seeds in compare mode |
| `--sensitivity-generations` | 30 | generations per sensitivity run |
| `--sensitivity-seeds` | 2 | seeds per sensitivity setting |
| `--full-factorial` | off | run all sensitivity combinations |
| `--no-cache` | off | disable evaluation caching |
| `--quiet` | off | suppress progress output |

Each run writes to a timestamped folder under `--output-dir`. The final report
artifacts are committed under `results/ga_seed42`, `results/random_seed42`,
`results/compare_seed42`, `results/tuned_seed42` and
`results/sensitivity_seed42`.

---

## 5. Chromosome representation

```
Chromosome
├── deployments[3]             # evolved (x, y) start cell per drone
└── routes[3]                  # one DroneRoute per drone
    └── waypoints[6]           # fixed count
        ├── x : int
        ├── y : int
        ├── speed : {0, 1, 2}
        └── sensor : {0, 1, 2}
```

A chromosome stores 3 x 2 deployment values plus 3 x 6 x 4 waypoint values =
**78 genes**. A leg from the previous point to waypoint `w` uses waypoint `w`'s
speed and sensor settings. The deployment cell is the route's first point.

| speed | meaning | cells / time unit |
| --- | --- | --- |
| 0 | slow | 0.70 |
| 1 | cruise | 1.00 |
| 2 | fast | 1.50 |

| sensor | meaning | detection radius | energy / time |
| --- | --- | --- | --- |
| 0 | off | 0.0 | 0.00 |
| 1 | medium | 1.6 cells | 0.35 |
| 2 | high | 3.2 cells | 0.85 |

Speed and sensor modes are discrete, which makes mutation and repair simple.
The deployment cell and its route stay in one drone block, so crossover never
separates a base from the route evolved with it. Waypoints are immutable frozen
dataclasses; `Chromosome.copy()` creates fully independent route lists.

---

## 6. Environment and priority zones

- 50 x 50 grid, deterministic from the run seed.
- Circular obstacle blobs and no-fly zones are hard constraints.
- Hazard map is a sum of Gaussian hazard pockets clipped to `[0, 1]`.
- 14 weighted targets, priority in `[0.5, 1.0]`.
- Every target owns a circular priority zone (`target_zone_radius = 3.0`).
  The zone is normalised to unit mass and multiplied by the target priority, so
  edge truncation does not change a target's importance.
- `GridEnvironment.priority_map` is the normalised high-priority zone map used
  by the area-coverage metric.

The sensor disk is unioned along every sampled leg. Coverage is the
share of total priority-zone mass that the fleet's sensors actually surveyed,
not merely the count of target points visited.

---

## 7. Fitness function

For each drone the evaluator starts at its evolved deployment cell, samples every
straight leg at 0.5-cell resolution, and checks the entire trajectory.

### Metrics

1. **Priority-zone coverage**

   `coverage = surveyed priority-zone mass / total priority-zone mass`

   Repeated coverage counts once; the mask is a union across all drones.

2. **Detection rate**

   `detection_rate = detected targets / total targets`

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
   `normalized_time = mission_time / time_reference` (default 120).

5. **Hazard exposure**

   `mean(hazard intensity of sampled cells) * leg_distance`.
   `normalized_risk = hazard_exposure / total_distance`.

### Score

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

Feasibility is stored separately (`feasible: bool`) and violations are returned
as structured records (`kind`, `drone_id`, `detail`, `magnitude`). Tournament
selection and elitism compare `(feasible, fitness)`, so an infeasible plan can
never outrank a feasible one. The weights are configurable design choices, not
assignment-mandated values.

---

## 8. Genetic algorithm

- **Initialisation:** deployment cells are sampled with minimum spacing; routes
  are random but target-directed with jitter, then repaired to validity.
- **Selection:** tournament selection with configurable size `k`. Larger `k`
  raises pressure.
- **Crossover:** block-suffix crossover swaps complete drone blocks, including
  each block's deployment cell and route. This is spatial segment swapping and
  keeps deployment/route pairs intact.
- **Mutation:** per-waypoint probability 0.20; 60% move, 20% speed, 20% sensor.
  Each deployment cell is relocated with probability 0.05.
- **Diversity control:** if the best feasible score does not improve for four
  generations, 10% random immigrants replace the worst plans. This directly
  addresses premature convergence.
- **Repair:** snap out-of-bounds or forbidden waypoints, nudge blocked legs, or
  regenerate the route. The evaluator re-checks every leg independently.
- **Elitism:** top feasible plans are copied unchanged. If no feasible plan
  exists, elitism is skipped and logged.
- **Termination:** generation limit, optional stagnation limit, or
  `no_feasible_solution`.
- **Caching:** evaluations are cached by the full chromosome signature
  (deployments + routes), so stale results cannot be reused.

---

## 9. Constraint handling

Hard constraints are checked along every sampled leg:

- out-of-bounds cells;
- no-fly zones;
- obstacles;
- per-drone maximum range;
- per-drone battery capacity;
- structural validity (route count, waypoint count, integer coordinates,
  valid speed/sensor modes, deployment count).

Repair is a helper, not a substitute for validation. No infeasible chromosome is
reported as feasible.

---

## 10. Visualisations and outputs

A GA run writes:

| File | Content |
| --- | --- |
| `run_config.json` | CLI args, full configuration, simulation mode tables |
| `best_solution.json` | deployments, routes, metrics, violations, per-drone detail |
| `fitness_history.csv` | per-generation best/mean fitness and mission metrics |
| `mission_map.png` | environment, targets, evolved deployment cells and routes |
| `mission_map_initial.png` | generation-0 best plan |
| `convergence.png` | best vs mean feasible fitness |
| `coverage_heatmap.png` | priority-zone coverage + visitation density |
| `coverage_evolution.png` | **coverage heatmaps at generations 0, 16, 33 and 49** |

A random-search run writes `best_solution.json`,
`random_search_history.csv`, `mission_map.png`, `coverage_heatmap.png` and
`random_search_progress.png`.

Compare mode writes `comparison.csv`, `comparison_summary.csv`,
`comparison.png`, per-seed GA/random subfolders, and `best_ga_solution.json`.
Sensitivity mode writes `sensitivity_results.csv`,
`sensitivity_summary.csv` and `sensitivity.png`.

`best_solution.json` and `run_config.json` include the seed, algorithm
parameters, fitness weights, environment summary, termination reason, evaluation
counts, feasibility and the complete chromosome (including deployment genes).

---

## 11. Tests

```bash
python3 -m pytest -q
```

Result: **47 passed** on Python 3.12. Tests cover:

- environment determinism, dimensions, target validity, priority zones;
- chromosome structure, deployments, copying, serialisation, invalid values;
- no-fly and obstacle handling, repair, bounds, range and battery;
- area coverage, duplicate coverage, energy/time/risk, deterministic scoring;
- crossover structure and independence, mutation probabilities and parent
  immutability, tournament feasibility preference;
- GA history length, elitism monotonicity, reproducibility, no-feasible runs.

---

## 12. Measured results

Equal budget: 3,000 evaluations per method, three seeds (42, 43, 44), 50 x 50
map. Values are means +/- standard deviation.

| Method | Fitness | Coverage | Targets | Energy | Time | Risk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Genetic algorithm | **102.9 +/- 6.6** | **80.6%** | **13.3 / 14** | 372.9 | **72.4** | **0.225** |
| Random search | 78.4 +/- 5.9 | 65.0% | 11.3 / 14 | **348.9** | 79.5 | 0.272 |

The GA won on every seed (101.3, 110.1, 97.2 versus 85.2, 73.9, 76.3). Random
search used less energy because it often took short direct routes; the GA spent
energy to survey more priority zones. The tuned run (population 200, tournament
5, 10,000 evaluations) reached **109.4**, **84.4% coverage** and detected all 14
targets.

Sensitivity (two seeds, 30 generations):

| Parameter | Values | Mean best fitness |
| --- | --- | --- |
| Population | 50 / 100 / 200 | 101.5 / 104.5 / 105.1 |
| Mutation | 0.10 / 0.20 / 0.30 | 103.4 / 103.0 / 102.4 |
| Tournament | 2 / 3 / 5 | 101.6 / 103.0 / 106.9 |

Full CSV/JSON/PNG evidence is under `results/`.

---

## 13. Reproducibility and limitations

- All random choices use `numpy.random.default_rng(seed)`.
- The environment is generated from the run seed.
- GA runs with the same seed and configuration produce the same best chromosome
  signature and fitness (tested).
- Compare mode uses seeds `seed, seed+1, ...` and the same environment per seed
  for GA and random search.
- Sensitivity runs share one environment and vary only the configured parameter
  and seed list.
- All units are normalised simulation units, not real UAV specifications.
- Flight is an offline straight-leg model; there is no real-time control,
  collision avoidance, wind dynamics or probabilistic sensing.
- Obstacles are treated as no-fly for the UAV.
- Drones execute in parallel; mission time is the slowest drone's duration.
- Fixed route length (6 waypoints) is a deliberate simplification.
- The GA is a stochastic heuristic and does not guarantee a global optimum.
