"""Command-line entry point for the UAV disaster-surveillance strategy engine.

Examples
--------
python main.py                         # default GA demonstration
python main.py --mode ga --generations 80 --population 80
python main.py --mode random
python main.py --mode compare --seeds 3
python main.py --mode sensitivity --sensitivity-generations 30
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from config import (
    EnvironmentConfig,
    FitnessWeights,
    FleetConfig,
    GAConfig,
    SimulationConfig,
)
from environment import EnvironmentGenerationError, generate_environment
from experiments.random_search import run_random_search
from experiments.sensitivity_analysis import run_sensitivity
from genetic_algorithm import GeneticAlgorithm, OptimizationResult
from uav import build_fleet, modes_table
from visualization import (
    plot_algorithm_comparison,
    plot_best_so_far,
    plot_convergence,
    plot_coverage_evolution,
    plot_coverage_overview,
    plot_mission_map,
    plot_sensitivity_summary,
)

LOGGER = logging.getLogger("uav_strategy_engine")


# ----------------------------------------------------------------------
# CLI and configuration
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Genetic-algorithm strategy engine for a fleet of disaster-"
            "surveillance UAVs on a 2-D hazard grid."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["ga", "random", "compare", "sensitivity"],
        default="ga",
        help="which experiment to run (default: ga)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population", type=int, default=60)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--crossover-rate", type=float, default=0.9)
    parser.add_argument("--elitism", type=int, default=2)
    parser.add_argument(
        "--stagnation-limit",
        type=int,
        default=None,
        help="optional early stop after this many generations without improvement",
    )
    parser.add_argument("--waypoints", type=int, default=6)
    parser.add_argument("--drones", type=int, default=3)
    parser.add_argument("--targets", type=int, default=14)
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--height", type=int, default=50)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="number of seeds for --mode compare",
    )
    parser.add_argument(
        "--sensitivity-generations",
        type=int,
        default=30,
        help="generations per sensitivity run",
    )
    parser.add_argument(
        "--sensitivity-seeds",
        type=int,
        default=2,
        help="seeds per sensitivity setting",
    )
    parser.add_argument(
        "--full-factorial",
        action="store_true",
        help="sensitivity mode: run every combination of parameter values",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="disable GA evaluation caching",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def make_config(args: argparse.Namespace, seed: int | None = None) -> SimulationConfig:
    run_seed = args.seed if seed is None else int(seed)
    environment = EnvironmentConfig(
        seed=run_seed,
        width=args.width,
        height=args.height,
        n_targets=args.targets,
    )
    fleet = FleetConfig(
        n_drones=args.drones,
        waypoints_per_route=args.waypoints,
    )
    ga = GAConfig(
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        tournament_size=args.tournament_size,
        crossover_rate=args.crossover_rate,
        elitism=args.elitism,
        stagnation_limit=args.stagnation_limit,
        seed=run_seed,
        cache_evaluations=not args.no_cache,
    )
    return SimulationConfig(
        environment=environment,
        fleet=fleet,
        weights=FitnessWeights(),
        ga=ga,
    )


def create_scenario(config: SimulationConfig):
    env = generate_environment(config.environment, config.fleet.n_drones)
    drones = build_fleet(env.starts, config.fleet)
    return env, drones


def create_run_dir(base: str | Path, mode: str, seed: int) -> Path:
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = base / f"run_{stamp}_{mode}_seed{seed}"
    counter = 1
    while candidate.exists():
        candidate = base / f"run_{stamp}_{mode}_seed{seed}_{counter}"
        counter += 1
    candidate.mkdir(parents=True)
    return candidate


def run_ga(
    config: SimulationConfig,
    env,
    drones,
    progress=None,
) -> OptimizationResult:
    return GeneticAlgorithm(
        env,
        drones,
        config.ga,
        config.weights,
        n_waypoints=config.fleet.waypoints_per_route,
        time_reference=config.time_reference,
        progress=progress,
    ).run()


# ----------------------------------------------------------------------
# Serialisation helpers
# ----------------------------------------------------------------------
def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def write_run_config(
    run_dir: Path,
    config: SimulationConfig,
    args: argparse.Namespace,
    mode: str,
) -> None:
    write_json(
        run_dir / "run_config.json",
        {
            "mode": mode,
            "command_line": vars(args),
            "config": config.to_dict(),
            "simulation_modes": modes_table(),
        },
    )


def write_fitness_history(run_dir: Path, result: OptimizationResult) -> None:
    fields = [
        "generation",
        "feasible_count",
        "best_feasible_fitness",
        "average_feasible_fitness",
        "best_coverage",
        "best_detection_rate",
        "best_total_energy",
        "best_normalized_energy",
        "best_mission_time",
        "best_hazard_exposure",
        "best_normalized_risk",
        "best_penalized_fitness",
    ]
    with (run_dir / "fitness_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for stats in result.history:
            writer.writerow(
                {
                    key: (
                        round(value, 6)
                        if isinstance(value, float)
                        else value
                    )
                    for key, value in stats.to_dict().items()
                }
            )


def _ga_payload(
    config: SimulationConfig,
    env,
    result: OptimizationResult,
) -> dict:
    evaluation = result.best_evaluation
    return {
        "method": "genetic_algorithm",
        "seed": config.ga.seed,
        "termination_reason": result.termination_reason,
        "generations_run": result.generations_run,
        "evaluation_budget": config.ga.population_size * config.ga.generations,
        "evaluation_requests": result.evaluation_requests,
        "unique_evaluations": result.unique_evaluations,
        "cache_hits": result.cache_hits,
        "feasible_evaluations": result.feasible_evaluations,
        "feasibility_rate": result.feasibility_rate,
        "runtime_seconds": result.runtime_seconds,
        "feasible": evaluation.feasible,
        "fitness": evaluation.fitness,
        "metrics": evaluation.to_dict(),
        "chromosome": result.best_chromosome.to_dict(),
        "config": config.to_dict(),
        "environment_summary": env.to_dict(),
        "simulation_modes": modes_table(),
    }


def _random_payload(
    config: SimulationConfig,
    env,
    result,
) -> dict:
    evaluation = result.best_evaluation
    return {
        "method": "random_search",
        "seed": result.seed,
        "termination_reason": "evaluation_budget",
        "evaluation_budget": result.evaluations,
        "evaluation_requests": result.evaluations,
        "feasible_evaluations": result.feasible_count,
        "feasibility_rate": result.feasibility_rate,
        "runtime_seconds": result.runtime_seconds,
        "feasible": evaluation.feasible,
        "fitness": evaluation.fitness,
        "metrics": evaluation.to_dict(),
        "chromosome": result.best_chromosome.to_dict(),
        "config": config.to_dict(),
        "environment_summary": env.to_dict(),
        "simulation_modes": modes_table(),
    }


def write_random_history(run_dir: Path, result) -> None:
    with (run_dir / "random_search_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["evaluation", "best_feasible_fitness"])
        for evaluation, fitness in result.history:
            writer.writerow(
                [
                    evaluation,
                    "" if fitness is None else round(float(fitness), 6),
                ]
            )


def save_ga_artifacts(
    run_dir: Path,
    config: SimulationConfig,
    env,
    drones,
    result: OptimizationResult,
    title_prefix: str,
    include_initial: bool = True,
) -> None:
    write_json(run_dir / "best_solution.json", _ga_payload(config, env, result))
    write_fitness_history(run_dir, result)
    plot_mission_map(
        env,
        result.best_chromosome,
        result.best_evaluation,
        drones,
        run_dir / "mission_map.png",
        title=f"{title_prefix} - evolved routes",
    )
    plot_convergence(
        result.history,
        run_dir / "convergence.png",
        title=f"{title_prefix} - fitness convergence",
    )
    plot_coverage_overview(
        env,
        result.best_chromosome,
        result.best_evaluation,
        drones,
        run_dir / "coverage_heatmap.png",
        title=f"{title_prefix} - coverage",
    )
    if include_initial:
        plot_mission_map(
            env,
            result.initial_best_chromosome,
            result.initial_best_evaluation,
            drones,
            run_dir / "mission_map_initial.png",
            title=f"{title_prefix} - generation 0 best",
        )
        if len(result.generation_best_chromosomes) > 1:
            plot_coverage_evolution(
                env,
                result.generation_best_chromosomes,
                result.generation_best_evaluations,
                run_dir / "coverage_evolution.png",
                title=f"{title_prefix} - coverage evolution",
            )


def print_ga_summary(result: OptimizationResult, run_dir: Path) -> None:
    evaluation = result.best_evaluation
    print(
        f"  best fitness : {evaluation.fitness:.2f} "
        f"(feasible={evaluation.feasible})"
    )
    print(
        f"  coverage     : {evaluation.coverage:.1%}  "
        f"detection: {evaluation.detection_rate:.1%}"
    )
    print(
        f"  energy       : {evaluation.total_energy:.1f} "
        f"({evaluation.normalized_energy:.3f} normalised)"
    )
    print(
        f"  mission time : {evaluation.mission_time:.1f} "
        f"({evaluation.normalized_time:.3f} normalised)"
    )
    print(
        f"  hazard risk  : {evaluation.hazard_exposure:.2f} "
        f"(mean {evaluation.normalized_risk:.3f})"
    )
    print(
        f"  generations  : {result.generations_run} | "
        f"reason: {result.termination_reason} | "
        f"evaluations: {result.evaluation_requests} "
        f"(unique {result.unique_evaluations})"
    )
    print(f"  runtime      : {result.runtime_seconds:.2f} s")
    print(f"  output       : {run_dir}")


def progress_printer(prefix: str):
    def _progress(generation: int, stats) -> None:
        best = (
            "n/a"
            if stats.best_feasible_fitness is None
            else f"{stats.best_feasible_fitness:.2f}"
        )
        average = (
            "n/a"
            if stats.average_feasible_fitness is None
            else f"{stats.average_feasible_fitness:.2f}"
        )
        print(
            f"[{prefix}] gen {generation:03d} | best {best} | "
            f"avg {average} | feasible {stats.feasible_count}"
        )

    return _progress


# ----------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------
def run_ga_mode(args: argparse.Namespace) -> Path:
    config = make_config(args)
    env, drones = create_scenario(config)
    run_dir = create_run_dir(args.output_dir, "ga", config.ga.seed)
    write_run_config(run_dir, config, args, "ga")

    print(
        f"GA | seed={config.ga.seed} population={config.ga.population_size} "
        f"generations={config.ga.generations} mutation={config.ga.mutation_rate} "
        f"tournament={config.ga.tournament_size}"
    )
    result = run_ga(
        config,
        env,
        drones,
        progress=None if args.quiet else progress_printer("ga"),
    )
    save_ga_artifacts(run_dir, config, env, drones, result, "GA run")
    print_ga_summary(result, run_dir)
    return run_dir


def run_random_mode(args: argparse.Namespace) -> Path:
    config = make_config(args)
    env, drones = create_scenario(config)
    budget = config.ga.population_size * config.ga.generations
    run_dir = create_run_dir(args.output_dir, "random", config.ga.seed)
    write_run_config(run_dir, config, args, "random")

    print(f"Random search | budget={budget} seed={config.ga.seed}")
    result = run_random_search(
        env,
        drones,
        config.weights,
        n_waypoints=config.fleet.waypoints_per_route,
        budget=budget,
        seed=config.ga.seed,
        time_reference=config.time_reference,
        progress=(
            None
            if args.quiet
            else lambda index, best: print(
                f"[random] {index}/{budget} | best feasible {best}"
            )
        ),
    )
    write_json(run_dir / "best_solution.json", _random_payload(config, env, result))
    write_random_history(run_dir, result)
    plot_mission_map(
        env,
        result.best_chromosome,
        result.best_evaluation,
        drones,
        run_dir / "mission_map.png",
        title="Random-search best candidate",
    )
    plot_coverage_overview(
        env,
        result.best_chromosome,
        result.best_evaluation,
        drones,
        run_dir / "coverage_heatmap.png",
        title="Random-search best coverage",
    )
    plot_best_so_far(
        result.history,
        run_dir / "random_search_progress.png",
        title="Random search best feasible fitness",
    )

    evaluation = result.best_evaluation
    print(f"  best fitness : {evaluation.fitness:.2f}")
    print(
        f"  coverage     : {evaluation.coverage:.1%}  "
        f"detection: {evaluation.detection_rate:.1%}"
    )
    print(f"  feasibility  : {result.feasibility_rate:.1%}")
    print(f"  runtime      : {result.runtime_seconds:.2f} s")
    print(f"  output       : {run_dir}")
    return run_dir


def _method_row(
    method: str,
    seed: int,
    budget: int,
    evaluation,
    runtime: float,
    evaluations: int,
    feasibility_rate: float,
    termination_reason: str,
) -> dict:
    return {
        "method": method,
        "seed": seed,
        "budget": budget,
        "evaluations": evaluations,
        "best_fitness": evaluation.fitness,
        "coverage": evaluation.coverage,
        "detection_rate": evaluation.detection_rate,
        "total_energy": evaluation.total_energy,
        "normalized_energy": evaluation.normalized_energy,
        "mission_time": evaluation.mission_time,
        "normalized_time": evaluation.normalized_time,
        "hazard_exposure": evaluation.hazard_exposure,
        "normalized_risk": evaluation.normalized_risk,
        "feasible": evaluation.feasible,
        "feasibility_rate": feasibility_rate,
        "runtime_seconds": runtime,
        "termination_reason": termination_reason,
    }


def _write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        round(value, 6)
                        if isinstance(value, float)
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _comparison_summary(rows: list[dict]) -> list[dict]:
    summary = []
    for method in ("genetic_algorithm", "random_search"):
        group = [row for row in rows if row["method"] == method]
        if not group:
            continue
        fitness = [row["best_fitness"] for row in group]
        mean_fitness = sum(fitness) / len(fitness)
        variance = (
            sum((value - mean_fitness) ** 2 for value in fitness)
            / (len(fitness) - 1)
            if len(fitness) > 1
            else 0.0
        )
        summary.append(
            {
                "method": method,
                "n_runs": len(group),
                "mean_best_fitness": mean_fitness,
                "std_best_fitness": variance**0.5,
                "mean_coverage": sum(row["coverage"] for row in group) / len(group),
                "mean_detection_rate": sum(
                    row["detection_rate"] for row in group
                )
                / len(group),
                "mean_total_energy": sum(
                    row["total_energy"] for row in group
                )
                / len(group),
                "mean_mission_time": sum(
                    row["mission_time"] for row in group
                )
                / len(group),
                "mean_hazard_exposure": sum(
                    row["hazard_exposure"] for row in group
                )
                / len(group),
                "mean_feasibility_rate": sum(
                    row["feasibility_rate"] for row in group
                )
                / len(group),
                "mean_runtime_seconds": sum(
                    row["runtime_seconds"] for row in group
                )
                / len(group),
                "mean_evaluations": sum(
                    row["evaluations"] for row in group
                )
                / len(group),
            }
        )
    return summary


def run_compare_mode(args: argparse.Namespace) -> Path:
    seeds = [args.seed + i for i in range(max(1, args.seeds))]
    budget = args.population * args.generations
    run_dir = create_run_dir(args.output_dir, "compare", seeds[0])
    write_json(
        run_dir / "compare_config.json",
        {
            "seeds": seeds,
            "evaluation_budget_per_method": budget,
            "base_config": make_config(args).to_dict(),
            "simulation_modes": modes_table(),
        },
    )
    rows: list[dict] = []
    best_ga = None

    for seed in seeds:
        config = make_config(args, seed=seed)
        env, drones = create_scenario(config)
        seed_dir = run_dir / f"seed_{seed}"
        seed_dir.mkdir()

        ga_dir = seed_dir / "ga"
        ga_dir.mkdir()
        write_run_config(ga_dir, config, args, "compare-ga")
        print(f"\n=== seed {seed}: GA (budget {budget}) ===")
        ga_result = run_ga(
            config,
            env,
            drones,
            progress=None if args.quiet else progress_printer(f"ga {seed}"),
        )
        save_ga_artifacts(
            ga_dir, config, env, drones, ga_result, f"GA seed {seed}"
        )
        rows.append(
            _method_row(
                "genetic_algorithm",
                seed,
                budget,
                ga_result.best_evaluation,
                ga_result.runtime_seconds,
                ga_result.evaluation_requests,
                ga_result.feasibility_rate,
                ga_result.termination_reason,
            )
        )
        if best_ga is None or (
            ga_result.best_evaluation.feasible,
            ga_result.best_evaluation.fitness,
        ) > (
            best_ga[0].best_evaluation.feasible,
            best_ga[0].best_evaluation.fitness,
        ):
            best_ga = (ga_result, config, env, drones)

        print(f"=== seed {seed}: random search (budget {budget}) ===")
        random_dir = seed_dir / "random"
        random_dir.mkdir()
        random_result = run_random_search(
            env,
            drones,
            config.weights,
            n_waypoints=config.fleet.waypoints_per_route,
            budget=budget,
            seed=seed,
            time_reference=config.time_reference,
            progress=(
                None
                if args.quiet
                else lambda index, best, total=budget: print(
                    f"[random {seed}] {index}/{total} | best {best}"
                )
            ),
        )
        write_json(
            random_dir / "best_solution.json",
            _random_payload(config, env, random_result),
        )
        write_run_config(random_dir, config, args, "compare-random")
        plot_mission_map(
            env,
            random_result.best_chromosome,
            random_result.best_evaluation,
            drones,
            random_dir / "mission_map.png",
            title=f"Random search seed {seed}",
        )
        rows.append(
            _method_row(
                "random_search",
                seed,
                budget,
                random_result.best_evaluation,
                random_result.runtime_seconds,
                random_result.evaluations,
                random_result.feasibility_rate,
                "evaluation_budget",
            )
        )

    comparison_fields = [
        "method",
        "seed",
        "budget",
        "evaluations",
        "best_fitness",
        "coverage",
        "detection_rate",
        "total_energy",
        "normalized_energy",
        "mission_time",
        "normalized_time",
        "hazard_exposure",
        "normalized_risk",
        "feasible",
        "feasibility_rate",
        "runtime_seconds",
        "termination_reason",
    ]
    _write_rows(run_dir / "comparison.csv", comparison_fields, rows)
    summary = _comparison_summary(rows)
    _write_rows(
        run_dir / "comparison_summary.csv",
        [
            "method",
            "n_runs",
            "mean_best_fitness",
            "std_best_fitness",
            "mean_coverage",
            "mean_detection_rate",
            "mean_total_energy",
            "mean_mission_time",
            "mean_hazard_exposure",
            "mean_feasibility_rate",
            "mean_runtime_seconds",
            "mean_evaluations",
        ],
        summary,
    )
    plot_algorithm_comparison(rows, run_dir / "comparison.png")
    if best_ga is not None:
        ga_result, ga_config, ga_env, _ = best_ga
        write_json(
            run_dir / "best_ga_solution.json",
            _ga_payload(ga_config, ga_env, ga_result),
        )

    print("\nComparison summary (means over seeds):")
    for row in summary:
        print(
            f"  {row['method']:<18} fitness {row['mean_best_fitness']:.2f} "
            f"+/- {row['std_best_fitness']:.2f} | "
            f"coverage {row['mean_coverage']:.1%} | "
            f"runtime {row['mean_runtime_seconds']:.2f}s"
        )
    print(f"  output: {run_dir}")
    return run_dir


def run_sensitivity_mode(args: argparse.Namespace) -> Path:
    config = make_config(args)
    env, drones = create_scenario(config)
    run_dir = create_run_dir(args.output_dir, "sensitivity", config.ga.seed)
    write_run_config(run_dir, config, args, "sensitivity")
    seeds = [args.seed + i for i in range(max(1, args.sensitivity_seeds))]

    print(
        f"Sensitivity | seeds={seeds} generations={args.sensitivity_generations} "
        f"full_factorial={args.full_factorial}"
    )

    def progress(done: int, total: int, combo: str, seed: int, fitness: float):
        print(
            f"[sensitivity] {done}/{total} | {combo} | seed {seed} | "
            f"fitness {fitness:.2f}"
        )

    rows, summary = run_sensitivity(
        env,
        drones,
        config.weights,
        config.ga,
        n_waypoints=config.fleet.waypoints_per_route,
        seeds=seeds,
        output_dir=run_dir,
        generations=args.sensitivity_generations,
        full_factorial=args.full_factorial,
        time_reference=config.time_reference,
        progress=None if args.quiet else progress,
    )
    plot_sensitivity_summary(summary, run_dir / "sensitivity.png")

    print("\nSensitivity summary:")
    for row in summary:
        print(
            f"  {row['combo']:<34} mean {row['mean_best_fitness']:.2f} "
            f"+/- {row['std_best_fitness']:.2f} | "
            f"coverage {row['mean_coverage']:.1%}"
        )
    print(f"  runs: {len(rows)} | output: {run_dir}")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.mode == "ga":
            run_ga_mode(args)
        elif args.mode == "random":
            run_random_mode(args)
        elif args.mode == "compare":
            run_compare_mode(args)
        elif args.mode == "sensitivity":
            run_sensitivity_mode(args)
    except EnvironmentGenerationError as exc:
        print(f"Environment generation failed: {exc}")
        return 2
    except ValueError as exc:
        print(f"Invalid configuration: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
