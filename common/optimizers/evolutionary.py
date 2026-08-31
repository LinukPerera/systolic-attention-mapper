"""
optimizers/evolutionary.py — evolutionary/genetic-algorithm candidate for
dataflow+SRAM-partition search.

Uses `deap` (plan §2). Genome: 3 floats in [0,1], decoded via
common.optimizers.environment.decode_raw_action — the same encoding every other
candidate searches (see that function's docstring), for a fair bake-off.

Fitness: each individual's real EvalResult.reward, from one real problem.evaluate()
call — never a synthetic surrogate. The first generation costs population_size real
evaluations; every later generation only re-evaluates individuals whose genome
actually changed under crossover/mutation this generation (standard DEAP practice —
an unchanged individual's fitness is still valid, so re-evaluating it would just be
a wasted, redundant simulator call) — so len(run()'s returned list) is an *upper
bound* of population_size * (n_iterations // population_size), not a guaranteed
exact count, and can vary noticeably run-to-run for small populations/generation
counts (confirmed by hand: population_size=3, n_iterations=6 → 2 generations, and
one seed produced only 3 real evaluations total because no individual changed in
generation 2). This is a real, honest property of how GA works, not a bug — log
whatever count actually comes back, don't pad it to n_iterations.

Operators: two-point crossover (cxTwoPoint) and Gaussian mutation (mutGaussian,
clamped back to [0,1] since the genome IS decode_raw_action's raw input) with
tournament selection (selTournament) — DEAP's standard real-valued GA building
blocks, not hand-rolled.

Seeding: DEAP's tools (cxTwoPoint, mutGaussian, selTournament) use Python's global
`random` module internally, not an injectable RNG instance — so `random.seed(seed)`
is called globally at the start of run(), not on a local Random() instance, to make
a given seed's sequence of evaluated actions actually reproducible.
"""
from __future__ import annotations

import random

from deap import base, creator, tools

from common.optimizers.base import DataflowTilingOptimizer
from common.optimizers.environment import AttentionMappingProblem, EvalResult, decode_raw_action

if not hasattr(creator, "AttentionMappingFitness"):
    creator.create("AttentionMappingFitness", base.Fitness, weights=(1.0,))
if not hasattr(creator, "AttentionMappingIndividual"):
    creator.create("AttentionMappingIndividual", list, fitness=creator.AttentionMappingFitness)


class EvolutionaryOptimizer(DataflowTilingOptimizer):
    """GA candidate (see module docstring for genome/operator/fitness design)."""

    def __init__(
        self,
        *,
        population_size: int = 5,
        crossover_prob: float = 0.5,
        mutation_prob: float = 0.3,
        mutation_sigma: float = 0.2,
        tournament_size: int = 3,
    ) -> None:
        """
        Args:
            population_size: Individuals per generation — from config/optimizers/evolutionary.yaml.
            crossover_prob: Per-pair crossover probability.
            mutation_prob: Per-individual mutation probability.
            mutation_sigma: Gaussian mutation standard deviation (genome is in
                [0,1], so this is directly comparable across genes).
            tournament_size: Tournament selection size.
        """
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_sigma = mutation_sigma
        self.tournament_size = tournament_size

    def run(self, problem: AttentionMappingProblem, *, n_iterations: int, seed: int) -> list[EvalResult]:
        random.seed(seed)  # see module docstring — DEAP's operators use global random
        n_generations = max(1, n_iterations // self.population_size)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.random)
        toolbox.register(
            "individual", tools.initRepeat, creator.AttentionMappingIndividual, toolbox.attr_float, n=3,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=self.mutation_sigma, indpb=0.5)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

        eval_results: list[EvalResult] = []

        def evaluate_individual(individual) -> tuple[float]:
            result = problem.evaluate(decode_raw_action(tuple(individual)))
            eval_results.append(result)
            return (result.reward,)

        population = toolbox.population(n=self.population_size)
        for ind in population:
            ind.fitness.values = evaluate_individual(ind)

        for _ in range(n_generations - 1):
            offspring = [toolbox.clone(ind) for ind in toolbox.select(population, len(population))]

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    toolbox.mutate(mutant)
                    for i in range(len(mutant)):
                        mutant[i] = min(1.0, max(0.0, mutant[i]))
                    del mutant.fitness.values

            for ind in offspring:
                if not ind.fitness.valid:
                    ind.fitness.values = evaluate_individual(ind)

            population[:] = offspring

        return eval_results
