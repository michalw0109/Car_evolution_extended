# ------------------ IMPORTS ------------------
from EvolutionEngine import EvolutionEngine
import random
import numpy as np
from Computation import Computation
from Crossover import Crossover
from Mutation import Mutation, MutationCombiner
from Selection import Selection
from DNA import *
from DNA_Initialization import Initializer
from DNA_Decoder import Decoder
import sys


def main() -> None:
    r = int(sys.argv[1])
    np.random.seed(r)
    random.seed(r)

    MAX_GENERATIONS = 10000
    POPULATION_SIZE = 1600      # total, divided evenly across islands
    ELITE_FRACTION = 0

    # True = interactive draw mode; False = load from assets/train*.png / val*.png
    DRAW_TRAIN_TRACK = True
    DRAW_VAL_TRACK = False

    # --- Island / population graph parameters ---
    NUM_SUBPOPULATIONS = 16      # number of islands
    MIGRATION_INTERVAL = 50     # migrate every N generations
    MIGRATION_SIZE = 1          # top-K migrants sent along each edge

    # Graph topology: 'ring' | 'complete' | 'grid' | adjacency matrix (list of lists)
    ISLAND_GRAPH = 'grid'

    # --- Full-population checkpointing (recover an entire run) ---
    # Every CHECKPOINT_INTERVAL generations, every individual's DNA is dumped
    # (grouped by island) to evolutionOutput/<run>/checkpoints/gen_<n>.txt.
    # 0 disables it.  To resume a crashed/finished run, set RESUME_CHECKPOINT to
    # such a file (e.g. "evolutionOutput/2026.06.16_12.00.00/checkpoints/gen_000500.txt").
    CHECKPOINT_INTERVAL = 100
    RESUME_CHECKPOINT = None

    # --- Simulation ---
    # Sensor ray-march step in pixels.  Single source of truth for every route:
    # the headless training sim (NumbaSim / BatchedSim fallback) and Car.py
    # (validation + drawn simulation) all receive this value.
    SENSOR_STEP = 1

    INPUTS = [0, 1, 2, 3, 4, 5]
    OUTPUTS = [12, 13, 14, 15]
    MARKER = [0, 1, 1, 1, 1, 1, 1, 1]

    DNA_INITIALIZATION = Initializer().init_single_DNA_one_chromosome(INPUTS, OUTPUTS, MARKER).random_bits
    DNA_DECODER = Decoder().decodes_single_DNA_one_chromosome(INPUTS, OUTPUTS, MARKER).connection_based_markers

    COMPUTATION = Computation(INPUTS, OUTPUTS, MARKER).connection_based_sort_feed_forward

    SELECTION = Selection().tournament_selection

    CROSSOVER = Crossover().crossover_single_DNA_one_chromosome(_MARKER=MARKER).connection_based_markers

    MUTATION = MutationCombiner([
        Mutation().mutate_single_DNA_one_chromosome(_MARKER=MARKER).random_bit_flip,
        Mutation().mutate_single_DNA_one_chromosome(_MARKER=MARKER).random_insert,
        Mutation().mutate_single_DNA_one_chromosome(_MARKER=MARKER).random_delete,
    ]).mutate

    window = EvolutionEngine(
        MAX_GENERATIONS, POPULATION_SIZE, ELITE_FRACTION,
        DNA_INITIALIZATION, DNA_DECODER, COMPUTATION,
        SELECTION, CROSSOVER, MUTATION,
        _NUM_SUBPOPULATIONS=NUM_SUBPOPULATIONS,
        _MIGRATION_INTERVAL=MIGRATION_INTERVAL,
        _MIGRATION_SIZE=MIGRATION_SIZE,
        _ISLAND_GRAPH=ISLAND_GRAPH,
        _DRAW_TRAIN_TRACK=DRAW_TRAIN_TRACK,
        _DRAW_VAL_TRACK=DRAW_VAL_TRACK,
        _CHECKPOINT_INTERVAL=CHECKPOINT_INTERVAL,
        _RESUME_CHECKPOINT=RESUME_CHECKPOINT,
        _SENSOR_STEP=SENSOR_STEP,
    )
    window.run()

if __name__ == "__main__":
    main()
