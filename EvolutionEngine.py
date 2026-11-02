# ------------------ IMPORTS ------------------
import math
import random
import pygame
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import cv2
from datetime import datetime
import os
from Colors import Color
from Car import Car
import numpy as np
import statistics
from copy import deepcopy
from Individual import Individual
from DNA_Decoder import decode_rates
from BatchedSim import build_weight_tensors, simulate
from NumbaSim import simulate_numba, NUMBA_AVAILABLE


# ------------------ STANDARD GRAPH COLORS ------------------
# Single source of truth so graph() and graph_islands() stay consistent.
COLOR_BEST_TRAIN = "royalblue"   # best fitness on the train track
COLOR_AVG_TRAIN  = "red"         # average / global-average train fitness
COLOR_VAL        = "green"       # best fitness on the validation track
COLOR_CROSSOVER  = "goldenrod"   # average crossover rate (secondary axis)
COLOR_MUTATION   = "black"       # average mutation rate (secondary axis)


class EvolutionEngine:

    def __init__(self, _MAX_GENERATIONS, _POPULATION_SIZE, _ELITE_FRACTION, _DNA_INITIALIZATION, _DNA_DECODER, _COMPUTATION, _SELECTION, _CROSSOVER, _MUTATION,
                 _NUM_SUBPOPULATIONS=1, _MIGRATION_INTERVAL=10, _MIGRATION_SIZE=1, _ISLAND_GRAPH='ring',
                 _DRAW_TRAIN_TRACK=False, _DRAW_VAL_TRACK=False,
                 _CHECKPOINT_INTERVAL=0, _RESUME_CHECKPOINT=None, _SENSOR_STEP=1):


        # some execution params, arent in research, can be constant
        self.FPS = 120000


        self.COLLISION_SURFACE_COLOR = Color.GREEN

        # Linear track: True = interactive draw, False = load from assets/
        self.DRAW_TRAIN_TRACK = _DRAW_TRAIN_TRACK
        self.DRAW_VAL_TRACK = _DRAW_VAL_TRACK

        self.LOAD_MODEL = False

        self.DRAW_SIMULATION = False

        # get evolution params from main
        self.MAX_GENERATIONS = _MAX_GENERATIONS
        self.POPULATION_SIZE = _POPULATION_SIZE
        self.ELITE_FRACTION = _ELITE_FRACTION

        self.DNA_init = _DNA_INITIALIZATION
        self.DNA_decode = _DNA_DECODER
        self.compute = _COMPUTATION
        self.selection = _SELECTION
        self.crossover = _CROSSOVER
        self.mutate = _MUTATION



        # some iniitialization of variables
        pygame.init()

        info = pygame.display.Info()

        self.WINDOW_WIDTH = int(info.current_w * 0.9)
        self.WINDOW_HEIGHT = int(info.current_h * 0.9)

        #self.WINDOW_WIDTH = int(1920 * 0.9)
        #self.WINDOW_HEIGHT = int(1080 * 0.9)

        self.elite_amount = int(self.POPULATION_SIZE * self.ELITE_FRACTION)
        self.new_amount = int(self.POPULATION_SIZE * 0.1)
        self.TIME = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")

        self.output_path = "evolutionOutput/" + self.TIME

        os.makedirs(self.output_path, exist_ok=True)

        self.WINDOW_MID_POINT = (self.WINDOW_WIDTH / 2, self.WINDOW_HEIGHT / 2)

        self.population: list[Individual] = []
        self.new_population : list[Individual]= []

        self.generationsList = [0]
        self.bestFitnessList = [0]
        self.medianFitnessList = [0]
        self.testFitnessList = [0]
        self.avgCrossoverRateList = [0.5]
        self.avgMutationRateList = [0.5]

        self.fig, self.ax = plt.subplots(1, 1)

        self.TITLE = "Ewolucja pojazdów"
        pygame.display.set_caption(self.TITLE)


        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        self.screen.fill(self.COLLISION_SURFACE_COLOR)

        self.track = None          # list[pygame.Surface], 2*n segments
        self.train_total_w = 0

        self.STARTING_POINT: list[float] = []
        self.STARTING_ANGLE = None

        self.valTrack = None       # list[pygame.Surface], 2*n segments
        self.val_total_w = 0

        self.STARTING_VAL_POINT: list[float] = []
        self.STARTING_VAL_ANGLE = None

        self.TIMING_CLOCK = pygame.time.Clock()

        self.generationNumber = 1

        self.CAR_WIDTH = self.WINDOW_WIDTH * 0.008
        self.CAR_HEIGHT = self.CAR_WIDTH * 0.6

        # Minimum brush radius when drawing track segments.  A stroke is a circle
        # of radius `brushSize`, so its width = 2*brushSize.  Clamping the radius
        # to CAR_WIDTH guarantees every drawn passage is at least 2*CAR_WIDTH wide,
        # i.e. you can't draw a corridor the cars wouldn't fit through.
        self.MIN_BRUSH_SIZE = max(1, int(round(1.1 * self.CAR_HEIGHT)))

        self.collision_map = None
        self.collision_val_map = None

        self.best_indv_val_fitness = 0
        self.best_val_checkpoint_fitness = 0

        # --- Linear track ---
        self.SEGMENT_WIDTH = self.WINDOW_WIDTH  # width of one drawn segment

        # Max ray length for the batched (headless) sensor sim.  The original
        # per-car march was unbounded; vectorising needs a cap.  Generous default
        # (~3 segments); raise toward exactness at the cost of memory/time.
        self.MAX_SENSOR_DIST = 1.0 * self.SEGMENT_WIDTH

        # Sensor ray-march step (px), from main.py.  Passed to every route —
        # NumbaSim, the BatchedSim fallback and Car.py — so all sense identically.
        self.SENSOR_STEP = _SENSOR_STEP

        # --- Island model ---
        self.NUM_SUBPOPULATIONS = _NUM_SUBPOPULATIONS
        self.MIGRATION_INTERVAL = _MIGRATION_INTERVAL
        self.MIGRATION_SIZE = _MIGRATION_SIZE
        self._island_graph_spec = _ISLAND_GRAPH
        self.SUBPOP_SIZE = self.POPULATION_SIZE // _NUM_SUBPOPULATIONS
        self.islands: list[list[Individual]] = []
        self.new_islands: list[list[Individual]] = []
        self.island_neighbors: list[list[int]] = []
        self.island_fitness_lists: list[list[float]] = [[0] for _ in range(_NUM_SUBPOPULATIONS)]
        self.fig_islands, self.ax_islands = plt.subplots(1, 1)

        # Render the matplotlib graphs only every GRAPH_INTERVAL generations
        # (data is still TRACKED every generation).  Drawing 3 figures/gen is
        # cheap relative to a 25k-individual step but dominant at small pops.
        self.GRAPH_INTERVAL = 3

        # --- Full-population checkpointing (crash recovery / resume) ---
        # Every CHECKPOINT_INTERVAL generations, dump EVERY individual's DNA,
        # grouped by island, to <output>/checkpoints/gen_<n>.txt.  Point
        # _RESUME_CHECKPOINT at such a file to reload the whole population
        # (per island) and continue training from that generation.
        # 0 disables periodic checkpointing.
        self.CHECKPOINT_INTERVAL = _CHECKPOINT_INTERVAL
        self._resume_checkpoint = _RESUME_CHECKPOINT
        self.checkpoint_path = self.output_path + "/checkpoints"
        if self.CHECKPOINT_INTERVAL > 0:
            os.makedirs(self.checkpoint_path, exist_ok=True)

        # --- Diversity instrumentation (to watch premature convergence) ---
        # 'genotypic' | 'phenotypic_jaccard' | 'unique_fraction' | 'fitness'
        self.DIVERSITY_METRIC = 'phenotypic_jaccard'
        self.island_diversity_lists: list[list[float]] = [[0.0] for _ in range(_NUM_SUBPOPULATIONS)]
        # Diversity of the WHOLE population pooled across islands (between+within).
        self.global_diversity_list: list[float] = [0.0]
        self.fig_div, self.ax_div = plt.subplots(1, 1)

        # --- Best-network visualisation (global best train / val) ---
        self._best_train_nn = None
        self._best_train_fitness = -1.0
        self._best_val_nn = None
        self._best_val_fitness = -1.0
        self.fig_net, self.ax_net = plt.subplots(1, 1)




    # ==================== LINEAR TRACK SYSTEM ====================

    # Tracks live in assets/<prefix>/ (assets/train/ and assets/val/).  Any
    # image in the folder is a segment; filenames only matter for ordering, so
    # you can delete a segment from the middle without renaming the rest.
    TRACK_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

    @staticmethod
    def _segment_sort_key(filename: str):
        """Ordering key with a dotted-decimal "insert between" convention.

        The filename stem is split on '.' and each part compared numerically:
            1  <  1.5  <  1.7  <  2  <  10  <  10.3
        So to drop a segment BETWEEN '5' and '6', name it '5.5'; between '5' and
        '5.5', name it '5.2'; and so on — neighbours never need renaming.

        Each part becomes (rank, int, str): numeric parts (rank 0) sort before
        any non-numeric part (rank 1) at the same position, and a shorter name
        sorts before a longer one sharing its prefix (so '5' < '5.5')."""
        stem = os.path.splitext(filename)[0]
        key = []
        for part in stem.split('.'):
            if part.isdigit():
                key.append((0, int(part), ""))
            else:
                key.append((1, 0, part.lower()))
        return key

    def _track_dir(self, prefix: str) -> str:
        return os.path.join("assets", prefix)

    def drawSegments(self, prefix: str) -> list:
        """Interactive per-segment drawing.  Enter = save segment + start next, Esc = done."""
        segments = []
        seg_num = 1

        while True:
            segment = pygame.Surface((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            segment.fill(self.COLLISION_SURFACE_COLOR)

            drawing = False
            erasing = False
            brushSize = max(20, self.MIN_BRUSH_SIZE)
            running = True

            pygame.display.set_caption(
                f"{self.TITLE}     Rysuj segment {prefix}{seg_num} "
                f"| Enter=zapisz i następny, Esc=zakończ "
                f"| min pędzel={self.MIN_BRUSH_SIZE}px (2x szerokość auta)"
            )

            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit()
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            drawing = True
                        elif event.button == 3:
                            erasing = True
                        elif event.button == 4 and brushSize < 1000:
                            brushSize += 1
                        elif event.button == 5 and brushSize > self.MIN_BRUSH_SIZE:
                            brushSize = max(self.MIN_BRUSH_SIZE, brushSize - 1)
                    elif event.type == pygame.MOUSEBUTTONUP:
                        if event.button == 1:
                            drawing = False
                        elif event.button == 3:
                            erasing = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN:
                            folder = self._track_dir(prefix)
                            os.makedirs(folder, exist_ok=True)
                            path = os.path.join(folder, f"{seg_num}.png")
                            pygame.image.save(segment, path)
                            segments.append(segment.copy())
                            seg_num += 1
                            running = False
                        elif event.key == pygame.K_ESCAPE:
                            return segments

                if drawing:
                    pygame.draw.circle(segment, Color.BLACK, pygame.mouse.get_pos(), brushSize)
                if erasing:
                    pygame.draw.circle(segment, self.COLLISION_SURFACE_COLOR, pygame.mouse.get_pos(), brushSize)

                self.screen.blit(segment, (0, 0))
                # Guide dots: left-center and right-center of the segment
                pygame.draw.circle(self.screen, Color.RED, (4, self.WINDOW_HEIGHT // 2), 6)
                pygame.draw.circle(self.screen, Color.RED, (self.WINDOW_WIDTH - 5, self.WINDOW_HEIGHT // 2), 6)

                if running:
                    pygame.draw.circle(self.screen, Color.BLACK, pygame.mouse.get_pos(), brushSize, 1)

                pygame.display.update()
                self.TIMING_CLOCK.tick(250)

    def loadSegments(self, prefix: str) -> list:
        """Load every image in assets/<prefix>/ (e.g. assets/train/), ordered by
        the dotted-decimal segment convention (see _segment_sort_key), scaled to
        window size.  Delete or insert segments by name without touching the rest
        (e.g. name a file '5.5.png' to place it between '5.png' and '6.png').

        A file whose name starts with 'x' (or 'X') is disabled: it stays in the
        folder but is skipped, so you can toggle a segment off without deleting
        it (e.g. rename '7.png' -> 'x7.png')."""
        folder = self._track_dir(prefix)
        segments = []
        if not os.path.isdir(folder):
            return segments
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(self.TRACK_IMAGE_EXTS)
                 and not f.lower().startswith("x")]
        files.sort(key=self._segment_sort_key)
        for fname in files:
            seg = pygame.image.load(os.path.join(folder, fname))
            seg = pygame.transform.scale(seg, (self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            segments.append(seg)
        return segments

    def assembleTrack(self, segments: list):
        """Concatenate segments into a 2-lap track.
        Lap 1: in drawing order.  Lap 2: shuffled randomly.
        Returns (all_segs, numpy_collision_map) where all_segs is a list of
        2*n pygame.Surface objects.  No single giant surface is created,
        avoiding pygame surface-width limits for large segment counts.
        """
        shuffled = list(segments)
        random.shuffle(shuffled)
        all_segs = list(segments) + shuffled

        seg_cmaps = []
        for seg in all_segs:
            px = pygame.surfarray.pixels3d(seg)
            seg_cmaps.append(np.all(px == self.COLLISION_SURFACE_COLOR, axis=2).copy())
            del px
        collision_map = np.concatenate(seg_cmaps, axis=0)

        return all_segs, collision_map

    def buildTrack(self, draw_flag: bool, prefix: str):
        """Draw or load segments then assemble.  Returns (all_segs, collision_map)."""
        if draw_flag:
            segments = self.drawSegments(prefix)
        else:
            segments = self.loadSegments(prefix)

        if not segments:
            raise ValueError(
                f"No track segments found in '{self._track_dir(prefix)}/'. "
                f"Add image files there, or set the draw flag to draw a track."
            )

        return self.assembleTrack(segments)

    def createPopulation(self):
        self.new_population.clear()
        for _ in range(self.POPULATION_SIZE):
            indv = Individual()
            indv.dnaType = self.DNA_init()
            self.new_population.append(indv)
        if self._resume_checkpoint:
            islands, gen = self.load_population_checkpoint(self._resume_checkpoint)
            flat = [indv for island in islands for indv in island]
            if flat:
                self.new_population = flat
                self.generationNumber = gen
                print(f"Wznowiono trening (pojedyncza populacja) z "
                      f"'{self._resume_checkpoint}': generacja {gen}, {len(flat)} osobników")
            return
        if self.LOAD_MODEL:
            fullDataPath = "assets/bestDNA.txt"
            with open(fullDataPath, 'r') as file:
                for line in file:
                    line = line.strip()
                    dna = []
                    for char in line:
                        dna.append(int(char))
                    self.new_population[0].dnaType.DNA = dna



    def dekodeDNAToNetwork(self):
        for i in range(self.POPULATION_SIZE):
            self.population[i].nn = self.DNA_decode(self.population[i].dnaType)
            self.population[i].crossover_rate, self.population[i].mutation_rate = decode_rates(self.population[i].dnaType)

    def _blit_track(self, segs: list, camera_x: float) -> None:
        """Tile two segments onto self.screen at the given camera offset."""
        seg_idx = int(camera_x) // self.WINDOW_WIDTH
        off_x = int(camera_x) % self.WINDOW_WIDTH
        self.screen.fill(self.COLLISION_SURFACE_COLOR)
        if seg_idx < len(segs):
            self.screen.blit(segs[seg_idx], (-off_x, 0))
        if seg_idx + 1 < len(segs):
            self.screen.blit(segs[seg_idx + 1], (self.WINDOW_WIDTH - off_x, 0))

    def evaluate_fitness(self):
        """Evaluate the population on the train track.

        Headless (DRAW_SIMULATION=False): use the vectorised batched simulation
        over all cars at once — orders of magnitude faster.  With drawing on,
        fall back to the per-car renderer so the simulation can be watched.

        Both paths honour the `new` flag: individuals carried over unchanged
        (elites/immigrants, new=False) keep their cached fitness instead of being
        re-simulated.  The train track is fixed for the whole run, so a re-sim
        would produce the identical value anyway.
        """
        if self.DRAW_SIMULATION:
            self._evaluate_fitness_drawn()
            return

        comp = self.compute.__self__          # the Computation instance
        new_idx = [i for i in range(self.POPULATION_SIZE) if self.population[i].new]
        if not new_idx:
            return

        W = build_weight_tensors([self.population[i].nn for i in new_idx], comp.NR_OF_NEURONS)

        if NUMBA_AVAILABLE:
            # Compiled, per-car-parallel, exact march-until-hit (no sensor cap).
            input_ids = np.asarray(comp.INPUTS)
            output_ids = np.asarray(comp.OUTPUTS)
            is_input = np.zeros(comp.NR_OF_NEURONS, dtype=np.bool_)
            is_input[input_ids] = True
            fitness = simulate_numba(
                W, input_ids, output_ids, is_input, self.collision_map,
                float(self.STARTING_POINT[0]), float(self.STARTING_POINT[1]),
                float(self.STARTING_ANGLE),
                float(self.train_total_w), float(self.WINDOW_HEIGHT),
                float(self.SEGMENT_WIDTH), float(self.CAR_WIDTH), float(self.CAR_HEIGHT),
                float(self.SENSOR_STEP), 400000,
            )
        else:
            # Pure-NumPy fallback (sensor range capped at MAX_SENSOR_DIST).
            fitness = simulate(
                W, comp.INPUTS, comp.OUTPUTS, self.collision_map,
                self.STARTING_POINT, self.STARTING_ANGLE,
                self.train_total_w, self.WINDOW_HEIGHT, self.SEGMENT_WIDTH,
                self.CAR_WIDTH, self.CAR_HEIGHT, self.MAX_SENSOR_DIST,
                sensor_step=self.SENSOR_STEP,
            )
        for j, i in enumerate(new_idx):
            self.population[i].fitness = float(fitness[j])

    def _evaluate_fitness_drawn(self):
        CarPopulation = []
        for i in range(self.POPULATION_SIZE):
            CarPopulation.append(Car(
                self.STARTING_POINT, self.STARTING_ANGLE,
                self.population[i].nn, self.compute, self.collision_map,
                (self.CAR_WIDTH, self.CAR_HEIGHT),
                self.train_total_w, self.WINDOW_HEIGHT,
                _segment_width=self.SEGMENT_WIDTH,
                _sensor_step=self.SENSOR_STEP,
            ))

        remainingCars = self.POPULATION_SIZE
        timer = 0
        bestFitness = 0
        bestIndex = 0
        camera_x = 0.0

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()

            if self.DRAW_SIMULATION:
                # Camera follows the rightmost alive car
                best_x = self.STARTING_POINT[0]
                for i in range(self.POPULATION_SIZE):
                    if CarPopulation[i].alive and CarPopulation[i].position[0] > best_x:
                        best_x = CarPopulation[i].position[0]
                camera_x = max(0.0, best_x - self.WINDOW_WIDTH / 2)
                camera_x = min(camera_x, self.train_total_w - self.WINDOW_WIDTH)
                self._blit_track(self.track, camera_x)

            for i in range(self.POPULATION_SIZE):
                if not (self.population[i].new) and CarPopulation[i].alive:
                    CarPopulation[i].alive = False
                    CarPopulation[i].fitness = self.population[i].fitness
                    remainingCars -= 1
                if CarPopulation[i].alive:
                    CarPopulation[i].update(timer)
                    if not CarPopulation[i].alive:
                        remainingCars -= 1
                    if CarPopulation[i].fitness > bestFitness:
                        bestFitness = CarPopulation[i].fitness
                        bestIndex = i
                if self.DRAW_SIMULATION:
                    CarPopulation[i].draw(self.screen, camera_x)

            if remainingCars == 0:
                break

            if timer % 5 == 0:
                t = "Generacja: " + str(self.generationNumber)
                t2 = "Żywych: " + str(remainingCars)
                t3 = "Czas: " + str(round(timer / 120, 2)) + "s"
                t4 = "Obecny najlepszy wynik generacji: " + str(round(bestFitness, 2))
                pygame.display.set_caption(self.TITLE + " - " + t + " - " + t2 + " - " + t3 + " - " + t4)

            if self.DRAW_SIMULATION:
                pygame.display.update()
            self.TIMING_CLOCK.tick(self.FPS)
            timer += 1

        for i in range(self.POPULATION_SIZE):
            self.population[i].fitness = CarPopulation[i].fitness

    def validateIndividual(self, individual: Individual):
        car = Car(
            self.STARTING_VAL_POINT, self.STARTING_VAL_ANGLE,
            individual.nn, self.compute, self.collision_val_map,
            (self.CAR_WIDTH, self.CAR_HEIGHT),
            self.val_total_w, self.WINDOW_HEIGHT,
            _segment_width=self.SEGMENT_WIDTH,
            _sensor_step=self.SENSOR_STEP,
        )
        timer = 0
        bestFitness = 0
        camera_x = 0.0

        if not individual.new:
            car.fitness = individual.testFitness
        else:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit()
                if self.DRAW_SIMULATION:
                    best_x = max(car.position[0], self.STARTING_VAL_POINT[0])
                    camera_x = max(0.0, best_x - self.WINDOW_WIDTH / 2)
                    camera_x = min(camera_x, self.val_total_w - self.WINDOW_WIDTH)
                    self._blit_track(self.valTrack, camera_x)
                car.update(timer)
                if self.DRAW_SIMULATION:
                    car.draw(self.screen, camera_x)
                bestFitness = car.fitness
                individual.testFitness = car.fitness
                if not car.alive:
                    break
                if timer % 5 == 0:
                    t = "Generacja: " + str(self.generationNumber)
                    t4 = "Walidacja najlepszego: " + str(round(bestFitness, 2))
                    pygame.display.set_caption(self.TITLE + " - " + t + " - " + t4)
                if self.DRAW_SIMULATION:
                    pygame.display.update()
                self.TIMING_CLOCK.tick(self.FPS)
                timer += 1

        self.best_indv_val_fitness = car.fitness

    def validateBest(self):
        self.validateIndividual(self.population[0])


    # ==================== ISLAND MODEL ====================

    def _build_island_graph(self) -> list[list[int]]:
        n = self.NUM_SUBPOPULATIONS
        spec = self._island_graph_spec

        if isinstance(spec, str):
            if spec == 'ring':
                return [[(i - 1) % n, (i + 1) % n] for i in range(n)]
            elif spec == 'complete':
                return [[j for j in range(n) if j != i] for i in range(n)]
            elif spec == 'grid':
                cols = max(1, int(n ** 0.5))
                rows = (n + cols - 1) // cols
                neighbors = []
                for i in range(n):
                    row, col = divmod(i, cols)
                    nbrs = []
                    if row > 0:
                        nbrs.append((row - 1) * cols + col)
                    if row < rows - 1 and (row + 1) * cols + col < n:
                        nbrs.append((row + 1) * cols + col)
                    if col > 0:
                        nbrs.append(row * cols + col - 1)
                    if col < cols - 1 and row * cols + col + 1 < n:
                        nbrs.append(row * cols + col + 1)
                    neighbors.append(nbrs)
                return neighbors

        if isinstance(spec, list) and n > 0:
            # Adjacency matrix (list of n lists of n ints 0/1)
            if isinstance(spec[0], list):
                return [[j for j in range(n) if spec[i][j]] for i in range(n)]
            # Already an adjacency list
            return spec

        return [[(i - 1) % n, (i + 1) % n] for i in range(n)]

    def createIslandPopulations(self):
        self.new_islands = []
        for _ in range(self.NUM_SUBPOPULATIONS):
            island = []
            for _ in range(self.SUBPOP_SIZE):
                indv = Individual()
                indv.dnaType = self.DNA_init()
                island.append(indv)
            self.new_islands.append(island)
        if self._resume_checkpoint:
            self._restore_islands_from_checkpoint(self._resume_checkpoint)
            return
        if self.LOAD_MODEL:
            fullDataPath = "assets/bestDNA.txt"
            with open(fullDataPath, 'r') as file:
                for line in file:
                    line = line.strip()
                    dna = [int(char) for char in line]
                    self.new_islands[0][0].dnaType.DNA = dna

    # ==================== POPULATION CHECKPOINTING ====================

    def _dna_to_string(self, dnaType) -> str:
        """Serialise one individual's DNA to a single text line.
        Single-chromosome -> a bitstring.  Double -> 'bitsA bitsB'."""
        if getattr(dnaType, 'DNA', None) is not None:
            return "".join(str(b) for b in dnaType.DNA)
        if hasattr(dnaType, 'DNAa'):
            return ("".join(str(b) for b in dnaType.DNAa) + " " +
                    "".join(str(b) for b in dnaType.DNAb))
        return ""

    def _string_to_dna(self, line: str):
        """Inverse of _dna_to_string.  Builds a DNA object of the run's active
        type via DNA_init() (so the class always matches the config), then
        overwrites its bits from the saved line."""
        dt = self.DNA_init()
        if " " in line:
            a, b = line.split(" ", 1)
            dt.DNAa = [int(c) for c in a]
            dt.DNAb = [int(c) for c in b]
        else:
            dt.DNA = [int(c) for c in line]
        return dt

    def save_population_checkpoint(self):
        """Every CHECKPOINT_INTERVAL generations, dump EVERY individual's DNA,
        grouped by island, so the whole run can be recovered later.  One file
        per checkpoint: <output>/checkpoints/gen_<completed>.txt.

        Saves the just-evaluated population of the completed generation; on
        resume it is reloaded as the starting population for that same
        generation and evolution continues (see _RESUME_CHECKPOINT)."""
        if self.CHECKPOINT_INTERVAL <= 0:
            return
        completed = self.generationNumber - 1   # gen just evaluated + evolved
        if completed % self.CHECKPOINT_INTERVAL != 0:
            return

        islands = self.islands if self.NUM_SUBPOPULATIONS > 1 else [self.population]
        total = sum(len(isl) for isl in islands)

        path = f"{self.checkpoint_path}/gen_{completed:06d}.txt"
        with open(path, 'w') as f:
            f.write(f"GENERATION {completed}\n")
            f.write(f"NUM_SUBPOPULATIONS {len(islands)}\n")
            f.write(f"POPULATION_SIZE {total}\n")
            for k, island in enumerate(islands):
                f.write(f"ISLAND {k} {len(island)}\n")
                for indv in island:
                    f.write(self._dna_to_string(indv.dnaType) + "\n")
        print(f"  >> Zapisano checkpoint populacji: {path} ({total} osobników)")

    def load_population_checkpoint(self, path: str):
        """Parse a checkpoint written by save_population_checkpoint.
        Returns (islands, generation) with fresh, unevaluated (new=True)
        individuals so they are re-decoded and re-simulated on resume."""
        islands: list[list[Individual]] = []
        generation = 0
        current = None
        with open(path, 'r') as f:
            for raw in f:
                line = raw.rstrip("\n")
                if not line:
                    continue
                if line.startswith("GENERATION"):
                    generation = int(line.split()[1])
                elif line.startswith(("NUM_SUBPOPULATIONS", "POPULATION_SIZE")):
                    continue
                elif line.startswith("ISLAND"):
                    current = []
                    islands.append(current)
                elif current is not None:
                    indv = Individual()
                    indv.dnaType = self._string_to_dna(line)
                    indv.new = True
                    indv.fitness = 0
                    indv.testFitness = 0
                    current.append(indv)
        return islands, generation

    def _restore_islands_from_checkpoint(self, path: str):
        """Replace the freshly-built random islands with the saved population
        and rewind generationNumber so training continues from the checkpoint."""
        islands, gen = self.load_population_checkpoint(path)
        if len(islands) != self.NUM_SUBPOPULATIONS:
            print(f"OSTRZEŻENIE: checkpoint ma {len(islands)} wysp, "
                  f"konfiguracja oczekuje {self.NUM_SUBPOPULATIONS}. "
                  f"Używam struktury z checkpointu.")
        self.new_islands = islands
        self.generationNumber = gen
        total = sum(len(isl) for isl in islands)
        print(f"Wznowiono trening z checkpointu '{path}': "
              f"generacja {gen}, {len(islands)} wysp, {total} osobników")

    def _plan_migration(self):
        """Decide this generation's emigrants (true MOVE, no copy).

        The top MIGRATION_SIZE individuals of every island leave for a randomly
        chosen neighbour and are NOT kept on their source island.  Returns:
          immigrants[dst]     – list[Individual] moving INTO island dst
          emigrant_count[src] – how many of island src's best left
        An island never sends so many that it would be emptied (keeps >=1).
        """
        immigrants: list[list[Individual]] = [[] for _ in range(self.NUM_SUBPOPULATIONS)]
        emigrant_count = [0] * self.NUM_SUBPOPULATIONS

        for src_idx in range(self.NUM_SUBPOPULATIONS):
            nbrs = self.island_neighbors[src_idx]
            if not nbrs:
                continue
            src = self.islands[src_idx]  # current gen, already sorted best-first
            k = max(0, min(self.MIGRATION_SIZE, len(src) - 1))
            for rank in range(k):
                migrant = src[rank].clone()
                migrant.new = True
                migrant.fitness = 0
                migrant.testFitness = 0
                migrant.nn = None
                dst_idx = random.choice(nbrs)              # each migrant chooses freely
                immigrants[dst_idx].append(migrant)
                emigrant_count[src_idx] += 1

        return immigrants, emigrant_count

    def evolve_island(self, island_idx: int, n_emigrants: int = 0, immigrants: list = None):
        """Build island_idx's next generation, kept at exactly SUBPOP_SIZE.

        Composition, in fill order:
          1. elites  – best individuals that did NOT emigrate, i.e. ranks
                       [n_emigrants : n_emigrants + elite_amount].  The few best
                       are still preserved, but anyone who already emigrated is
                       not also kept here (no individual lives in two islands).
          2. immigrants – individuals that moved in from neighbouring islands.
          3. random  – fresh diversity injection (clamped to free space).
          4. children – crossover + mutation fill every remaining slot.
        Robust to any elite/immigrant/random counts: children absorb the slack,
        and the result is trimmed to SUBPOP_SIZE.
        """
        if immigrants is None:
            immigrants = []

        island = self.islands[island_idx]      # sorted best-first
        subpop_size = self.SUBPOP_SIZE
        elite_amount = int(subpop_size * self.ELITE_FRACTION)
        new_amount = int(subpop_size * 0.0)

        new_island: list[Individual] = []

        # 1) Elites from the best NON-migrant individuals
        for rank in range(n_emigrants, n_emigrants + elite_amount):
            if rank < len(island) and len(new_island) < subpop_size:
                new_island.append(island[rank].clone())

        # 2) Immigrants from neighbouring islands
        for migrant in immigrants:
            if len(new_island) < subpop_size:
                new_island.append(migrant)

        # 3) Random fresh individuals (only as much as still fits)
        for _ in range(new_amount):
            if len(new_island) >= subpop_size:
                break
            indv = Individual()
            indv.dnaType = self.DNA_init()
            new_island.append(indv)

        # 4) Children fill whatever remains.  Parents are selected from the full
        #    evaluated island (an emigrant's genes may still breed here; only the
        #    individual itself is gone).
        while len(new_island) < subpop_size:
            parent1 = self.selection(island)
            parent2 = self.selection(island)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            child.new = True
            child.fitness = 0
            child.testFitness = 0
            child.nn = None
            new_island.append(child)

        self.new_islands[island_idx] = new_island[:subpop_size]

    def runEvolutionIslands(self):
        self.createIslandPopulations()

        while self.generationNumber <= self.MAX_GENERATIONS:
            # Advance new_islands → islands.  Shallow per-island list copy: the
            # individuals are not mutated in place during a generation (children
            # are fresh objects), so a deep copy is unnecessary and was costly.
            self.islands = [list(island) for island in self.new_islands]

            # Decode every individual in every island
            for island in self.islands:
                for indv in island:
                    indv.nn = self.DNA_decode(indv.dnaType)
                    indv.crossover_rate, indv.mutation_rate = decode_rates(indv.dnaType)

            # Evaluate all islands together (flat merge)
            flat_pop: list[Individual] = []
            island_ranges: list[tuple[int, int]] = []
            for island in self.islands:
                start = len(flat_pop)
                flat_pop.extend(island)
                island_ranges.append((start, len(flat_pop)))

            saved_pop, saved_size = self.population, self.POPULATION_SIZE
            self.population, self.POPULATION_SIZE = flat_pop, len(flat_pop)
            self.evaluate_fitness()
            self.population, self.POPULATION_SIZE = saved_pop, saved_size

            # Write fitnesses back to island individuals
            for k, (start, end) in enumerate(island_ranges):
                for j, flat_idx in enumerate(range(start, end)):
                    self.islands[k][j].fitness = flat_pop[flat_idx].fitness

            # Sort each island
            for island in self.islands:
                island.sort(key=Individual.sortKey, reverse=True)

            # Validate global best on val track
            global_best = max((island[0] for island in self.islands), key=lambda x: x.fitness)
            self.validateIndividual(global_best)

            # Mark all as not-new for next generation caching
            for island in self.islands:
                for indv in island:
                    indv.new = False

            # Plan migration first (emigrants MOVE, not copy), then rebuild every
            # island with its plan.  On non-migration generations nobody moves.
            if self.generationNumber % self.MIGRATION_INTERVAL == 0:
                immigrants, emigrant_count = self._plan_migration()
                moved = sum(emigrant_count)
                if moved:
                    print(f"  >> Migracja: przeniesiono {moved} osobników")
            else:
                immigrants = [[] for _ in range(self.NUM_SUBPOPULATIONS)]
                emigrant_count = [0] * self.NUM_SUBPOPULATIONS

            for k in range(self.NUM_SUBPOPULATIONS):
                self.evolve_island(k, emigrant_count[k], immigrants[k])

            print(f"Generacja nr. {self.generationNumber}")
            self.generationNumber += 1
            self.save_fitness_results_islands()      # track every generation
            self.save_population_checkpoint()        # full per-island DNA dump every N gens
            if self._should_graph():                 # render only periodically
                self.graph_islands()

    def save_fitness_results_islands(self):
        global_best_ind = max(
            (ind for island in self.islands for ind in island),
            key=lambda x: x.fitness
        )
        all_fitness = [ind.fitness for island in self.islands for ind in island]
        global_best = global_best_ind.fitness
        global_avg = float(np.mean(all_fitness))

        fullPath = self.output_path + "/bestDNA.txt"
        with open(fullPath, 'w') as file:
            file.write("".join(str(b) for b in global_best_ind.dnaType.DNA))

        # Keep the best-ever TRAIN network (for visualisation); the global best
        # can be lost between generations (anti-elitism + migration).
        if global_best_ind.nn is not None and global_best > self._best_train_fitness:
            self._best_train_fitness = global_best
            self._best_train_nn = [dict(c) for c in global_best_ind.nn]

        if self.best_indv_val_fitness > self.best_val_checkpoint_fitness:
            self.best_val_checkpoint_fitness = self.best_indv_val_fitness
            fullValPath = self.output_path + "/bestValDNA.txt"
            with open(fullValPath, 'w') as file:
                file.write("".join(str(b) for b in global_best_ind.dnaType.DNA))
            # The validated individual (global best) just set a new val record:
            # snapshot its network for visualisation.
            self._best_val_fitness = self.best_indv_val_fitness
            if global_best_ind.nn is not None:
                self._best_val_nn = [dict(c) for c in global_best_ind.nn]
            print(f"Nowy najlepszy wynik walidacyjny: {self.best_val_checkpoint_fitness:.3f} -> zapisano checkpoint")

        all_inds = [ind for island in self.islands for ind in island]
        avg_cr = float(np.mean([ind.crossover_rate for ind in all_inds]))
        avg_mr = float(np.mean([ind.mutation_rate for ind in all_inds]))

        island_div = [self._island_diversity(island) for island in self.islands]
        avg_div = float(np.mean(island_div)) if island_div else 0.0
        # Global diversity over the whole pooled population (across islands).
        # Sits >= mean per-island; the GAP between them is between-island diversity.
        global_div = self._island_diversity(all_inds)

        island_bests = [island[0].fitness for island in self.islands]
        print(f"Globalny najlepszy: {global_best:.3f}  |  Walidacja: {self.best_indv_val_fitness:.3f}  |  Średnia: {global_avg:.3f}")
        print(f"Avg crossover_rate: {avg_cr:.4f}  Avg mutation_rate: {avg_mr:.6f}")
        for k, b in enumerate(island_bests):
            nbrs = self.island_neighbors[k]
            print(f"  Wyspa {k} (sąsiedzi: {nbrs}): {b:.3f}")
        print(f"Różnorodność ({self.DIVERSITY_METRIC}) globalna: {global_div:.4f}  "
              f"|  śr. wyspy: {avg_div:.4f}  (między-wyspowa: {global_div - avg_div:.4f})")
        print()

        self.bestFitnessList.append(global_best)
        self.medianFitnessList.append(global_avg)
        self.testFitnessList.append(self.best_indv_val_fitness)
        self.avgCrossoverRateList.append(avg_cr)
        self.avgMutationRateList.append(avg_mr)
        self.generationsList.append(len(self.generationsList))

        for k, b in enumerate(island_bests):
            self.island_fitness_lists[k].append(b)

        for k, d in enumerate(island_div):
            # On the first call, backfill the gen-0 placeholder so the curve
            # starts at the true initial diversity instead of 0.
            if len(self.island_diversity_lists[k]) == 1:
                self.island_diversity_lists[k][0] = d
            self.island_diversity_lists[k].append(d)

        if len(self.global_diversity_list) == 1:
            self.global_diversity_list[0] = global_div
        self.global_diversity_list.append(global_div)

        fullDataPath = self.output_path + "/graphData.txt"
        with open(fullDataPath, 'w') as file:
            for i in range(len(self.bestFitnessList)):
                row = (f"{self.generationsList[i]} {self.bestFitnessList[i]} "
                       f"{self.medianFitnessList[i]} {self.testFitnessList[i]} "
                       f"{self.avgCrossoverRateList[i]} {self.avgMutationRateList[i]}")
                for k in range(self.NUM_SUBPOPULATIONS):
                    row += f" {self.island_fitness_lists[k][i]}"
                file.write(row + "\n")

        fullDivPath = self.output_path + "/diversityData.txt"
        with open(fullDivPath, 'w') as file:
            for i in range(len(self.island_diversity_lists[0])):
                row = f"{self.generationsList[i]}"
                for k in range(self.NUM_SUBPOPULATIONS):
                    row += f" {self.island_diversity_lists[k][i]}"
                file.write(row + "\n")

    def _should_graph(self) -> bool:
        """True on the generations we actually render the matplotlib figures.
        generationNumber was already incremented, so the just-finished generation
        is generationNumber - 1.  Always draw the first and last generation."""
        done = self.generationNumber - 1
        return (done <= 1
                or done >= self.MAX_GENERATIONS
                or done % self.GRAPH_INTERVAL == 0)

    def _save_figure(self, fig, path) -> bool:
        """Save a graph PNG without killing the run if the file is locked.
        On Windows an image viewer (e.g. an IDE preview) can keep the PNG
        memory-mapped; overwriting it then raises OSError (Errno 22).  Skip
        this render instead — the next one retries.  True if written."""
        try:
            fig.savefig(path, bbox_inches="tight")
            return True
        except OSError as e:
            print(f"  >> Pominięto zapis wykresu {os.path.basename(path)} "
                  f"(plik otwarty w innym programie?): {e}")
            return False

    def graph_islands(self):
        tick_spacing = max(1, int(len(self.generationsList) / 10 + 1))
        gens = self.generationsList

        self.ax.clear()
        self.ax.plot(gens, self.medianFitnessList, label="Global avg", color=COLOR_AVG_TRAIN, linewidth=2)
        self.ax.plot(gens, self.testFitnessList, label="Val best", color=COLOR_VAL, linewidth=2)

        island_colors = plt.cm.tab10(np.linspace(0, 1, self.NUM_SUBPOPULATIONS))
        for k in range(self.NUM_SUBPOPULATIONS):
            self.ax.plot(gens, self.island_fitness_lists[k],
                         label=f"Island {k}", color=island_colors[k], linewidth=1, linestyle='--', alpha=0.7)

        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        self.ax.set_xlabel("Generacje")
        self.ax.set_ylabel("Przystosowanie")
        self.ax.grid(True, alpha=0.3)

        ax2 = self.ax.twinx()
        ax2.plot(gens, self.avgCrossoverRateList, label="Avg crossover rate", color=COLOR_CROSSOVER, linewidth=1.8)
        ax2.plot(gens, self.avgMutationRateList, label="Avg mutation rate", color=COLOR_MUTATION, linewidth=1.8)
        ax2.set_ylabel("Rate (0–1)", color="gray")
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='y', labelcolor='gray')

        lines1, labels1 = self.ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        self.fig.legend(lines1 + lines2, labels1 + labels2, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4)
        self.fig.tight_layout()

        fullPath = self.output_path + "/bestFitnessGraph.png"
        if self._save_figure(self.fig, fullPath) and self.DRAW_SIMULATION:
            img = cv2.imread(fullPath, cv2.IMREAD_ANYCOLOR)
            cv2.imshow("Wykres najlepszego wyniku", img)

        # Island topology graph with fitness annotations
        self._draw_island_topology()

        # Per-island diversity graph
        self._draw_island_diversity()

        # Best-network images (global best train / val), pruned to the
        # input->output feed-forward subgraph.
        if self._best_train_nn is not None:
            self._render_network(self._best_train_nn,
                                 f"Najlepsza sieć — trening (fitness {self._best_train_fitness:.2f})",
                                 "/bestTrainNetwork.png", "Sieć — trening")
        if self._best_val_nn is not None:
            self._render_network(self._best_val_nn,
                                 f"Najlepsza sieć — walidacja (fitness {self._best_val_fitness:.2f})",
                                 "/bestValNetwork.png", "Sieć — walidacja")

    def _render_network(self, nn, title, filename, window_name):
        """Draw a decoded network, pruned to the connections that actually carry
        signal from inputs to outputs (no dead ends).  Backward edges are
        reversed and self-loops dropped — matching how Computation evaluates the net.
        Edge colour: red = positive weight, blue = negative; thickness ~ |weight|.
        Nodes: green = input, orange = output, blue-grey = hidden.  Layout is a
        longest-path layering (inputs left, outputs right)."""
        from collections import defaultdict, deque
        comp = self.compute.__self__
        in_set = set(comp.INPUTS)
        out_set = set(comp.OUTPUTS)

        # backward edges reversed, self-loops dropped, duplicate weights summed (matches eval)
        edges = {}
        for c in nn:
            s, t, w = int(c['source']), int(c['target']), float(c['weight'])
            if s > t:
                s, t = t, s
            if s < t:
                edges[(s, t)] = edges.get((s, t), 0.0) + w

        succ = defaultdict(list)
        pred = defaultdict(list)
        for (s, t) in edges:
            succ[s].append(t)
            pred[t].append(s)

        # reachable forward from any input
        reach_in = set(comp.INPUTS)
        dq = deque(comp.INPUTS)
        while dq:
            u = dq.popleft()
            for v in succ.get(u, []):
                if v not in reach_in:
                    reach_in.add(v); dq.append(v)
        # can reach any output (search backwards)
        reach_out = set(comp.OUTPUTS)
        dq = deque(comp.OUTPUTS)
        while dq:
            u = dq.popleft()
            for v in pred.get(u, []):
                if v not in reach_out:
                    reach_out.add(v); dq.append(v)

        # keep only edges on an input -> output path
        use_edges = {(s, t): w for (s, t), w in edges.items()
                     if s in reach_in and t in reach_out}
        nodes = set()
        for (s, t) in use_edges:
            nodes.add(s); nodes.add(t)

        ax = self.ax_net
        ax.clear()
        ax.set_title(title, fontsize=10)
        ax.axis('off')

        if not use_edges:
            ax.text(0.5, 0.5, "brak ścieżki wejście → wyjście",
                    ha='center', va='center', fontsize=11)
            if self._save_figure(self.fig_net, self.output_path + filename) and self.DRAW_SIMULATION:
                cv2.imshow(window_name, cv2.imread(self.output_path + filename, cv2.IMREAD_ANYCOLOR))
            return

        # longest-path layering (node ids are a topological order since s < t)
        upred = defaultdict(list)
        for (s, t) in use_edges:
            upred[t].append(s)
        layer = {}
        for n in sorted(nodes):
            if n in in_set or not upred[n]:
                layer[n] = 0
            else:
                layer[n] = max(layer[p] for p in upred[n]) + 1
        max_ih = max([layer[n] for n in nodes if n not in out_set], default=0)
        for n in nodes:                      # right-align all outputs
            if n in out_set:
                layer[n] = max_ih + 1
        max_layer = max(layer.values())

        cols = defaultdict(list)
        for n in sorted(nodes):
            cols[layer[n]].append(n)
        pos = {}
        for lx, ns in cols.items():
            x = lx / max_layer if max_layer > 0 else 0.5
            m = len(ns)
            for j, n in enumerate(ns):
                pos[n] = (x, 0.5 if m == 1 else j / (m - 1))

        maxw = max(abs(w) for w in use_edges.values())
        for (s, t), w in use_edges.items():
            x1, y1 = pos[s]; x2, y2 = pos[t]
            color = 'red' if w > 0 else 'blue'
            lw = 0.6 + (abs(w) / maxw) * 5.0 if maxw > 0 else 1.0
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                        alpha=0.75, shrinkA=9, shrinkB=9))

        for n in nodes:
            x, y = pos[n]
            color = ('mediumseagreen' if n in in_set
                     else 'orange' if n in out_set else 'lightsteelblue')
            ax.scatter(x, y, s=650, color=color, edgecolors='black', linewidths=1.2, zorder=3)
            ax.text(x, y, str(n), ha='center', va='center', fontsize=8, fontweight='bold', zorder=4)

        ax.text(0.5, -0.08,
                "czerwony = waga + ,  niebieski = waga − ,  grubość ~ |waga|   "
                "(zielony = wejście, pomarańczowy = wyjście)",
                transform=ax.transAxes, ha='center', va='center', fontsize=7, color='gray')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.15, 1.15)
        self.fig_net.tight_layout()
        if self._save_figure(self.fig_net, self.output_path + filename) and self.DRAW_SIMULATION:
            cv2.imshow(window_name, cv2.imread(self.output_path + filename, cv2.IMREAD_ANYCOLOR))

    def _draw_island_topology(self):
        n = self.NUM_SUBPOPULATIONS
        island_bests = [island[0].fitness if island else 0.0 for island in self.islands]
        max_fit = max(island_bests) if max(island_bests) > 0 else 1.0

        self.ax_islands.clear()
        # Grid layout matching _build_island_graph's 'grid' indexing
        # (island i sits at row, col = divmod(i, cols)); row 0 drawn at the top.
        cols = max(1, int(n ** 0.5))
        rows = (n + cols - 1) // cols
        positions = []
        for i in range(n):
            row, col = divmod(i, cols)
            positions.append((col, -row))

        # Edges
        drawn_edges = set()
        for i in range(n):
            for j in self.island_neighbors[i]:
                edge = (min(i, j), max(i, j))
                if edge not in drawn_edges:
                    drawn_edges.add(edge)
                    x1, y1 = positions[i]
                    x2, y2 = positions[j]
                    self.ax_islands.plot([x1, x2], [y1, y2], color='gray', alpha=0.6, zorder=1, linewidth=2)

        # Nodes
        for i in range(n):
            x, y = positions[i]
            norm = island_bests[i] / max_fit
            color = (1 - norm, norm, 0.3)
            self.ax_islands.scatter(x, y, s=1200, color=[color], zorder=2, edgecolors='black', linewidths=1.5)
            label = f"I{i}\n{island_bests[i]:.1f}"
            self.ax_islands.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold', zorder=3)

        migr_gen = self.generationNumber - 1
        next_migr = self.MIGRATION_INTERVAL - (migr_gen % self.MIGRATION_INTERVAL)
        self.ax_islands.set_title(
            f"Graf populacji – gen. {migr_gen}  (migracja za {next_migr} gen.)",
            fontsize=9
        )
        self.ax_islands.set_xlim(-0.7, (cols - 1) + 0.7)
        self.ax_islands.set_ylim(-(rows - 1) - 0.7, 0.7)
        self.ax_islands.set_aspect('equal')
        self.ax_islands.axis('off')

        fullPath = self.output_path + "/islandTopology.png"
        if self._save_figure(self.fig_islands, fullPath) and self.DRAW_SIMULATION:
            img = cv2.imread(fullPath, cv2.IMREAD_ANYCOLOR)
            cv2.imshow("Graf wysep", img)

    # ==================== DIVERSITY METRICS ====================

    def _island_diversity(self, island: list) -> float:
        """Diversity of one island under the configured metric.
        Higher = more varied; ~0 = the island has converged to near-clones."""
        method = getattr(self, 'DIVERSITY_METRIC', 'genotypic')
        if method == 'phenotypic_jaccard':
            return self._diversity_phenotypic_jaccard(island)
        if method == 'unique_fraction':
            return self._diversity_unique_fraction(island)
        if method == 'fitness':
            return self._diversity_fitness(island)
        return self._diversity_genotypic(island)

    @staticmethod
    def _diversity_genotypic(island: list) -> float:
        """Mean per-locus expected pairwise mismatch  mean_i [2*p_i*(1-p_i)],
        where p_i = fraction of bit-1 among individuals that HAVE locus i.
        Range [0, 0.5]: 0 = identical genomes, 0.5 = every locus a 50/50 coin.

        Accumulated per locus (no [N, L] matrix), so it scales to the whole
        pooled population.  Empty genomes are skipped and each locus is only
        scored where >=2 individuals reach it, so a single short/empty DNA no
        longer collapses the whole measure to 0 (that was the spike artifact)."""
        dnas = [ind.dnaType.DNA for ind in island
                if ind.dnaType is not None and len(ind.dnaType.DNA) > 0]
        n = len(dnas)
        if n < 2:
            return 0.0
        lmax = max(len(d) for d in dnas)
        ones = np.zeros(lmax, dtype=np.float64)
        cnt = np.zeros(lmax, dtype=np.float64)
        for d in dnas:
            a = np.asarray(d, dtype=np.float64)
            l = a.shape[0]
            ones[:l] += a
            cnt[:l] += 1.0
        valid = cnt >= 2.0
        if not valid.any():
            return 0.0
        p = ones[valid] / cnt[valid]
        return float(np.mean(2.0 * p * (1.0 - p)))

    @staticmethod
    def _edge_signature(ind) -> frozenset:
        """Topology fingerprint: the set of edges the network actually evaluates —
        backward edges stored reversed as (lower id, higher id), self-loops
        dropped, duplicates collapsed, weights ignored.  So genes 9 -> 2 and
        2 -> 9 count as the same edge, matching Computation.  Requires ind.nn."""
        if not ind.nn:
            return frozenset()
        return frozenset((min(c['source'], c['target']), max(c['source'], c['target']))
                         for c in ind.nn if c['source'] != c['target'])

    def _diversity_phenotypic_jaccard(self, island: list, max_n: int = 600) -> float:
        """Mean pairwise Jaccard DISTANCE between networks' edge sets:
        1 - |A∩B| / |A∪B|, averaged over all pairs.  Range [0, 1]:
        0 = every network shares the same topology, 1 = no shared edges.
        O(n^2); subsamples to max_n individuals so the pooled/global call over
        the whole population stays feasible."""
        sigs = [self._edge_signature(ind) for ind in island]
        n = len(sigs)
        if n < 2:
            return 0.0
        if n > max_n:
            sigs = [sigs[i] for i in random.sample(range(n), max_n)]
            n = max_n
        total, pairs = 0.0, 0
        for i in range(n):
            for j in range(i + 1, n):
                a, b = sigs[i], sigs[j]
                union = len(a | b)
                jac = (len(a & b) / union) if union else 1.0
                total += 1.0 - jac
                pairs += 1
        return total / pairs if pairs else 0.0

    def _diversity_unique_fraction(self, island: list) -> float:
        """Fraction of distinct network topologies: n_unique / n.
        Range (0, 1]: 1/n = all identical, 1 = every individual unique."""
        if not island:
            return 0.0
        sigs = {self._edge_signature(ind) for ind in island}
        return len(sigs) / len(island)

    @staticmethod
    def _diversity_fitness(island: list) -> float:
        """Cheapest proxy: fitness spread, std / (mean + eps).
        Collapses toward 0 as the island converges in fitness."""
        f = [ind.fitness for ind in island]
        if len(f) < 2:
            return 0.0
        mean = float(np.mean(f))
        std = float(np.std(f))
        return std / (mean + 1e-9)

    def _draw_island_diversity(self):
        tick_spacing = max(1, int(len(self.generationsList) / 10 + 1))
        gens = self.generationsList

        self.ax_div.clear()
        island_colors = plt.cm.tab10(np.linspace(0, 1, self.NUM_SUBPOPULATIONS))
        for k in range(self.NUM_SUBPOPULATIONS):
            self.ax_div.plot(gens, self.island_diversity_lists[k],
                             color=island_colors[k], linewidth=1, linestyle='--',
                             alpha=0.7, label=f"Island {k}")

        # Mean per-island diversity (within-island), bold black.
        mean_div = [float(np.mean([self.island_diversity_lists[k][i]
                                   for k in range(self.NUM_SUBPOPULATIONS)]))
                    for i in range(len(gens))]
        self.ax_div.plot(gens, mean_div, color="black", linewidth=2.2, label="Mean per-island (within)")

        # Global diversity over the whole pooled population (within + between).
        # It sits above the mean; the GAP = between-island diversity.  If global
        # stays high while the mean is low, islands hold DIFFERENT solutions; if
        # both are low, the whole population has converged.
        self.ax_div.plot(gens, self.global_diversity_list, color="red",
                         linewidth=2.2, label="Global (pooled, within+between)")

        self.ax_div.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        self.ax_div.set_xlabel("Generacje")
        self.ax_div.set_ylabel(f"Różnorodność ({self.DIVERSITY_METRIC})")
        self.ax_div.set_ylim(bottom=0)
        self.ax_div.grid(True, alpha=0.3)

        lines, labels = self.ax_div.get_legend_handles_labels()
        self.fig_div.legend(lines, labels, loc="lower center",
                            bbox_to_anchor=(0.5, 1.02), ncol=5)
        self.fig_div.tight_layout()

        fullPath = self.output_path + "/diversityGraph.png"
        if self._save_figure(self.fig_div, fullPath) and self.DRAW_SIMULATION:
            img = cv2.imread(fullPath, cv2.IMREAD_ANYCOLOR)
            cv2.imshow("Wykres roznorodnosci", img)

    # ==================== SINGLE POPULATION ====================

    def runEvolution(self):

        self.createPopulation()

        while self.generationNumber <= self.MAX_GENERATIONS:

            self.population = deepcopy(self.new_population)

            # dekodowanie
            self.dekodeDNAToNetwork()

            #ewaluowanie
            self.evaluate_fitness()
            self.population.sort(key=Individual.sortKey, reverse=True)
            self.validateBest()

            for i in range(self.POPULATION_SIZE):
                self.population[i].new = False

            # elity
            for i in range(self.elite_amount):
                self.new_population[i] = deepcopy(self.population[i])

            # nowi (losowi)
            for i in range(self.new_amount):
                indv = Individual()
                indv.dnaType = self.DNA_init()
                self.new_population[self.elite_amount + i] = indv

            # selekcja rodzicow, krzyżowanie, dzieci, mutacje
            # Children start AFTER the elite + random blocks so the random
            # injection is not overwritten (was elite_amount * 2 — a bug when
            # new_amount != elite_amount).
            offset = self.elite_amount + self.new_amount
            for i in range(self.POPULATION_SIZE - offset):
                parent1 = self.selection(self.population)
                parent2 = self.selection(self.population)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                child.new = True
                child.fitness = 0
                child.testFitness = 0
                child.nn = None
                self.new_population[offset + i] = child


            print("Generacja nr.", self.generationNumber)
            self.generationNumber += 1
            self.save_fitness_results()              # track every generation
            self.save_population_checkpoint()        # full population DNA dump every N gens
            if self._should_graph():                 # render only periodically
                self.graph()

    def run(self) -> None:
        train_segs, train_cmap = self.buildTrack(self.DRAW_TRAIN_TRACK, "train")
        val_segs, val_cmap = self.buildTrack(self.DRAW_VAL_TRACK, "val")

        self.track = train_segs
        self.collision_map = train_cmap
        self.train_total_w = len(train_segs) * self.WINDOW_WIDTH

        self.valTrack = val_segs
        self.collision_val_map = val_cmap
        self.val_total_w = len(val_segs) * self.WINDOW_WIDTH

        # Fixed start: left-center of first segment, facing right
        self.STARTING_POINT = [self.WINDOW_WIDTH * 0.02, self.WINDOW_HEIGHT / 2]
        self.STARTING_ANGLE = 0

        self.STARTING_VAL_POINT = [self.WINDOW_WIDTH * 0.02, self.WINDOW_HEIGHT / 2]
        self.STARTING_VAL_ANGLE = 0

        if self.NUM_SUBPOPULATIONS > 1:
            self.island_neighbors = self._build_island_graph()

        self.saveParams()

        if self.NUM_SUBPOPULATIONS > 1:
            self.runEvolutionIslands()
        else:
            self.runEvolution()

    def save_fitness_results(self):

        fullPath = self.output_path + "/bestDNA.txt"
        with open(fullPath, 'w') as file:
            stringDNA = "".join(str(b) for b in self.population[0].dnaType.DNA)
            file.write(stringDNA)

        if self.best_indv_val_fitness > self.best_val_checkpoint_fitness:
            self.best_val_checkpoint_fitness = self.best_indv_val_fitness
            fullValPath = self.output_path + "/bestValDNA.txt"
            with open(fullValPath, 'w') as file:
                stringDNA = "".join(str(b) for b in self.population[0].dnaType.DNA)
                file.write(stringDNA)
            print("Nowy najlepszy wynik walidacyjny: " + format(self.best_val_checkpoint_fitness, ".3f") + " -> zapisano checkpoint")

        bestFitness = self.population[0].fitness

        fitness_values = [ind.fitness for ind in self.population]
        avg = np.mean(fitness_values)
        medianFitness = statistics.median(fitness_values)

        avg_cr = float(np.mean([ind.crossover_rate for ind in self.population]))
        avg_mr = float(np.mean([ind.mutation_rate for ind in self.population]))

        print("Najlepszy wynik generacji: " + format(bestFitness, ".3f"))
        print("Srednie przystosowanie: " + format(avg, ".3f"))
        print("Testowe przystosowanie: " + format(self.best_indv_val_fitness, ".3f"))
        print(f"Avg crossover_rate: {avg_cr:.4f}  Avg mutation_rate: {avg_mr:.6f}")
        print("")

        self.bestFitnessList.append(bestFitness)
        self.medianFitnessList.append(avg)
        self.testFitnessList.append(self.best_indv_val_fitness)
        self.avgCrossoverRateList.append(avg_cr)
        self.avgMutationRateList.append(avg_mr)
        self.generationsList.append(len(self.generationsList))

        fullDataPath = self.output_path + "/graphData.txt"
        with open(fullDataPath, 'w') as file:
            for i in range(len(self.bestFitnessList)):
                file.write(f"{self.generationsList[i]} {self.bestFitnessList[i]} "
                           f"{self.medianFitnessList[i]} {self.testFitnessList[i]} "
                           f"{self.avgCrossoverRateList[i]} {self.avgMutationRateList[i]}\n")


    def saveParams(self):
        fullDataPath = self.output_path + "/parameters.txt"
        with open(fullDataPath, 'w') as file:
            file.write("FPS  " + str(self.FPS) + "\n")
            file.write("WINDOW_WIDTH  " + str(self.WINDOW_WIDTH) + "\n")
            file.write("WINDOW_HEIGHT  " + str(self.WINDOW_HEIGHT) + "\n")
            file.write("MAX_GENERATIONS  " + str(self.MAX_GENERATIONS) + "\n")
            file.write("POPULATION_SIZE  " + str(self.POPULATION_SIZE) + "\n")
            file.write("CAR_X  " + str(self.STARTING_POINT[0]) + "\n")
            file.write("CAR_Y  " + str(self.STARTING_POINT[1]) + "\n")
            file.write("CAR_A  " + str(self.STARTING_ANGLE) + "\n")
            file.write("CAR_VAL_X  " + str(self.STARTING_VAL_POINT[0]) + "\n")
            file.write("CAR_VAL_Y  " + str(self.STARTING_VAL_POINT[1]) + "\n")
            file.write("CAR_VAL_A  " + str(self.STARTING_VAL_ANGLE) + "\n")
            file.write("NUM_SUBPOPULATIONS  " + str(self.NUM_SUBPOPULATIONS) + "\n")
            file.write("MIGRATION_INTERVAL  " + str(self.MIGRATION_INTERVAL) + "\n")
            file.write("MIGRATION_SIZE  " + str(self.MIGRATION_SIZE) + "\n")
            file.write("ISLAND_GRAPH  " + str(self._island_graph_spec) + "\n")
            file.write("CHECKPOINT_INTERVAL  " + str(self.CHECKPOINT_INTERVAL) + "\n")
            file.write("RESUME_CHECKPOINT  " + str(self._resume_checkpoint) + "\n")
            file.write("SENSOR_STEP  " + str(self.SENSOR_STEP) + "\n")
            if self.NUM_SUBPOPULATIONS > 1:
                file.write("ISLAND_NEIGHBORS  " + str(self.island_neighbors) + "\n")

    def graph(self):
        tick_spacing = int(len(self.generationsList) / 10 + 1)

        self.ax.clear()
        self.ax.plot(self.generationsList, self.bestFitnessList, label="Best fitness (train)", color=COLOR_BEST_TRAIN)
        self.ax.plot(self.generationsList, self.medianFitnessList, label="Avg fitness (train)", color=COLOR_AVG_TRAIN)
        self.ax.plot(self.generationsList, self.testFitnessList, label="Best fitness (val)", color=COLOR_VAL)
        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        self.ax.set_xlabel("Generacje")
        self.ax.set_ylabel("Przystosowanie")
        self.ax.grid(True, alpha=0.3)

        ax2 = self.ax.twinx()
        ax2.plot(self.generationsList, self.avgCrossoverRateList, label="Avg crossover rate", color=COLOR_CROSSOVER, linewidth=1.8)
        ax2.plot(self.generationsList, self.avgMutationRateList, label="Avg mutation rate", color=COLOR_MUTATION, linewidth=1.8)
        ax2.set_ylabel("Rate (0–1)", color="gray")
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='y', labelcolor='gray')

        lines1, labels1 = self.ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        self.fig.legend(lines1 + lines2, labels1 + labels2, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
        self.fig.tight_layout()

        fullPath = self.output_path + "/bestFitnessGraph.png"
        if self._save_figure(self.fig, fullPath) and self.DRAW_SIMULATION:
            img = cv2.imread(fullPath, cv2.IMREAD_ANYCOLOR)
            cv2.imshow("Wykres najlepszego wyniku", img)



