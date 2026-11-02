from Individual import Individual
from copy import deepcopy
import random
from utils import connectionToDNA, generateRandomDna
from DNA_Decoder import decode_rates
from FastDNA import crossover_markers

RATE_BITS = 32

class Crossover:
    def __init__(self):
        pass

    def no_crossover(self, parent1: Individual, parent2: Individual):
        child = parent1.clone()
        return child

    class crossover_single_DNA_one_chromosome:
        def __init__(self, _MARKER=None):
            self.MARKER = _MARKER

        # All methods read (parent1.crossover_rate + parent2.crossover_rate) / 2 as probability,
        # then decode rates into the child before returning so attributes stay in sync with DNA.

        #### SIMPLE CUT BASED ####

        def single_cut(self, parent1: Individual, parent2: Individual):
            child = parent1.clone()
            rate = (parent1.crossover_rate + parent2.crossover_rate) / 2
            if random.random() < rate:
                dna1 = parent1.dnaType.DNA
                dna2 = parent2.dnaType.DNA
                if min(len(dna1) - 1, len(dna2) - 1) > 0:
                    cut = random.randint(0, min(len(dna1) - 1, len(dna2) - 1))
                    child.dnaType.DNA = dna1[:cut] + dna2[cut:]
            child.crossover_rate, child.mutation_rate = decode_rates(child.dnaType)
            return child

        def multi_cut(self, parent1: Individual, parent2: Individual):
            child = parent1.clone()
            rate = (parent1.crossover_rate + parent2.crossover_rate) / 2
            if random.random() < rate:
                dna1 = parent1.dnaType.DNA
                dna2 = parent2.dnaType.DNA
                if min(len(dna1) - 1, len(dna2) - 1) > 0:
                    cut1 = random.randint(0, min(len(dna1) - 1, len(dna2) - 1))
                    cut2 = random.randint(0, min(len(dna1) - 1, len(dna2) - 1))
                    if cut1 > cut2:
                        cut1, cut2 = cut2, cut1
                    child.dnaType.DNA = dna1[:cut1] + dna2[cut1:cut2] + dna1[cut2:]
            child.crossover_rate, child.mutation_rate = decode_rates(child.dnaType)
            return child

        def random_bit(self, parent1: Individual, parent2: Individual):
            child = parent1.clone()
            rate = (parent1.crossover_rate + parent2.crossover_rate) / 2
            if random.random() < rate:
                dna1 = parent1.dnaType.DNA
                dna2 = parent2.dnaType.DNA
                min_len = min(len(dna1), len(dna2))
                new_dna = [dna1[i] if random.random() < 0.5 else dna2[i] for i in range(min_len)]
                new_dna.extend(dna1[min_len:] if len(dna1) > len(dna2) else dna2[min_len:])
                child.dnaType.DNA = new_dna
            child.crossover_rate, child.mutation_rate = decode_rates(child.dnaType)
            return child

        def single_cut_weighted(self, parent1: Individual, parent2: Individual):
            child = parent1.clone()
            rate = (parent1.crossover_rate + parent2.crossover_rate) / 2
            if random.random() < rate:
                dna1 = parent1.dnaType.DNA
                dna2 = parent2.dnaType.DNA
                min_len = min(len(dna1), len(dna2))
                if min_len > 0:
                    f1, f2 = parent1.fitness, parent2.fitness
                    total = f1 + f2
                    w1 = f1 / total if total > 0 else 0.5
                    cut = int(random.triangular(0, min_len, w1 * min_len))
                    child.dnaType.DNA = dna1[:cut] + dna2[cut:]
            child.crossover_rate, child.mutation_rate = decode_rates(child.dnaType)
            return child

        def multi_cut_weighted(self, parent1: Individual, parent2: Individual):
            child = parent1.clone()
            rate = (parent1.crossover_rate + parent2.crossover_rate) / 2
            if random.random() < rate:
                dna1 = parent1.dnaType.DNA
                dna2 = parent2.dnaType.DNA
                min_len = min(len(dna1), len(dna2))
                if min_len > 0:
                    f1, f2 = parent1.fitness, parent2.fitness
                    total = f1 + f2
                    w2 = f2 / total if total > 0 else 0.5
                    seg = int(random.triangular(0, min_len, w2 * min_len))
                    cut1 = random.randint(0, max(0, min_len - seg))
                    cut2 = min(cut1 + seg, min_len)
                    child.dnaType.DNA = dna1[:cut1] + dna2[cut1:cut2] + dna1[cut2:]
            child.crossover_rate, child.mutation_rate = decode_rates(child.dnaType)
            return child

        def random_bit_weighted(self, parent1: Individual, parent2: Individual):
            child = parent1.clone()
            rate = (parent1.crossover_rate + parent2.crossover_rate) / 2
            if random.random() < rate:
                dna1 = parent1.dnaType.DNA
                dna2 = parent2.dnaType.DNA
                f1, f2 = parent1.fitness, parent2.fitness
                total = f1 + f2
                w1 = f1 / total if total > 0 else 0.5
                min_len = min(len(dna1), len(dna2))
                new_dna = [dna1[i] if random.random() < w1 else dna2[i] for i in range(min_len)]
                new_dna.extend(dna1[min_len:] if f1 >= f2 else dna2[min_len:])
                child.dnaType.DNA = new_dna
            child.crossover_rate, child.mutation_rate = decode_rates(child.dnaType)
            return child

        #### smart gene-aware crossovers ####

        def _w(self, f1, f2):
            total = f1 + f2
            return f1 / total if total > 0 else 0.5

        def connection_based(self, parent1: Individual, parent2: Individual):
            child = parent1.clone()
            rate = (parent1.crossover_rate + parent2.crossover_rate) / 2
            if random.random() < rate:
                dna1 = parent1.dnaType.DNA
                dna2 = parent2.dnaType.DNA
                f1, f2 = parent1.fitness, parent2.fitness
                w1 = self._w(f1, f2)
                min_len = min(len(dna1), len(dna2))
                new_dna = []
                for i in range(min(RATE_BITS, min_len)):
                    new_dna.append(dna1[i] if random.random() < w1 else dna2[i])
                for idx in range(RATE_BITS, min_len - 23, 24):
                    new_dna.extend(dna1[idx:idx+24] if random.random() < w1 else dna2[idx:idx+24])
                processed = RATE_BITS + ((max(min_len - RATE_BITS, 0) // 24) * 24)
                new_dna.extend(dna1[processed:] if f1 >= f2 else dna2[processed:])
                child.dnaType.DNA = new_dna
            child.crossover_rate, child.mutation_rate = decode_rates(child.dnaType)
            return child

        def connection_based_markers(self, parent1: Individual, parent2: Individual):
            # Numba-compiled (FastDNA.crossover_markers); identical algorithm to the
            # unbiased single-index gene-aware crossover, ~10x faster.
            child = parent1.clone()
            rate = (parent1.crossover_rate + parent2.crossover_rate) / 2
            if random.random() < rate:
                child.dnaType.DNA = crossover_markers(
                    parent1.dnaType.DNA, parent2.dnaType.DNA, self.MARKER)
            child.crossover_rate, child.mutation_rate = decode_rates(child.dnaType)
            return child

        def matrix_connections(self, parent1: Individual, parent2: Individual):
            child = parent1.clone()
            rate = (parent1.crossover_rate + parent2.crossover_rate) / 2
            if random.random() < rate:
                dna1 = parent1.dnaType.DNA
                dna2 = parent2.dnaType.DNA
                if len(dna1) >= RATE_BITS + 256 and len(dna2) >= RATE_BITS + 256:
                    f1, f2 = parent1.fitness, parent2.fitness
                    w1 = self._w(f1, f2)
                    new_dna = [dna1[i] if random.random() < w1 else dna2[i] for i in range(RATE_BITS)]
                    new_dna.extend(dna1[RATE_BITS:RATE_BITS+256] if random.random() < w1 else dna2[RATE_BITS:RATE_BITS+256])
                    min_tail = min(len(dna1), len(dna2))
                    for idx in range(RATE_BITS + 256, min_tail - 7, 8):
                        new_dna.extend(dna1[idx:idx+8] if random.random() < w1 else dna2[idx:idx+8])
                    processed = RATE_BITS + 256 + ((max(min_tail - RATE_BITS - 256, 0)) // 8) * 8
                    new_dna.extend(dna1[processed:] if f1 >= f2 else dna2[processed:])
                    child.dnaType.DNA = new_dna
            child.crossover_rate, child.mutation_rate = decode_rates(child.dnaType)
            return child

        def triangular_matrix_connections(self, parent1: Individual, parent2: Individual):
            child = parent1.clone()
            rate = (parent1.crossover_rate + parent2.crossover_rate) / 2
            if random.random() < rate:
                dna1 = parent1.dnaType.DNA
                dna2 = parent2.dnaType.DNA
                if len(dna1) >= RATE_BITS + 120 and len(dna2) >= RATE_BITS + 120:
                    f1, f2 = parent1.fitness, parent2.fitness
                    w1 = self._w(f1, f2)
                    new_dna = [dna1[i] if random.random() < w1 else dna2[i] for i in range(RATE_BITS)]
                    new_dna.extend(dna1[RATE_BITS:RATE_BITS+120] if random.random() < w1 else dna2[RATE_BITS:RATE_BITS+120])
                    min_tail = min(len(dna1), len(dna2))
                    for idx in range(RATE_BITS + 120, min_tail - 7, 8):
                        new_dna.extend(dna1[idx:idx+8] if random.random() < w1 else dna2[idx:idx+8])
                    processed = RATE_BITS + 120 + ((max(min_tail - RATE_BITS - 120, 0)) // 8) * 8
                    new_dna.extend(dna1[processed:] if f1 >= f2 else dna2[processed:])
                    child.dnaType.DNA = new_dna
            child.crossover_rate, child.mutation_rate = decode_rates(child.dnaType)
            return child

        def fixed_topology(self, parent1: Individual, parent2: Individual):
            child = parent1.clone()
            rate = (parent1.crossover_rate + parent2.crossover_rate) / 2
            if random.random() < rate:
                dna1 = parent1.dnaType.DNA
                dna2 = parent2.dnaType.DNA
                f1, f2 = parent1.fitness, parent2.fitness
                w1 = self._w(f1, f2)
                min_len = min(len(dna1), len(dna2))
                new_dna = [dna1[i] if random.random() < w1 else dna2[i] for i in range(min(RATE_BITS, min_len))]
                for idx in range(RATE_BITS, min_len - 7, 8):
                    new_dna.extend(dna1[idx:idx+8] if random.random() < w1 else dna2[idx:idx+8])
                processed = RATE_BITS + ((max(min_len - RATE_BITS, 0) // 8) * 8)
                new_dna.extend(dna1[processed:] if f1 >= f2 else dna2[processed:])
                child.dnaType.DNA = new_dna
            child.crossover_rate, child.mutation_rate = decode_rates(child.dnaType)
            return child

        def grammar_matrix(self, parent1: Individual, parent2: Individual):
            child = parent1.clone()
            rate = (parent1.crossover_rate + parent2.crossover_rate) / 2
            if random.random() < rate:
                dna1 = parent1.dnaType.DNA
                dna2 = parent2.dnaType.DNA
                if len(dna1) >= RATE_BITS + 224 and len(dna2) >= RATE_BITS + 224:
                    f1, f2 = parent1.fitness, parent2.fitness
                    w1 = self._w(f1, f2)
                    new_dna = [dna1[i] if random.random() < w1 else dna2[i] for i in range(RATE_BITS)]
                    for i in range(8):
                        s = RATE_BITS + i * 12
                        new_dna.extend(dna1[s:s+12] if random.random() < w1 else dna2[s:s+12])
                    for i in range(8):
                        s = RATE_BITS + 96 + i * 16
                        new_dna.extend(dna1[s:s+16] if random.random() < w1 else dna2[s:s+16])
                    min_tail = min(len(dna1), len(dna2))
                    for idx in range(RATE_BITS + 224, min_tail - 7, 8):
                        new_dna.extend(dna1[idx:idx+8] if random.random() < w1 else dna2[idx:idx+8])
                    processed = RATE_BITS + 224 + ((max(min_tail - RATE_BITS - 224, 0)) // 8) * 8
                    new_dna.extend(dna1[processed:] if f1 >= f2 else dna2[processed:])
                    child.dnaType.DNA = new_dna
            child.crossover_rate, child.mutation_rate = decode_rates(child.dnaType)
            return child

        def cellular_division(self, parent1: Individual, parent2: Individual):
            child = parent1.clone()
            rate = (parent1.crossover_rate + parent2.crossover_rate) / 2
            if random.random() < rate:
                dna1 = parent1.dnaType.DNA
                dna2 = parent2.dnaType.DNA
                f1, f2 = parent1.fitness, parent2.fitness
                w1 = self._w(f1, f2)
                min_len = min(len(dna1), len(dna2))
                new_dna = [dna1[i] if random.random() < w1 else dna2[i] for i in range(min(RATE_BITS, min_len))]
                for idx in range(RATE_BITS, min_len - 24, 25):
                    new_dna.extend(dna1[idx:idx+25] if random.random() < w1 else dna2[idx:idx+25])
                processed = RATE_BITS + ((max(min_len - RATE_BITS, 0) // 25) * 25)
                new_dna.extend(dna1[processed:] if f1 >= f2 else dna2[processed:])
                child.dnaType.DNA = new_dna
            child.crossover_rate, child.mutation_rate = decode_rates(child.dnaType)
            return child

    #---------------------------
    class crossover_double_DNA_one_chromosome:
        def __init__(self, _MARKER=None):
            if _MARKER is None:
                _MARKER = [0, 1, 1, 1, 1, 1, 1, 1]
            self.MARKER = _MARKER

        def random_strand_pick(self, parent1: Individual, parent2: Individual):
            child = parent1.clone()
            rate = (parent1.crossover_rate + parent2.crossover_rate) / 2
            if random.random() < rate:
                child.dnaType.DNAa = parent1.dnaType.combine('random')
                child.dnaType.DNAb = parent2.dnaType.combine('random')
            return child

        def connection_based(self, parent1: Individual, parent2: Individual):
            child = parent1.clone()
            rate = (parent1.crossover_rate + parent2.crossover_rate) / 2
            if random.random() < rate:
                child.dnaType.DNAa = parent1.dnaType.combine('connection_based')
                child.dnaType.DNAb = parent2.dnaType.combine('connection_based')
            return child

        def connection_based_markers(self, parent1: Individual, parent2: Individual):
            child = parent1.clone()
            rate = (parent1.crossover_rate + parent2.crossover_rate) / 2
            if random.random() < rate:
                child.dnaType.DNAa = parent1.dnaType.combine('connection_based_markers', marker=self.MARKER)
                child.dnaType.DNAb = parent2.dnaType.combine('connection_based_markers', marker=self.MARKER)
            return child

        def matrix_connections(self, parent1: Individual, parent2: Individual):
            child = parent1.clone()
            rate = (parent1.crossover_rate + parent2.crossover_rate) / 2
            if random.random() < rate:
                child.dnaType.DNAa = parent1.dnaType.combine('matrix_connections')
                child.dnaType.DNAb = parent2.dnaType.combine('matrix_connections')
            return child

        def triangular_matrix_connections(self, parent1: Individual, parent2: Individual):
            child = parent1.clone()
            rate = (parent1.crossover_rate + parent2.crossover_rate) / 2
            if random.random() < rate:
                child.dnaType.DNAa = parent1.dnaType.combine('triangular_matrix_connections')
                child.dnaType.DNAb = parent2.dnaType.combine('triangular_matrix_connections')
            return child

        def fixed_topology(self, parent1: Individual, parent2: Individual):
            child = parent1.clone()
            rate = (parent1.crossover_rate + parent2.crossover_rate) / 2
            if random.random() < rate:
                child.dnaType.DNAa = parent1.dnaType.combine('fixed_topology')
                child.dnaType.DNAb = parent2.dnaType.combine('fixed_topology')
            return child

        def grammar_matrix(self, parent1: Individual, parent2: Individual):
            child = parent1.clone()
            rate = (parent1.crossover_rate + parent2.crossover_rate) / 2
            if random.random() < rate:
                child.dnaType.DNAa = parent1.dnaType.combine('grammar_matrix')
                child.dnaType.DNAb = parent2.dnaType.combine('grammar_matrix')
            return child

        def cellular_division(self, parent1: Individual, parent2: Individual):
            child = parent1.clone()
            rate = (parent1.crossover_rate + parent2.crossover_rate) / 2
            if random.random() < rate:
                child.dnaType.DNAa = parent1.dnaType.combine('cellular_division')
                child.dnaType.DNAb = parent2.dnaType.combine('cellular_division')
            return child
