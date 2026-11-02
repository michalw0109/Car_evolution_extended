import random
from DNA import DNA, Single_DNA_one_chromosome
from Individual import Individual
from utils import *
from copy import deepcopy
from FastDNA import mutate_bit_flip, mutate_insert, mutate_delete

RATE_BITS = 32


class MutationCombiner:
    def __init__(self, mutations: list):
        self.mutations = mutations

    def mutate(self, child: Individual):
        for mutation in self.mutations:
            child = mutation(child)
        return child


class Mutation:
    def __init__(self):
        pass

    def no_mutation(self, child: Individual):
        return child

    class mutate_single_DNA_one_chromosome:
        def __init__(self, _MARKER=None):
            if _MARKER is None:
                _MARKER = [0, 1, 1, 1, 1, 1, 1, 1]
            self.MARKER = _MARKER

        # ─────────────────────────────────────────────────────────────────────
        # Simple mutations — operate on the full DNA (including rate bits).
        # ─────────────────────────────────────────────────────────────────────

        def random_bit_flip(self, child: Individual):
            # NumPy-vectorised (FastDNA); same per-bit flip rule, np.random RNG.
            child.dnaType.DNA = mutate_bit_flip(child.dnaType.DNA, child.mutation_rate)
            return child

        def random_insert(self, child: Individual):
            child.dnaType.DNA = mutate_insert(child.dnaType.DNA, child.mutation_rate)
            return child

        def random_delete(self, child: Individual):
            child.dnaType.DNA = mutate_delete(child.dnaType.DNA, child.mutation_rate)
            return child

        # =========================================================================
        # Smart structure-aware mutations — preserve rate prefix (RATE_BITS).
        # Each method fires with probability individual.mutation_rate.
        # =========================================================================

        def connection_based(self, individual: Individual):
            ind = deepcopy(individual)
            if random.random() >= ind.mutation_rate:
                return ind

            GENE = 24
            dna = ind.dnaType.DNA
            tail = list(dna[RATE_BITS:])
            n_genes = len(tail) // GENE
            if n_genes == 0:
                return ind

            op = random.choice(['duplicate', 'delete'])

            if op == 'duplicate':
                src = random.randint(0, n_genes - 1) * GENE
                gene = tail[src:src + GENE]
                dst = random.randint(0, len(tail) // GENE) * GENE
                tail = tail[:dst] + gene + tail[dst:]
            elif op == 'delete' and n_genes > 1:
                src = random.randint(0, n_genes - 1) * GENE
                tail = tail[:src] + tail[src + GENE:]

            ind.dnaType.DNA = dna[:RATE_BITS] + tail
            return ind

        def connection_based_markers(self, individual: Individual):
            ind = deepcopy(individual)
            if random.random() >= ind.mutation_rate:
                return ind

            MARKER = self.MARKER
            ML = len(MARKER)
            GENE = ML + 24

            dna = list(ind.dnaType.DNA)

            gene_pos = []
            i = RATE_BITS
            while i <= len(dna) - GENE:
                if dna[i:i + ML] == MARKER:
                    gene_pos.append(i)
                    i += GENE
                else:
                    i += 1

            op = random.choice(['duplicate', 'delete'])

            if op == 'duplicate' and gene_pos:
                pos = random.choice(gene_pos)
                gene = dna[pos:pos + GENE]
                dst = random.randint(RATE_BITS, len(dna))
                dna = dna[:dst] + gene + dna[dst:]
            elif op == 'delete' and len(gene_pos) > 1:
                pos = random.choice(gene_pos)
                dna = dna[:pos] + dna[pos + GENE:]

            ind.dnaType.DNA = dna
            return ind

        def matrix_connections(self, individual: Individual):
            ind = deepcopy(individual)
            if random.random() >= ind.mutation_rate:
                return ind

            dna = list(ind.dnaType.DNA)
            if len(dna) < RATE_BITS + 256:
                return ind

            MATRIX_END = RATE_BITS + 256
            op = random.choice(['weight_duplicate', 'weight_delete'])

            if op == 'weight_duplicate' and len(dna) > MATRIX_END + 8:
                tail = dna[MATRIX_END:]
                nw = len(tail) // 8
                if nw:
                    i = random.randint(0, nw - 1) * 8
                    block = tail[i:i + 8]
                    dst = random.randint(0, nw) * 8
                    new_tail = tail[:dst] + block + tail[dst:]
                    dna = dna[:MATRIX_END] + new_tail

            elif op == 'weight_delete' and len(dna) > MATRIX_END + 8:
                tail = dna[MATRIX_END:]
                nw = len(tail) // 8
                if nw > 1:
                    i = random.randint(0, nw - 1) * 8
                    dna = dna[:MATRIX_END] + tail[:i] + tail[i + 8:]

            ind.dnaType.DNA = dna
            return ind

        def triangular_matrix_connections(self, individual: Individual):
            ind = deepcopy(individual)
            if random.random() >= ind.mutation_rate:
                return ind

            dna = list(ind.dnaType.DNA)
            if len(dna) < RATE_BITS + 120:
                return ind

            MATRIX_END = RATE_BITS + 120
            op = random.choice(['weight_duplicate', 'weight_delete'])

            if op == 'weight_duplicate' and len(dna) > MATRIX_END + 8:
                tail = dna[MATRIX_END:]
                nw = len(tail) // 8
                if nw:
                    i = random.randint(0, nw - 1) * 8
                    block = tail[i:i + 8]
                    dst = random.randint(0, nw) * 8
                    new_tail = tail[:dst] + block + tail[dst:]
                    dna = dna[:MATRIX_END] + new_tail

            elif op == 'weight_delete' and len(dna) > MATRIX_END + 8:
                tail = dna[MATRIX_END:]
                nw = len(tail) // 8
                if nw > 1:
                    i = random.randint(0, nw - 1) * 8
                    dna = dna[:MATRIX_END] + tail[:i] + tail[i + 8:]

            ind.dnaType.DNA = dna
            return ind

        def fixed_topology(self, individual: Individual):
            ind = deepcopy(individual)
            if random.random() >= ind.mutation_rate:
                return ind

            dna = list(ind.dnaType.DNA)
            tail = dna[RATE_BITS:]
            nw = len(tail) // 8
            if nw == 0:
                return ind

            op = random.choice(['weight_copy', 'weight_delete'])
            i = random.randint(0, nw - 1) * 8

            if op == 'weight_copy' and nw > 1:
                src = random.randint(0, nw - 1) * 8
                tail[i:i + 8] = tail[src:src + 8]
            elif op == 'weight_delete' and len(tail) > 128:
                tail = tail[:i] + tail[i + 8:]

            ind.dnaType.DNA = dna[:RATE_BITS] + tail
            return ind

        def grammar_matrix(self, individual: Individual):
            ind = deepcopy(individual)
            if random.random() >= ind.mutation_rate:
                return ind

            dna = list(ind.dnaType.DNA)
            if len(dna) < RATE_BITS + 224:
                return ind

            GR = RATE_BITS
            FR = RATE_BITS + 96
            WT = RATE_BITS + 224

            op = random.choice([
                'swap_grammar_rules', 'duplicate_grammar_rule',
                'swap_final_rules', 'duplicate_final_rule',
                'weight_duplicate', 'weight_delete'
            ])

            if op == 'swap_grammar_rules':
                a, b = random.sample(range(8), 2)
                sa, sb = GR + a * 12, GR + b * 12
                dna[sa:sa + 12], dna[sb:sb + 12] = dna[sb:sb + 12], dna[sa:sa + 12]
            elif op == 'duplicate_grammar_rule':
                src, dst = random.sample(range(8), 2)
                dna[GR + dst * 12:GR + (dst + 1) * 12] = dna[GR + src * 12:GR + (src + 1) * 12]
            elif op == 'swap_final_rules':
                a, b = random.sample(range(8), 2)
                sa, sb = FR + a * 16, FR + b * 16
                dna[sa:sa + 16], dna[sb:sb + 16] = dna[sb:sb + 16], dna[sa:sa + 16]
            elif op == 'duplicate_final_rule':
                src, dst = random.sample(range(8), 2)
                dna[FR + dst * 16:FR + (dst + 1) * 16] = dna[FR + src * 16:FR + (src + 1) * 16]
            elif op == 'weight_duplicate' and len(dna) > WT + 8:
                tail = dna[WT:]
                nw = len(tail) // 8
                if nw:
                    i = random.randint(0, nw - 1) * 8
                    block = tail[i:i + 8]
                    dst = random.randint(0, nw) * 8
                    new_tail = tail[:dst] + block + tail[dst:]
                    dna = dna[:WT] + new_tail
            elif op == 'weight_delete' and len(dna) > WT + 8:
                tail = dna[WT:]
                nw = len(tail) // 8
                if nw > 1:
                    i = random.randint(0, nw - 1) * 8
                    dna = dna[:WT] + tail[:i] + tail[i + 8:]

            ind.dnaType.DNA = dna
            return ind

        def cellular_division(self, individual: Individual):
            ind = deepcopy(individual)
            if random.random() >= ind.mutation_rate:
                return ind

            INSTR = 25
            dna = list(ind.dnaType.DNA)
            tail = dna[RATE_BITS:]
            n = len(tail) // INSTR
            if n == 0:
                return ind

            op = random.choice(['duplicate_instruction', 'delete_instruction'])

            if op == 'duplicate_instruction':
                src = random.randint(0, n - 1) * INSTR
                instr = tail[src:src + INSTR]
                dst = random.randint(0, len(tail) // INSTR) * INSTR
                tail = tail[:dst] + instr + tail[dst:]
            elif op == 'delete_instruction' and n > 1:
                src = random.randint(0, n - 1) * INSTR
                tail = tail[:src] + tail[src + INSTR:]

            ind.dnaType.DNA = dna[:RATE_BITS] + tail
            return ind


    class mutate_double_DNA_one_chromosome:

        def __init__(self, _MARKER=None):
            if _MARKER is None:
                _MARKER = [0, 1, 1, 1, 1, 1, 1, 1]
            self.MARKER = _MARKER

        def random_bit_flip(self, child: Individual):
            ind = deepcopy(child)
            rate = child.mutation_rate
            dna = ind.dnaType
            dna.DNAa = [1 - b if random.random() < rate else b for b in dna.DNAa]
            dna.DNAb = [1 - b if random.random() < rate else b for b in dna.DNAb]
            return ind

        def random_insert(self, child: Individual):
            ind = deepcopy(child)
            rate = child.mutation_rate
            dna = ind.dnaType
            new_a, new_b = [], []
            for ba, bb in zip(dna.DNAa, dna.DNAb):
                new_a.append(ba); new_b.append(bb)
                if random.random() < rate:
                    bit = random.randint(0, 1)
                    new_a.append(bit); new_b.append(bit)
            dna.DNAa, dna.DNAb = new_a, new_b
            return ind

        def random_delete(self, child: Individual):
            ind = deepcopy(child)
            rate = child.mutation_rate
            dna = ind.dnaType
            pairs = [(a, b) for a, b in zip(dna.DNAa, dna.DNAb) if random.random() >= rate]
            dna.DNAa = [p[0] for p in pairs]
            dna.DNAb = [p[1] for p in pairs]
            return ind

        @staticmethod
        def _mutate_strand(single_mut, strand: list, mutation_rate: float) -> list:
            tmp = Individual()
            tmp.dnaType = Single_DNA_one_chromosome()
            tmp.dnaType.DNA = list(strand)
            tmp.fitness = 0
            tmp.mutation_rate = mutation_rate
            return single_mut(tmp).dnaType.DNA

        def _single_mut(self):
            return Mutation.mutate_single_DNA_one_chromosome(self.MARKER)

        def connection_based(self, child: Individual):
            ind = deepcopy(child)
            s = self._single_mut()
            ind.dnaType.DNAa = self._mutate_strand(s.connection_based, ind.dnaType.DNAa, child.mutation_rate)
            ind.dnaType.DNAb = self._mutate_strand(s.connection_based, ind.dnaType.DNAb, child.mutation_rate)
            return ind

        def connection_based_markers(self, child: Individual):
            ind = deepcopy(child)
            s = self._single_mut()
            ind.dnaType.DNAa = self._mutate_strand(s.connection_based_markers, ind.dnaType.DNAa, child.mutation_rate)
            ind.dnaType.DNAb = self._mutate_strand(s.connection_based_markers, ind.dnaType.DNAb, child.mutation_rate)
            return ind

        def matrix_connections(self, child: Individual):
            ind = deepcopy(child)
            s = self._single_mut()
            ind.dnaType.DNAa = self._mutate_strand(s.matrix_connections, ind.dnaType.DNAa, child.mutation_rate)
            ind.dnaType.DNAb = self._mutate_strand(s.matrix_connections, ind.dnaType.DNAb, child.mutation_rate)
            return ind

        def triangular_matrix_connections(self, child: Individual):
            ind = deepcopy(child)
            s = self._single_mut()
            ind.dnaType.DNAa = self._mutate_strand(s.triangular_matrix_connections, ind.dnaType.DNAa, child.mutation_rate)
            ind.dnaType.DNAb = self._mutate_strand(s.triangular_matrix_connections, ind.dnaType.DNAb, child.mutation_rate)
            return ind

        def fixed_topology(self, child: Individual):
            ind = deepcopy(child)
            s = self._single_mut()
            ind.dnaType.DNAa = self._mutate_strand(s.fixed_topology, ind.dnaType.DNAa, child.mutation_rate)
            ind.dnaType.DNAb = self._mutate_strand(s.fixed_topology, ind.dnaType.DNAb, child.mutation_rate)
            return ind

        def grammar_matrix(self, child: Individual):
            ind = deepcopy(child)
            s = self._single_mut()
            ind.dnaType.DNAa = self._mutate_strand(s.grammar_matrix, ind.dnaType.DNAa, child.mutation_rate)
            ind.dnaType.DNAb = self._mutate_strand(s.grammar_matrix, ind.dnaType.DNAb, child.mutation_rate)
            return ind

        def cellular_division(self, child: Individual):
            ind = deepcopy(child)
            s = self._single_mut()
            ind.dnaType.DNAa = self._mutate_strand(s.cellular_division, ind.dnaType.DNAa, child.mutation_rate)
            ind.dnaType.DNAb = self._mutate_strand(s.cellular_division, ind.dnaType.DNAb, child.mutation_rate)
            return ind
