import numpy as np
from DNA import DNA

class Individual:
    def __init__(self):
        self.dnaType = None
        self.nn = None
        self.fitness = 0
        self.testFitness = 0
        self.new = True
        self.crossover_rate = 0
        self.mutation_rate = 0

    def sortKey(self):
        return self.fitness

    @staticmethod
    def _copy_dna(d):
        # numpy array -> .copy() (slicing an array is a VIEW, not a copy); list -> slice
        if isinstance(d, np.ndarray):
            return d.copy()
        return d[:]

    def clone(self):
        """Cheap independent copy for reproduction: copies scalar attrs and the
        DNA (numpy array .copy(), ~100x faster than deepcopy), but NOT the decoded
        network (nn) — that is recomputed every generation.  Use instead of
        copy.deepcopy in the hot GA loop."""
        new = Individual()
        dt = self.dnaType
        if dt is not None:
            nd = type(dt)()
            if getattr(dt, 'DNA', None) is not None:
                nd.DNA = self._copy_dna(dt.DNA)
            if hasattr(dt, 'DNAa'):
                nd.DNAa = self._copy_dna(dt.DNAa)
                nd.DNAb = self._copy_dna(dt.DNAb)
            new.dnaType = nd
        new.fitness = self.fitness
        new.testFitness = self.testFitness
        new.new = self.new
        new.crossover_rate = self.crossover_rate
        new.mutation_rate = self.mutation_rate
        return new
