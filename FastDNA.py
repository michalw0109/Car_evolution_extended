"""
Compiled / vectorised genetic operators for the active single-chromosome path.

DNA stays a plain Python list of 0/1 ints everywhere (no storage change), so the
rest of the system is untouched.  The hot operators convert to a numpy uint8
array internally:

  * decode_markers  - Numba-compiled marker scan (no randomness -> exact).
  * mutate_*        - NumPy-vectorised bit flip / insert / delete (uses np.random,
                      reproducible via np.random.seed; replaces per-bit Python loops).

Gene layout (matches DNA_Decoder / DNA_Initialization):
  bits [0:32] = self-adaptive rates;  gene = 8-bit MARKER + 8 src + 8 tgt + 8 weight
"""

import numpy as np

RATE_BITS = 32
GENE_LEN = 32          # 8-bit marker + 24-bit payload

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:                      # pragma: no cover
    NUMBA_AVAILABLE = False
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if (not a or not callable(a[0])) else a[0]


@njit(cache=True)
def _decode_markers(dna, marker, nr_neurons):
    """Scan for marker-delimited genes; return (sources, targets, weights)."""
    n = dna.shape[0]
    ml = marker.shape[0]
    limit = n - (GENE_LEN - 1)              # mirror old loop bound: len - 31
    maxg = n // GENE_LEN + 1
    src = np.empty(maxg, dtype=np.int64)
    tgt = np.empty(maxg, dtype=np.int64)
    wt = np.empty(maxg, dtype=np.float64)
    cnt = 0
    i = RATE_BITS
    while i < limit:
        match = True
        for k in range(ml):
            if dna[i + k] != marker[k]:
                match = False
                break
        if match:
            idx = i + ml
            s = 0
            for k in range(8):
                s = (s << 1) | dna[idx + k]
            t = 0
            for k in range(8):
                t = (t << 1) | dna[idx + 8 + k]
            rw = 0
            for k in range(8):
                rw = (rw << 1) | dna[idx + 16 + k]
            src[cnt] = s % nr_neurons
            tgt[cnt] = t % nr_neurons
            wt[cnt] = (rw / 255.0) * 10.0 - 5.0
            cnt += 1
            i += GENE_LEN
        else:
            i += 1
    return src[:cnt], tgt[:cnt], wt[:cnt]


def decode_markers(dna_list, marker, nr_neurons):
    """Wrapper: list[int] DNA -> nn as a list of {source,target,weight} dicts."""
    dna = np.asarray(dna_list, dtype=np.uint8)
    mk = np.asarray(marker, dtype=np.uint8)
    if dna.shape[0] < RATE_BITS + GENE_LEN:
        return []
    src, tgt, wt = _decode_markers(dna, mk, nr_neurons)
    return [{'source': int(src[i]), 'target': int(tgt[i]), 'weight': float(wt[i])}
            for i in range(src.shape[0])]


# ─────────────────────────── mutation (NumPy) ───────────────────────────────
# Operate on a Python list, return a Python list.  Vectorised with np.random
# (seeded by np.random.seed in main, so runs stay reproducible).

# DNA is a numpy uint8 array.  Each mutation returns a NEW array (never mutates
# its input in place), so callers/parents are never aliased.

def mutate_bit_flip(dna, rate):
    a = np.array(dna, dtype=np.uint8)          # copy
    if rate > 0.0:
        flip = np.random.random(a.shape[0]) < rate
        a[flip] ^= 1
    return a


def mutate_insert(dna, rate):
    """After each bit, with prob `rate`, insert a random bit (matches the
    original: iterate bits, append bit, then maybe append a random bit)."""
    a = np.asarray(dna, dtype=np.uint8)
    if rate <= 0.0:
        return a
    n = a.shape[0]
    ins_mask = np.random.random(n) < rate                  # insert after position i?
    n_ins = int(ins_mask.sum())
    if n_ins == 0:
        return a
    rand_bits = (np.random.random(n_ins) < 0.5).astype(np.uint8)
    out = np.empty(n + n_ins, dtype=np.uint8)
    ins_idx = np.nonzero(ins_mask)[0]
    shift = np.zeros(n, dtype=np.int64)
    shift[ins_idx] = 1
    out_pos = np.arange(n) + np.cumsum(shift) - shift       # original bit landing slots
    out[out_pos] = a
    out[out_pos[ins_idx] + 1] = rand_bits                    # inserted bit right after
    return out


def mutate_delete(dna, rate):
    a = np.asarray(dna, dtype=np.uint8)
    if rate <= 0.0:
        return a
    keep = np.random.random(a.shape[0]) >= rate
    return a[keep]


# ───────────────────── gene-aware crossover (Numba) ─────────────────────────
# Mirrors Crossover.connection_based_markers.  Uses Numba's own PRNG, seeded
# once (lazily) from the main numpy seed so runs stay reproducible.

@njit(cache=True)
def _is_gene(dna, i, nlen, marker):
    if i > nlen - GENE_LEN:
        return False
    for k in range(marker.shape[0]):
        if dna[i + k] != marker[k]:
            return False
    return True


@njit(cache=True)
def _crossover_markers(dna1, dna2, marker):
    """Unbiased (0.5) single-index gene-aware crossover.
    Mirrors Crossover.connection_based_markers exactly, incl. random order."""
    n1 = dna1.shape[0]
    n2 = dna2.shape[0]
    out = np.empty(n1 + n2 + 128, dtype=np.uint8)
    o = 0

    pref = RATE_BITS
    if pref > n1:
        pref = n1
    if pref > n2:
        pref = n2
    for k in range(pref):
        out[o] = dna1[k] if np.random.random() < 0.5 else dna2[k]
        o += 1

    i = RATE_BITS
    while i < n1 and i < n2:
        g1 = _is_gene(dna1, i, n1, marker)
        g2 = _is_gene(dna2, i, n2, marker)
        if (not g1) and (not g2):
            out[o] = dna1[i] if np.random.random() < 0.5 else dna2[i]
            o += 1
            i += 1
        elif g1 and (not g2):
            if np.random.random() < 0.5:
                for k in range(GENE_LEN):
                    out[o] = dna1[i + k]
                    o += 1
                i += GENE_LEN
            else:
                out[o] = dna1[i] if np.random.random() < 0.5 else dna2[i]
                o += 1
                i += 1
        elif (not g1) and g2:
            if np.random.random() < 0.5:
                for k in range(GENE_LEN):
                    out[o] = dna2[i + k]
                    o += 1
                i += GENE_LEN
            else:
                out[o] = dna1[i] if np.random.random() < 0.5 else dna2[i]
                o += 1
                i += 1
        else:
            if np.random.random() < 0.5:
                for k in range(GENE_LEN):
                    out[o] = dna1[i + k]
                    o += 1
            else:
                for k in range(GENE_LEN):
                    out[o] = dna2[i + k]
                    o += 1
            i += GENE_LEN

    if np.random.random() < 0.5:
        while i < n1:
            out[o] = dna1[i]
            o += 1
            i += 1
    else:
        while i < n2:
            out[o] = dna2[i]
            o += 1
            i += 1

    return out[:o].copy()


@njit(cache=True)
def _nb_seed(s):
    np.random.seed(s)


_nb_seeded = [False]


def crossover_markers(dna1_list, dna2_list, marker):
    """Wrapper: list DNA -> child list DNA via the compiled gene-aware crossover."""
    if not _nb_seeded[0]:
        _nb_seed(int(np.random.randint(0, 2**31 - 1)))   # tie Numba PRNG to main seed
        _nb_seeded[0] = True
    d1 = np.asarray(dna1_list, dtype=np.uint8)
    d2 = np.asarray(dna2_list, dtype=np.uint8)
    mk = np.asarray(marker, dtype=np.uint8)
    return _crossover_markers(d1, d2, mk)              # numpy uint8 array
