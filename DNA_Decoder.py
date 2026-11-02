from collections import deque
from utils import bits_to_int
from DNA import *
from FastDNA import decode_markers

RATE_BITS = 32

def decode_rates(dna: Single_DNA_one_chromosome) -> tuple[float, float]:
    if len(dna.DNA) < RATE_BITS:
        return 0.5, 0.005
    crossover_rate = bits_to_int(dna.DNA[0:16]) / 65536.0
    mutation_rate  = bits_to_int(dna.DNA[16:32]) / 65536.0
    return crossover_rate, mutation_rate

class Decoder:
    def __init__(self):
        pass

    class decodes_single_DNA_one_chromosome:

        def __init__(self, inputs: list[int], outputs: list[int], marker=None):
            if marker is None:
                marker = [0, 1, 1, 1, 1, 1, 1, 1]

            self.INPUTS = inputs
            self.OUTPUTS = outputs
            self.MARKER = marker

            self.NR_OF_NEURONS  = self.OUTPUTS[len(self.OUTPUTS) - 1] + 1


        def connection_based(self, DNA: Single_DNA_one_chromosome):
            nn = []

            for idx in range(RATE_BITS, len(DNA.DNA) - 23, 24):
                source_bits = DNA.DNA[idx: idx + 8]
                target_bits = DNA.DNA[idx + 8: idx + 16]
                weight_bits = DNA.DNA[idx + 16: idx + 24]

                source_id = bits_to_int(source_bits) % self.NR_OF_NEURONS
                target_id = bits_to_int(target_bits) % self.NR_OF_NEURONS

                raw_weight = bits_to_int(weight_bits)
                weight = (raw_weight / 255.0) * 10.0 - 5.0

                nn.append({
                    'source': source_id,
                    'target': target_id,
                    'weight': weight
                })

            return nn

        def connection_based_markers(self, DNA: Single_DNA_one_chromosome):
            # Numba-compiled marker scan (FastDNA); identical output to the old
            # pure-Python loop, ~5-10x faster.  Accepts a list or numpy DNA.
            return decode_markers(DNA.DNA, self.MARKER, self.NR_OF_NEURONS)

        def matrix_connections(self, DNA: Single_DNA_one_chromosome):

            nn = []
            if len(DNA.DNA) < RATE_BITS + 256:
                return nn

            matrix_bits = DNA.DNA[RATE_BITS: RATE_BITS + 256]
            weights_bits = DNA.DNA[RATE_BITS + 256:]

            for i in range(16):
                for j in range(16):
                    if matrix_bits[i * 16 + j] == 1:
                        nn.append({
                            'source': i,
                            'target': j,
                            'weight': 0
                        })

            for idx in range(0, len(weights_bits) - 7, 8):
                weight_bits = weights_bits[idx:idx + 8]
                raw_weight = bits_to_int(weight_bits)
                weight = (raw_weight / 255.0) * 10 - 5

                if idx/8 >= len(nn):
                    return nn

                nn[int(idx/8)]["weight"] = weight

            return nn


        def triangular_matrix_connections(self, DNA: Single_DNA_one_chromosome):

            nn = []
            if len(DNA.DNA) < RATE_BITS + 120:
                return nn

            matrix_bits = DNA.DNA[RATE_BITS: RATE_BITS + 120]
            weights_bits = DNA.DNA[RATE_BITS + 120:]

            bit_idx = 0

            for i in range(16):
                for j in range(i + 1, 16):

                    bit = matrix_bits[bit_idx]
                    bit_idx += 1

                    if bit == 1:

                        nn.append({
                            'source': i,
                            'target': j,
                            'weight': 0
                        })

            for idx in range(0, len(weights_bits) - 7, 8):
                weight_bits = weights_bits[idx:idx + 8]

                raw_weight = bits_to_int(weight_bits)
                weight = (raw_weight / 255.0) * 10 - 5
                if idx / 8 >= len(nn):
                    return nn

                nn[int(idx/8)]["weight"] = weight

            return nn


        def fixed_topology(self, DNA: Single_DNA_one_chromosome):

            nn = []

            # input->hidden
            for i in range(len(self.INPUTS)):
                for j in range(len(self.INPUTS), self.NR_OF_NEURONS - len(self.OUTPUTS)):
                    nn.append({
                        'source': i,
                        'target': j,
                        'weight': 0
                    })

            # hidden->output
            for i in range(len(self.INPUTS), self.NR_OF_NEURONS - len(self.OUTPUTS)):
                for j in range(self.NR_OF_NEURONS - len(self.OUTPUTS), self.NR_OF_NEURONS):
                    nn.append({
                        'source': i,
                        'target': j,
                        'weight': 0
                    })

            bits_tail = DNA.DNA[RATE_BITS:]
            for idx in range(0, len(bits_tail) - 7, 8):
                weight_bits = bits_tail[idx:idx + 8]
                raw_weight = bits_to_int(weight_bits)
                weight = (raw_weight / 255.0) * 10 - 5

                if idx/8 >= len(nn):
                    return nn

                nn[int(idx/8)]["weight"] = weight

            return nn


        def grammar_matrix(self, DNA: Single_DNA_one_chromosome):

            nn = []

            if len(DNA.DNA) < RATE_BITS + 224:
                return nn

            def get_rule(bits, idx, size):
                return bits[idx * size:(idx + 1) * size]

            grammar_rules = DNA.DNA[RATE_BITS: RATE_BITS + 96]    # 8 * 12
            final_rules   = DNA.DNA[RATE_BITS + 96: RATE_BITS + 224]  # 8 * 16
            weights_bits  = DNA.DNA[RATE_BITS + 224:]

            matrix_symbols = [0, 0, 0]  # start symbol

            # --- grammar expansion (2 iterations) ---
            for _ in range(2):
                new_symbols = []

                for i in range(0, len(matrix_symbols), 3):
                    symbol = matrix_symbols[i:i + 3]
                    idx = bits_to_int(symbol)

                    rule = get_rule(grammar_rules, idx, 12)
                    new_symbols.extend(rule)

                matrix_symbols = new_symbols

            # now we have 16 symbols (4x4), each 3 bits

            matrix_bits = []

            # --- symbol → 16 bit expansion ---
            for i in range(0, len(matrix_symbols), 3):
                symbol = matrix_symbols[i:i + 3]
                idx = bits_to_int(symbol)

                rule = get_rule(final_rules, idx, 16)

                matrix_bits.extend(rule)


            for i in range(16):
                for j in range(16):
                    if matrix_bits[i * 16 + j] == 1:
                        nn.append({
                            'source': i,
                            'target': j,
                            'weight': 0
                        })

            for idx in range(0, len(weights_bits) - 7, 8):
                weight_bits = weights_bits[idx:idx + 8]
                raw_weight = bits_to_int(weight_bits)
                weight = (raw_weight / 255.0) * 10 - 5

                if idx / 8 >= len(nn):
                    return nn

                nn[int(idx / 8)]["weight"] = weight

            return nn




        def cellular_division(self, DNA: Single_DNA_one_chromosome):

            cells = [{'id': 0}]
            connections = []

            for idx in range(RATE_BITS, len(DNA.DNA) - 24, 25):

                opcode = DNA.DNA[idx]

                if opcode == 0:
                    cell = bits_to_int(DNA.DNA[idx + 1:idx + 9]) % self.NR_OF_NEURONS
                    cells.append({'id': cell})
                else:

                    cell = bits_to_int(DNA.DNA[idx + 1:idx + 9]) % len(cells)

                    target = bits_to_int(DNA.DNA[idx + 9:idx + 17]) % len(cells)

                    raw = bits_to_int(DNA.DNA[idx + 17:idx + 25])
                    weight = (raw / 255) * 10 - 5

                    connections.append({
                        'source': cells[cell]['id'],
                        'target': cells[target]['id'],
                        'weight': weight
                    })

            return connections


        def cellular_encoding(self, DNA: Single_DNA_one_chromosome, l: int = 1) -> list[dict]:
            """
            Gruau-style cell-rewriting developmental decoder.

            Chromosome layout (bits):
              [0..31]          – rate bits (crossover / mutation), skipped here
              [32 .. tree_end] – grammar-tree: 3 bits per node, depth-first pre-order
                                 000=E  001=1  010=M  011=W  100=R  101=P  110=S  111=B
              [tree_end ..]    – weight pool: 8 bits per connection (assigned in creation order)

            Operator arities and semantics
            ──────────────────────────────
            E / 1  (arity 0) – terminal: cell becomes a neuron (threshold 0 or 1)
            M / W  (arity 1) – timing delays: advance reading-head to child, re-queue cell
            R      (arity 1) – recurrent: decrement life; if life>0 restart from root,
                               else fall through to child subtree (base case)
            B      (arity 2) – branch: if life>1 use left subtree, else right subtree
            P      (arity 2) – parallel division: cell → 2 children that both inherit
                               ALL existing input/output links of the parent
            S      (arity 2) – sequential division: cell → 2 children; child-1 inherits
                               input links, child-2 inherits output links; a new link
                               child-1 → child-2 (weight from pool) is created

            Weights: all connection weights come from the 8-bit pool that follows the
            tree in the chromosome.  Pool bits are assigned in the order connections are
            created (each S step appends one new link).  The M operator no longer hard-
            codes -1; it acts as a pure timing delay so the GA can still evolve weight
            values through the pool.

            Pointer-cell initialisation (mirrors Gruau's paper, Figure 3):
            Before development starts, input and output neurons are pre-created as
            already-finalised cells and wired to the single initial developing cell:
              inputs[i] → init_cell → outputs[j]   for all i, j
            S divisions then naturally propagate input links to the "left" child and
            output links to the "right" child, anchoring the grown network between
            the fixed I/O neurons.  Developing cells that terminate are assigned neuron
            IDs from the hidden-neuron pool (all IDs that are neither inputs nor outputs),
            cycling if more cells terminate than there are hidden slots.

            Parameter l sets the initial value of the `life` register used by R/B.
            Default l=1 means R always takes the base-case branch (no recursion).
            """
            bits = DNA.DNA

            # ── operator table ─────────────────────────────────────────────────────
            _ARITY = {'E': 0, '1': 0, 'M': 1, 'W': 1, 'R': 1, 'P': 2, 'S': 2, 'B': 2}
            _OPS   = ['E', '1', 'M', 'W', 'R', 'P', 'S', 'B']   # index == 3-bit code

            # ── Step 1: parse grammar-tree depth-first, 3 bits per node ────────────
            pos = RATE_BITS

            def _parse(depth=0):
                nonlocal pos
                if pos + 3 > len(bits) or depth > 200:
                    return {'op': 'E', 'ch': []}
                code = bits_to_int(bits[pos: pos + 3])
                pos += 3
                op = _OPS[code]
                return {'op': op, 'ch': [_parse(depth + 1) for _ in range(_ARITY[op])]}

            root         = _parse()
            weight_start = pos          # first bit of the weight pool

            # ── Step 2: developmental process ──────────────────────────────────────
            # Each cell dict: {rh, life, is_neuron, threshold, nid}
            # nid = final neuron ID used in the output connection list.
            # Pre-created I/O cells have their nid fixed; developing cells receive
            # a hidden-pool nid when they terminate.
            # links: [{f: cell_id, t: cell_id}]  – global directed-link list
            _next_id    = [0]
            cells       = {}
            links       = []

            _io_set     = set(self.INPUTS) | set(self.OUTPUTS)
            _hidden_ids = [nid for nid in range(self.NR_OF_NEURONS) if nid not in _io_set]
            _next_h     = [0]   # cycles through _hidden_ids when developing cells terminate

            MAX_CELLS = 500
            MAX_STEPS = 5000

            def _fixed_neuron(neuron_id):
                """Pre-create an already-finalised input or output neuron."""
                cid = _next_id[0]
                _next_id[0] += 1
                cells[cid] = {'rh': None, 'life': l, 'is_neuron': True,
                              'threshold': 0, 'nid': neuron_id}
                return cid

            def _new_cell(rh, life):
                cid = _next_id[0]
                _next_id[0] += 1
                cells[cid] = {'rh': rh, 'life': life, 'is_neuron': False,
                              'threshold': 0, 'nid': None}
                return cid

            def _terminate(cid):
                """Mark a developing cell as a neuron and assign it a hidden nid."""
                c = cells[cid]
                c['is_neuron'] = True
                if c['nid'] is None:
                    if _hidden_ids:
                        c['nid'] = _hidden_ids[_next_h[0] % len(_hidden_ids)]
                        _next_h[0] += 1
                    else:
                        c['nid'] = 0    # fallback: no hidden slots defined

            # ── Pointer-cell pre-wiring ─────────────────────────────────────────────
            input_cids  = [_fixed_neuron(nid) for nid in self.INPUTS]
            output_cids = [_fixed_neuron(nid) for nid in self.OUTPUTS]
            init_cid    = _new_cell(root, l)

            for ic in input_cids:
                links.append({'f': ic, 't': init_cid})
            for oc in output_cids:
                links.append({'f': init_cid, 't': oc})

            fifo  = deque([init_cid])
            steps = 0

            while fifo and steps < MAX_STEPS:
                steps += 1
                cid = fifo.popleft()

                if cid not in cells or cells[cid]['is_neuron']:
                    continue

                cell = cells[cid]
                op   = cell['rh']['op']
                ch   = cell['rh']['ch']

                # ── terminal operators ──────────────────────────────────────────────
                if op in ('E', '1'):
                    cell['threshold'] = 1 if op == '1' else 0
                    _terminate(cid)

                # ── timing / value operators (arity 1) ─────────────────────────────
                elif op in ('M', 'W'):
                    cell['rh'] = ch[0]
                    fifo.append(cid)

                # ── recurrent operator ──────────────────────────────────────────────
                elif op == 'R':
                    cell['life'] -= 1
                    cell['rh']   = root if cell['life'] > 0 else ch[0]
                    fifo.append(cid)

                # ── branching operator ──────────────────────────────────────────────
                elif op == 'B':
                    cell['rh'] = ch[0] if cell['life'] > 1 else ch[1]
                    fifo.append(cid)

                # ── parallel division ───────────────────────────────────────────────
                elif op == 'P':
                    if len(cells) >= MAX_CELLS:
                        _terminate(cid)
                        continue
                    id1 = _new_cell(ch[0], cell['life'])
                    id2 = _new_cell(ch[1], cell['life'])
                    # Both children inherit ALL of the parent's links (fan-out).
                    # A link that touched cid is duplicated for id1 and id2.
                    new_links = []
                    for lk in links:
                        f, t   = lk['f'], lk['t']
                        f_list = [id1, id2] if f == cid else [f]
                        t_list = [id1, id2] if t == cid else [t]
                        for ff in f_list:
                            for tt in t_list:
                                new_links.append({'f': ff, 't': tt})
                    links[:] = new_links
                    del cells[cid]
                    fifo.append(id1)    # left subtree child enters FIFO first
                    fifo.append(id2)

                # ── sequential division ─────────────────────────────────────────────
                elif op == 'S':
                    if len(cells) >= MAX_CELLS:
                        _terminate(cid)
                        continue
                    id1 = _new_cell(ch[0], cell['life'])
                    id2 = _new_cell(ch[1], cell['life'])
                    # id1 takes input links; id2 takes output links.
                    new_links = []
                    for lk in links:
                        f, t = lk['f'], lk['t']
                        if t == cid:
                            new_links.append({'f': f, 't': id1})   # incoming → id1
                        elif f == cid:
                            new_links.append({'f': id2, 't': t})   # outgoing from id2
                        else:
                            new_links.append(lk)
                    new_links.append({'f': id1, 't': id2})          # new id1→id2 link
                    links[:] = new_links
                    del cells[cid]
                    fifo.append(id1)
                    fifo.append(id2)

            # Force any still-reading cells into neurons so their links are usable.
            for cid in list(cells.keys()):
                if not cells[cid]['is_neuron']:
                    _terminate(cid)

            # ── Step 3: collect connections between neurons ─────────────────────────
            neuron_set = {cid for cid, c in cells.items() if c['is_neuron']}
            raw_conns  = [
                lk for lk in links
                if lk['f'] in neuron_set
                and lk['t'] in neuron_set
                and cells[lk['f']]['nid'] != cells[lk['t']]['nid']   # drop nid self-loops
            ]

            # ── Step 4: assign 8-bit weights from the pool after the tree ───────────
            nn = []
            for i, lk in enumerate(raw_conns):
                wb = weight_start + i * 8
                raw    = bits_to_int(bits[wb: wb + 8]) if wb + 8 <= len(bits) else 127
                weight = (raw / 255.0) * 10.0 - 5.0
                nn.append({
                    'source': cells[lk['f']]['nid'],
                    'target': cells[lk['t']]['nid'],
                    'weight': weight
                })

            return nn


    class decodes_double_DNA_one_chromosome:

        def __init__(self, inputs: list[int], outputs: list[int], marker=None):
            if marker is None:
                marker = [0, 1, 1, 1, 1, 1, 1, 1]

            self.INPUTS = inputs
            self.OUTPUTS = outputs
            self.MARKER = marker

            self.NR_OF_NEURONS  = self.OUTPUTS[len(self.OUTPUTS) - 1] + 1

            # delegate single-strand logic to the existing single decoder
            self._single = Decoder.decodes_single_DNA_one_chromosome(inputs, outputs, marker)

        # ─────────────────────────────────────────────────────────────────────
        # Helper: wrap a plain bit list in a temporary Single_DNA object
        # ─────────────────────────────────────────────────────────────────────
        @staticmethod
        def _wrap(bits: list[int]) -> Single_DNA_one_chromosome:
            tmp = Single_DNA_one_chromosome()
            tmp.DNA = bits
            return tmp

        # =====================================================================
        # GROUP 1 – combine-based decoders (7)
        # Each method uses the matching gene-aware combine so the strand
        # merging respects the same structure the decoder expects.
        # =====================================================================

        def connection_based(self, dna: Double_DNA_one_chromosome):
            return self._single.connection_based(self._wrap(dna.combine('connection_based')))

        def connection_based_markers(self, dna: Double_DNA_one_chromosome):
            return self._single.connection_based_markers(self._wrap(dna.combine('connection_based_markers', marker=self.MARKER)))

        def matrix_connections(self, dna: Double_DNA_one_chromosome):
            return self._single.matrix_connections(self._wrap(dna.combine('matrix_connections')))

        def triangular_matrix_connections(self, dna: Double_DNA_one_chromosome):
            return self._single.triangular_matrix_connections(self._wrap(dna.combine('triangular_matrix_connections')))

        def fixed_topology(self, dna: Double_DNA_one_chromosome):
            return self._single.fixed_topology(self._wrap(dna.combine('fixed_topology')))

        def grammar_matrix(self, dna: Double_DNA_one_chromosome):
            return self._single.grammar_matrix(self._wrap(dna.combine('grammar_matrix')))

        def cellular_division(self, dna: Double_DNA_one_chromosome):
            return self._single.cellular_division(self._wrap(dna.combine('cellular_division')))

        # =====================================================================
        # GROUP 2 – dual-decode decoders (7)
        # Each strand is decoded independently and the resulting connection
        # lists are concatenated.  Duplicate connections are kept (they add
        # weight contribution when the network processes them).
        # =====================================================================

        def connection_based_dual(self, dna: Double_DNA_one_chromosome):
            return (self._single.connection_based(self._wrap(list(dna.DNAa))) +
                    self._single.connection_based(self._wrap(list(dna.DNAb))))

        def connection_based_markers_dual(self, dna: Double_DNA_one_chromosome):
            return (self._single.connection_based_markers(self._wrap(list(dna.DNAa))) +
                    self._single.connection_based_markers(self._wrap(list(dna.DNAb))))

        def matrix_connections_dual(self, dna: Double_DNA_one_chromosome):
            return (self._single.matrix_connections(self._wrap(list(dna.DNAa))) +
                    self._single.matrix_connections(self._wrap(list(dna.DNAb))))

        def triangular_matrix_connections_dual(self, dna: Double_DNA_one_chromosome):
            return (self._single.triangular_matrix_connections(self._wrap(list(dna.DNAa))) +
                    self._single.triangular_matrix_connections(self._wrap(list(dna.DNAb))))

        def fixed_topology_dual(self, dna: Double_DNA_one_chromosome):
            return (self._single.fixed_topology(self._wrap(list(dna.DNAa))) +
                    self._single.fixed_topology(self._wrap(list(dna.DNAb))))

        def grammar_matrix_dual(self, dna: Double_DNA_one_chromosome):
            return (self._single.grammar_matrix(self._wrap(list(dna.DNAa))) +
                    self._single.grammar_matrix(self._wrap(list(dna.DNAb))))

        def cellular_division_dual(self, dna: Double_DNA_one_chromosome):
            return (self._single.cellular_division(self._wrap(list(dna.DNAa))) +
                    self._single.cellular_division(self._wrap(list(dna.DNAb))))

        def cellular_encoding(self, dna: Double_DNA_one_chromosome, l: int = 1):
            return self._single.cellular_encoding(self._wrap(dna.combine('cellular_encoding')), l)

        def cellular_encoding_dual(self, dna: Double_DNA_one_chromosome, l: int = 1):
            return (self._single.cellular_encoding(self._wrap(list(dna.DNAa)), l) +
                    self._single.cellular_encoding(self._wrap(list(dna.DNAb)), l))
