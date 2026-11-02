"""
Vectorised, whole-population car simulation (NumPy).

Replaces the per-Car Python loop in EvolutionEngine.evaluate_fitness with a
batched simulation: all cars are stepped together as NumPy arrays.  Designed to
be BIT-EXACT with Car.update / EvolutionEngine.evaluate_fitness, with one
deliberate exception:

  * Sensors gain a maximum range (MAX_SENSOR_DIST).  The original ray-march was
    unbounded (it marched until it hit a wall or left the map); vectorising
    requires a fixed number of samples per ray.  Rays that find no wall within
    MAX_SENSOR_DIST report that capped distance instead of the true (farther)
    one.  Set the cap generously and the difference is negligible.

Keep the physics constants below in sync with Car.py.
"""

import numpy as np

# --- physics constants (mirror Car.py) ---
MAX_SPEED = 4.0
MINIMUM_SPEED = 0.0
TURN_SPEED = 3.0
ACCELERATION = 2.0
SENSOR_ANGLES = 25.0          # degrees between adjacent rays
NO_PROGRESS_FRAMES = 12000    # die if no fitness progress within this window
STALL_FRAMES = 1200           # die if not moving within this window
MIN_FITNESS_PROGRESS = 0.1    # min fitness units to count as "progress"
COMPLETION_BONUS = 50000.0
FPS_FOR_BONUS = 120.0


def build_weight_tensors(nn_list, n_neurons):
    """Pack a list of decoded networks into a dense [N, n_neurons, n_neurons]
    weight tensor.  Forward edges (source < target) are kept, backward edges
    (source > target) are reversed into target -> source, self-loops are
    dropped — matching Computation.connection_based_sort_feed_forward — and
    duplicate edges are summed (also matching the original += accumulation)."""
    n = len(nn_list)
    W = np.zeros((n, n_neurons, n_neurons), dtype=np.float64)
    for i, nn in enumerate(nn_list):
        if not nn:
            continue
        for c in nn:
            s, t = c['source'], c['target']
            if s < t:
                W[i, s, t] += c['weight']
            elif s > t:
                W[i, t, s] += c['weight']
    return W


def _norm_pair(a, b):
    """Vectorised version of Car.computeReaction.normalize_pair over arrays."""
    with np.errstate(divide='ignore', invalid='ignore'):
        both_pos = (a > 0) & (b > 0)
        both_neg = (a < 0) & (b < 0)

        s_pp = a + b
        pp_a = a / s_pp
        pp_b = b / s_pp

        # both negative: negate, swap, normalise  ->  (b/(a+b), a/(a+b))
        nn_a = b / s_pp
        nn_b = a / s_pp

        # otherwise: shift so the smaller becomes 0, then normalise
        amin = np.minimum(a, b)
        oa = a - amin
        ob = b - amin
        s_o = oa + ob
        safe = s_o != 0
        denom = np.where(safe, s_o, 1.0)
        o_a = np.where(safe, oa / denom, 0.0)
        o_b = np.where(safe, ob / denom, 0.0)

        out_a = np.where(both_pos, pp_a, np.where(both_neg, nn_a, o_a))
        out_b = np.where(both_pos, pp_b, np.where(both_neg, nn_b, o_b))
    return out_a, out_b


def simulate(W, input_ids, output_ids, collision_map,
             start_xy, start_angle, total_w, total_h, seg_w,
             car_w, car_h, max_sensor_dist, max_frames=300000, sensor_step=1.0):
    """Run the whole population to completion and return fitness[N].

    W            : [N, K, K] dense weight tensor (K = number of neurons)
    input_ids    : neuron ids for the 5 sensors + speed (length 6)
    output_ids   : neuron ids for turn_l, turn_r, accel, brake (length 4)
    collision_map: bool array [map_w, map_h]; True = wall
    """
    SENSOR_STEP = float(sensor_step)       # from main.py via EvolutionEngine
    N = W.shape[0]
    K = W.shape[1]
    map_w, map_h = collision_map.shape[0], collision_map.shape[1]

    input_ids = np.asarray(input_ids)
    output_ids = np.asarray(output_ids)
    is_input = np.zeros(K, dtype=bool)
    is_input[input_ids] = True

    ray_off = (np.array([-2, -1, 0, 1, 2], dtype=np.float64)) * SENSOR_ANGLES  # [5]
    M = max(1, int(max_sensor_dist // SENSOR_STEP))
    sample_d = SENSOR_STEP * np.arange(M, dtype=np.float64)                     # [M]

    hw, hh = car_w / 2.0, car_h / 2.0
    corners = [(0.0, 0.0), (-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]

    startX = float(start_xy[0])
    pos = np.tile(np.array(start_xy, dtype=np.float64), (N, 1))   # [N,2]
    angle = np.full(N, float(start_angle), dtype=np.float64)
    speed = np.zeros(N, dtype=np.float64)
    alive = np.ones(N, dtype=bool)
    fitness = np.zeros(N, dtype=np.float64)
    maxX = np.full(N, startX, dtype=np.float64)
    xReset = np.full(N, startX, dtype=np.float64)
    lastProg = np.zeros(N, dtype=np.float64)
    lastFast = np.zeros(N, dtype=np.float64)

    completion_thresh = total_w - seg_w * 0.1

    t = 0
    while t < max_frames:
        idx = np.nonzero(alive)[0]
        if idx.size == 0:
            break
        n = idx.size

        # ---------- sensors ----------
        sa = angle[idx][:, None] + ray_off[None, :]            # [n,5] degrees
        rad = np.radians(360.0 - sa)
        cos_r = np.cos(rad)
        sin_r = np.sin(rad)
        xs = pos[idx, 0][:, None, None] + cos_r[:, :, None] * sample_d[None, None, :]  # [n,5,M]
        ys = pos[idx, 1][:, None, None] + sin_r[:, :, None] * sample_d[None, None, :]
        ix = xs.astype(np.intp)
        iy = ys.astype(np.intp)
        oob = (ix < 0) | (ix >= map_w) | (iy < 0) | (iy >= map_h)
        np.clip(ix, 0, map_w - 1, out=ix)
        np.clip(iy, 0, map_h - 1, out=iy)
        hit = oob | collision_map[ix, iy]                      # [n,5,M]
        any_hit = hit.any(axis=2)
        first = np.argmax(hit, axis=2)                         # [n,5], 0 if none
        first = np.where(any_hit, first, M)                    # cap if no hit
        sensor_norm = (SENSOR_STEP * first) / seg_w            # [n,5]

        # ---------- neural net (dense, exact feed-forward) ----------
        Wn = W[idx]                                            # [n,K,K]
        val = np.zeros((n, K), dtype=np.float64)
        for i in range(5):
            val[:, input_ids[i]] = sensor_norm[:, i]
        val[:, input_ids[5]] = speed[idx] / MAX_SPEED          # normalised to 0..1
        for tt in range(1, K):
            acc = (val[:, :tt] * Wn[:, :tt, tt]).sum(axis=1)
            if is_input[tt]:
                val[:, tt] = val[:, tt] + acc                  # inputs: accumulate, no tanh
            else:
                val[:, tt] = np.tanh(acc)
        out = val[:, output_ids]                               # [n,4]

        # ---------- controls ----------
        tl, tr = _norm_pair(out[:, 0], out[:, 1])
        ac, br = _norm_pair(out[:, 2], out[:, 3])
        angle[idx] = angle[idx] + TURN_SPEED * (tl - tr)
        sp = np.minimum(speed[idx] + ACCELERATION * ac, MAX_SPEED)
        sp = np.maximum(sp - ACCELERATION * br, MINIMUM_SPEED)
        speed[idx] = sp

        # ---------- position ----------
        radp = np.radians(360.0 - angle[idx])
        cos_p = np.cos(radp)
        sin_p = np.sin(radp)
        pos[idx, 0] = pos[idx, 0] + cos_p * sp
        pos[idx, 1] = pos[idx, 1] + sin_p * sp

        # ---------- collision (out-of-bounds + 5 body points) ----------
        cx = pos[idx, 0]
        cy = pos[idx, 1]
        dead = (cx < 0) | (cx >= total_w) | (cy < 0) | (cy >= total_h)
        for dx, dy in corners:
            px = cx + dx * cos_p - dy * sin_p
            py = cy + dx * sin_p + dy * cos_p
            pix = px.astype(np.intp)
            piy = py.astype(np.intp)
            p_oob = (pix < 0) | (pix >= map_w) | (piy < 0) | (piy >= map_h)
            np.clip(pix, 0, map_w - 1, out=pix)
            np.clip(piy, 0, map_h - 1, out=piy)
            dead = dead | p_oob | collision_map[pix, piy]

        alive[idx[dead]] = False
        surv = idx[~dead]

        # ---------- fitness (only survivors of this frame) ----------
        if surv.size:
            newX = pos[surv, 0]
            improved = newX > maxX[surv]
            maxX[surv] = np.where(improved, newX, maxX[surv])
            gained = (maxX[surv] - xReset[surv]) / seg_w * 10.0
            reset_mask = improved & (gained >= MIN_FITNESS_PROGRESS)
            lastProg[surv] = np.where(reset_mask, t, lastProg[surv])
            xReset[surv] = np.where(reset_mask, maxX[surv], xReset[surv])

            noprog = (t - lastProg[surv]) > NO_PROGRESS_FRAMES
            prog = (maxX[surv] - startX) / seg_w * 10.0
            completed = maxX[surv] >= completion_thresh
            elapsed = t / FPS_FOR_BONUS
            fit_new = np.where(completed, prog + COMPLETION_BONUS / (elapsed + 1.0), prog)
            # no-progress death returns BEFORE updating fitness in the original
            fitness[surv] = np.where(noprog, fitness[surv], fit_new)

            fast = speed[surv] > 0.001
            lastFast[surv] = np.where(noprog, lastFast[surv],
                                      np.where(fast, t, lastFast[surv]))
            stall = (~noprog) & ((t - lastFast[surv]) > STALL_FRAMES)
            dead2 = noprog | ((~noprog) & completed) | stall
            alive[surv[dead2]] = False

        t += 1

    return fitness