"""
Numba-compiled, per-car-parallel population simulation.

Each car is an independent sequential simulation; `prange` runs them across CPU
cores, and each car's inner loop is JIT-compiled to native code.  Unlike the
batched-NumPy version this marches each sensor ray *until it hits* (exactly like
the original Car.updateSensors) — no fixed sample count, no max-range cap — so it
is faithful to the original and has no dense-sampling waste.

Physics constants mirror Car.py / BatchedSim.py.  The neural-net feed-forward is
the same dense low->high id propagation as Computation.connection_based_sort_feed_forward.
"""

import math
import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:                      # pragma: no cover
    NUMBA_AVAILABLE = False
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if (not a or not callable(a[0])) else a[0]
    prange = range


@njit(cache=True, inline='always')
def _norm_pair(a, b):
    if a > 0.0 and b > 0.0:
        s = a + b
        return a / s, b / s
    elif a < 0.0 and b < 0.0:
        na = -b          # negate + swap  ->  (b/(a+b), a/(a+b))
        nb = -a
        s = na + nb
        return na / s, nb / s
    else:
        amin = a if a < b else b
        aa = a - amin
        bb = b - amin
        s = aa + bb
        if s != 0.0:
            return aa / s, bb / s
        return 0.0, 0.0


@njit(parallel=True, cache=True, fastmath=False)
def simulate_numba(W, input_ids, output_ids, is_input, col_map,
                   start_x, start_y, start_angle,
                   total_w, total_h, seg_w, car_w, car_h, sensor_step, max_frames):
    N = W.shape[0]
    K = W.shape[1]
    map_w = col_map.shape[0]
    map_h = col_map.shape[1]

    SENSOR_STEP = sensor_step  # from main.py, same value Car.py gets
    MAX_SPEED = 4.0
    MIN_SPEED = 0.0
    TURN_SPEED = 3.0
    ACCEL = 2.0
    SENSOR_ANGLES = 25.0
    hw = car_w / 2.0
    hh = car_h / 2.0

    out0 = output_ids[0]
    out1 = output_ids[1]
    out2 = output_ids[2]
    out3 = output_ids[3]

    fitness = np.zeros(N)

    for i in prange(N):
        px = start_x
        py = start_y
        angle = start_angle
        speed = 0.0
        maxX = start_x
        xReset = start_x
        lastProg = 0.0
        lastFast = 0.0
        fit = 0.0
        alive = True

        val = np.zeros(K)
        t = 0
        while alive and t < max_frames:
            # ---- sensors: march each ray until wall / out of bounds ----
            for r in range(5):
                sang = angle + (r - 2) * SENSOR_ANGLES
                rad = math.radians(360.0 - sang)
                xst = math.cos(rad) * SENSOR_STEP
                yst = math.sin(rad) * SENSOR_STEP
                sx = px
                sy = py
                length = 0.0
                while True:
                    ix = int(sx)
                    iy = int(sy)
                    if ix < 0 or ix >= map_w or iy < 0 or iy >= map_h:
                        break
                    if col_map[ix, iy]:
                        break
                    sx += xst
                    sy += yst
                    length += SENSOR_STEP
                val[input_ids[r]] = length / seg_w
            val[input_ids[5]] = speed / MAX_SPEED          # normalised to 0..1

            # ---- neural net: dense feed-forward, neuron id order ----
            for tt in range(1, K):
                acc = 0.0
                for s in range(tt):
                    acc += val[s] * W[i, s, tt]
                if is_input[tt]:
                    val[tt] = val[tt] + acc
                else:
                    val[tt] = math.tanh(acc)

            # ---- controls ----
            tl, tr = _norm_pair(val[out0], val[out1])
            ac, br = _norm_pair(val[out2], val[out3])
            angle = angle + TURN_SPEED * (tl - tr)
            speed = speed + ACCEL * ac
            if speed > MAX_SPEED:
                speed = MAX_SPEED
            speed = speed - ACCEL * br
            if speed < MIN_SPEED:
                speed = MIN_SPEED

            # ---- position ----
            radp = math.radians(360.0 - angle)
            cosp = math.cos(radp)
            sinp = math.sin(radp)
            px += cosp * speed
            py += sinp * speed

            # ---- collision: bounds + 5 body points ----
            dead = (px < 0.0) or (px >= total_w) or (py < 0.0) or (py >= total_h)
            if not dead:
                for ci in range(5):
                    if ci == 0:
                        dx = 0.0; dy = 0.0
                    elif ci == 1:
                        dx = -hw; dy = -hh
                    elif ci == 2:
                        dx = hw; dy = -hh
                    elif ci == 3:
                        dx = hw; dy = hh
                    else:
                        dx = -hw; dy = hh
                    qx = px + dx * cosp - dy * sinp
                    qy = py + dx * sinp + dy * cosp
                    iqx = int(qx)
                    iqy = int(qy)
                    if iqx < 0 or iqx >= map_w or iqy < 0 or iqy >= map_h:
                        dead = True
                        break
                    if col_map[iqx, iqy]:
                        dead = True
                        break
            if dead:
                alive = False
                break

            # ---- fitness / death timers (only if not collision-dead) ----
            if px > maxX:
                maxX = px
                gained = (maxX - xReset) / seg_w * 10.0
                if gained >= 0.1:
                    lastProg = t
                    xReset = maxX
            if t - lastProg > 12000:
                alive = False
                break
            prog = (maxX - start_x) / seg_w * 10.0
            if maxX >= total_w - seg_w * 0.1:
                alive = False
                fit = prog + 50000.0 / (t / 120.0 + 1.0)
                break
            fit = prog
            if speed > 0.001:
                lastFast = t
            if t - lastFast > 1200:
                alive = False
                break

            t += 1

        fitness[i] = fit

    return fitness
