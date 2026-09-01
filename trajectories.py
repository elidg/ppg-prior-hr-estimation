import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def wrap(f):
            return f
        return wrap


@njit
def _viterbi_best(score, lo, hi):
    """
    Best path in a trellis with row-local transition windows [lo[r], hi[r]).
    score: shape (n_hr, n_q), additive score per node.
    Returns (best_score, best_path_rows).
    """
    n_hr, n_q = score.shape

    dp = np.empty((n_hr, n_q), dtype=np.float64)
    parent = np.full((n_hr, n_q - 1), -1, dtype=np.int32)

    dp[:, n_q - 1] = score[:, n_q - 1]

    dq = np.empty(n_hr, dtype=np.int32)

    for c in range(n_q - 2, -1, -1):
        prev = dp[:, c + 1]

        best = np.full(n_hr, -np.inf, dtype=np.float64)
        arg = np.full(n_hr, -1, dtype=np.int32)

        head = 0
        tail = 0
        add = 0

        # Monotone deque over the moving windows [lo[r], hi[r])
        for r in range(n_hr):
            h = hi[r]
            while add < h:
                v = prev[add]
                while tail > head and prev[dq[tail - 1]] <= v:
                    tail -= 1
                dq[tail] = add
                tail += 1
                add += 1

            l = lo[r]
            while tail > head and dq[head] < l:
                head += 1

            if tail > head:
                j = dq[head]
                best[r] = prev[j]
                arg[r] = j

        for r in range(n_hr):
            s = score[r, c]
            if np.isfinite(s) and np.isfinite(best[r]):
                dp[r, c] = s + best[r]
                parent[r, c] = arg[r]
            else:
                dp[r, c] = -np.inf
                parent[r, c] = -1

    start = -1
    best0 = -np.inf
    for r in range(n_hr):
        if dp[r, 0] > best0:
            best0 = dp[r, 0]
            start = r

    if start < 0 or not np.isfinite(best0):
        return -np.inf, np.empty(0, dtype=np.int32)

    path = np.empty(n_q, dtype=np.int32)
    path[0] = start
    for c in range(n_q - 1):
        nr = parent[path[c], c]
        if nr < 0:
            return -np.inf, np.empty(0, dtype=np.int32)
        path[c + 1] = nr

    return best0, path


def iter_trajectories(
    Z,
    hr_grid,
    epsilon,
    local_eps,
    *,
    tube_scale=0.75,
    penalty=10.0,
    hard_suppression=False,
    max_internal_candidates=10000,
):
    """
    Fast practical generator of diverse trajectories.

    Parameters
    ----------
    Z : (n_hr, n_q) ndarray
        Probability matrix.
    hr_grid : (n_hr,) ndarray
        Sorted HR values.
    epsilon : float
        Minimum RMSE in HR between yielded trajectories.
    local_eps : float or None
        Max absolute HR change between consecutive columns.
    tube_scale : float, default 0.75
        Suppression tube radius = tube_scale * epsilon.
        Larger => more diversity, fewer trajectories.
    penalty : float, default 10.0
        Log-score penalty applied inside the suppression tube.
        10 means multiplicative factor exp(-10), i.e. very strong.
    hard_suppression : bool, default False
        If True, suppress tube by setting score to -inf instead of subtracting penalty.
    max_internal_candidates : int
        Safety limit to avoid infinite loops if many candidates fail RMSE.

    Yields
    ------
    hr_path : (n_q,) ndarray
    log_score_original : float
        Score on the original Z, not on the suppressed working copy.

    Notes
    -----
    - Very fast in practice.
    - Exact wrt local_eps.
    - Exact final RMSE filtering wrt epsilon.
    - NOT exact global ranking by original probability.
    """

    Z = np.asarray(Z, dtype=np.float64)
    hr_grid = np.asarray(hr_grid, dtype=np.float64)

    if Z.ndim != 2:
        raise ValueError("Z must be 2D")
    n_hr, n_q = Z.shape
    if len(hr_grid) != n_hr:
        raise ValueError("len(hr_grid) must equal Z.shape[0]")
    if np.any(np.diff(hr_grid) < 0):
        raise ValueError("hr_grid must be sorted ascending")
    if epsilon < 0:
        raise ValueError("epsilon must be >= 0")

    # Log-domain scores
    base = np.full_like(Z, -np.inf, dtype=np.float64)
    mask = Z > 0
    base[mask] = np.log(Z[mask])

    # Transition windows
    if local_eps is None or np.isinf(local_eps):
        lo = np.zeros(n_hr, dtype=np.int32)
        hi = np.full(n_hr, n_hr, dtype=np.int32)
    else:
        local_eps = float(local_eps)
        lo = np.searchsorted(hr_grid, hr_grid - local_eps, side="left").astype(np.int32)
        hi = np.searchsorted(hr_grid, hr_grid + local_eps, side="right").astype(np.int32)

    # Suppression tube around accepted / rejected candidates
    tube_eps = float(tube_scale * epsilon)
    if tube_eps < 0:
        tube_eps = 0.0

    tube_lo = np.searchsorted(hr_grid, hr_grid - tube_eps, side="left").astype(np.int32)
    tube_hi = np.searchsorted(hr_grid, hr_grid + tube_eps, side="right").astype(np.int32)

    work = base.copy()
    cols = np.arange(n_q)
    accepted = []

    eps2 = epsilon * epsilon

    def far_enough(hr_path):
        if epsilon == 0 or not accepted:
            return True
        for a in accepted:
            if np.mean((hr_path - a) ** 2) < eps2:
                return False
        return True

    def suppress_path(path_rows):
        if hard_suppression:
            for c, r in enumerate(path_rows):
                work[tube_lo[r]:tube_hi[r], c] = -np.inf
        else:
            for c, r in enumerate(path_rows):
                sl = slice(tube_lo[r], tube_hi[r])
                block = work[sl, c]
                finite = np.isfinite(block)
                block[finite] -= penalty
                work[sl, c] = block

    internal = 0
    while internal < max_internal_candidates:
        internal += 1

        _, path_rows = _viterbi_best(work, lo, hi)
        if path_rows.size == 0:
            return

        hr_path = hr_grid[path_rows]
        original_score = float(base[path_rows, cols].sum())

        # Always suppress the found candidate so the next search moves elsewhere
        suppress_path(path_rows)

        # Only yield if it passes the exact RMSE test
        if far_enough(hr_path):
            accepted.append(hr_path.copy())
            yield hr_path, original_score
