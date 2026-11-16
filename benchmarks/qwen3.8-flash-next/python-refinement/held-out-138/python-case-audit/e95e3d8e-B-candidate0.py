def solve(grid):
    h = len(grid)
    w = len(grid[0]) if h else 0
    if h == 0 or w == 0:
        return []

    # Try to infer a small repeating tile from the non-zero cells.
    # The tile size is the smallest period consistent with all non-zero cells.
    best = None
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            if ph * pw > 100:  # reasonable for ARC-sized grids
                continue
            tile = [[None] * pw for _ in range(ph)]
            ok = True
            for r in range(h):
                for c in range(w):
                    v = grid[r][c]
                    if v != 0:
                        tr, tc = r % ph, c % pw
                        if tile[tr][tc] is None:
                            tile[tr][tc] = v
                        elif tile[tr][tc] != v:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                # Prefer the smallest tile that explains the grid.
                if best is None or ph * pw < best[0] * best[1]:
                    best = (ph, pw, tile)
                break
        if best is not None:
            break

    if best is None:
        return [row[:] for row in grid]

    ph, pw, tile = best
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            out[r][c] = tile[r % ph][c % pw]
    return out
