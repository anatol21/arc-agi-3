def solve(grid):
    # The demonstrations show isolated non-background pixels moving by fixed
    # color-specific offsets:
    #   1 -> (0, +1)
    #   2 -> (0, -2)
    #   7 -> (-2, 0)
    #   9 -> (+2, 0)
    # Background color is 8.
    moves = {
        1: (0, 1),
        2: (0, -2),
        7: (-2, 0),
        9: (2, 0),
    }

    h = len(grid)
    w = len(grid[0]) if h else 0
    out = [[8 for _ in range(w)] for _ in range(h)]

    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            if v == 8:
                continue
            if v in moves:
                dr, dc = moves[v]
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr][nc] = v
            else:
                out[r][c] = v

    return out