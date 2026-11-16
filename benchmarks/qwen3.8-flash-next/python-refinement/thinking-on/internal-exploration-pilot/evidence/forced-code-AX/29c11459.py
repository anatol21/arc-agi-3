def solve(grid):
    """
    For each row containing exactly two non-zero endpoint pixels, draw a
    horizontal line between them. The left side is filled with the left
    endpoint's color, the right side with the right endpoint's color, and the
    exact midpoint (when it exists) is filled with color 5.
    """
    out = [row[:] for row in grid]
    h = len(out)
    if h == 0:
        return out
    w = len(out[0])

    for r in range(h):
        nonzero = [c for c in range(w) if out[r][c] != 0]
        if len(nonzero) != 2:
            continue

        c1, c2 = nonzero
        if c1 > c2:
            c1, c2 = c2, c1
            out[r][c1], out[r][c2] = out[r][c2], out[r][c1]

        left_color = out[r][c1]
        right_color = out[r][c2]

        if left_color == right_color:
            for c in range(c1, c2 + 1):
                out[r][c] = left_color
            continue

        mid2 = c1 + c2
        for c in range(c1, c2 + 1):
            if 2 * c < mid2:
                out[r][c] = left_color
            elif 2 * c > mid2:
                out[r][c] = right_color
            else:
                out[r][c] = 5

    return out