def on_line(p, q, r):
    """Check if point q lies on the line segment pr"""

    # 1. Check for collinearity
    area = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if area != 0:
        return False

    # 2. Check if q is inside the bounding box
    if (
        q[0] <= max(p[0], r[0])
        and q[0] >= min(p[0], r[0])
        and q[1] <= max(p[1], r[1])
        and q[1] >= min(p[1], r[1])
    ):
        return True

    return False
