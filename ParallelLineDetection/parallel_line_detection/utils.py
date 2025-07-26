import numpy as np

def compute_slope(line):
    x1, y1, x2, y2 = line
    if x2 - x1 == 0:
        return 0
    return (y2 - y1) / (x2 - x1)

def compute_length(line):
    x1, y1, x2, y2 = line
    return np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

def update(line, x1, y1, x2, y2, slope):
    x1 = min(x1, line[0])
    x2 = max(x2, line[2])

    if slope > 0:
        y2 = max(y2, line[3])
        y1 = min(y1, line[1])
    else:
        y2 = min(y2, line[3])
        y1 = max(y1, line[1])

    return x1, y1, x2, y2
