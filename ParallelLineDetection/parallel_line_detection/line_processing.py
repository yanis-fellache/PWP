from utils import compute_slope, compute_length, update


def compute_line(lines):
    used = [False] * len(lines)
    ans = []
    for i in range(len(lines)):
        if used[i] == False:
            used[i] = True

            x1, y1, x2, y2 = lines[i][0]
            base_slope = compute_slope(lines[i][0])

            if 0.1 > base_slope > -0.1:
                for j in range(i, len(lines)):
                    temp = compute_slope(lines[j][0])

                    if used[j] == False and 0.1 > temp > -0.1 and y1 + 40 > lines[j][0][1] > y1 - 40:
                        x1, y1, x2, y2 = update(
                            lines[j][0], x1, y1, x2, y2, temp
                        )
                        used[j] = True

                ans.append([x1, int((y1 + y2) / 2), x2, int((y1 + y2) / 2)])
            else:
                for j in range(i, len(lines)):
                    temp = compute_slope(lines[j][0])

                    if used[j] == False and base_slope + 0.20 > temp > base_slope - 0.20:
                        x1, y1, x2, y2 = update(
                            lines[j][0], x1, y1, x2, y2, temp
                        )
                        used[j] = True
                ans.append([x1, y1, x2, y2])
    return ans


def are_parallel(slope1, slope2):
        if abs(slope1) < 0.1 and abs(slope2) < 0.1:
            return True
        if abs(slope1) > 10 and abs(slope2) > 10:
            return True
        return abs(abs(slope1) - abs(slope2)) < 0.5


def compute_center(lines):
    # parallel_pairs = []
    # for i in range(len(lines)):
    #     for j in range(i + 1, len(lines)):
    #         slope1 = compute_slope(lines[i])
    #         slope2 = compute_slope(lines[j])

    #         if are_parallel(slope1, slope2):
    #             length1 = compute_length(lines[i])
    #             length2 = compute_length(lines[j])

    #             if length1 > 100 and length2 > 100:
    #                 parallel_pairs.append((lines[i], lines[j], length1 + length2))

    # if not parallel_pairs:
    #     return 

    # parallel_pairs.sort(key=lambda x: x[2], reverse=True)
    # line1, line2, _ = parallel_pairs[0]
    ans = [0, 0, 0, 0]
    if len(lines) > 1:
        # print(f"n_lines: {lines}")
        line1, line2 = lines[0], lines[1]

        x_temp = sorted([line1[0], line1[2]])
        y_temp = sorted([line1[1], line1[3]])
        line1 = [x_temp[0], y_temp[0], x_temp[1], y_temp[1]]

        x_temp = sorted([line2[0], line2[2]])
        y_temp = sorted([line2[1], line2[3]])
        line2 = [x_temp[0], y_temp[0], x_temp[1], y_temp[1]]
        
        x1 = int((line1[0] + line2[0]) / 2)
        y1 = int((line1[1] + line2[1]) / 2)
        x2 = int((line1[2] + line2[2]) / 2)
        y2 = int((line1[3] + line2[3]) / 2)
        ans = [x1, y1, x2, y2]
        print(ans)
    return ans
