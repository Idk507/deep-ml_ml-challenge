def gridworld_policy_evaluation(policy: dict, gamma: float, threshold: float) -> list[list[float]]:
    grid_size = 5
    V = [[0.0 for _ in range(grid_size)] for _ in range(grid_size)]
    actions = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
    reward = -1

    while True:
        delta = 0.0
        new_V = [row[:] for row in V]
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) in [(0, 0), (0, grid_size - 1), (grid_size - 1, 0), (grid_size - 1, grid_size - 1)]:
                    continue  # Skip terminal states
                v = 0.0
                for action, prob in policy[(i, j)].items():
                    di, dj = actions[action]
                    ni = i + di
                    nj = j + dj
                    # Stay in place if move goes off grid
                    if ni < 0 or ni >= grid_size or nj < 0 or nj >= grid_size:
                        ni, nj = i, j
                    v += prob * (reward + gamma * V[ni][nj])
                new_V[i][j] = v
                delta = max(delta, abs(V[i][j] - new_V[i][j]))
        V = new_V
        if delta < threshold:
            break
    return V
