from collections import deque
import numpy as np

def get_bfs_order(edge_index, n, start):

    neighborhood = [[] for _ in range(n)]
    for i in range(edge_index.shape[1]):
        neighborhood[edge_index[0][i]].append(edge_index[1][i])

    depth = [-1] * n

    queue = deque([start])
    depth[start] = 0
    order = [0] * n
    order_index = 0
    while order_index < n:
        # complete queue
        while queue:
            current = queue.popleft()
            order[order_index] = current
            order_index += 1
            for neighbor in neighborhood[current]:
                if depth[neighbor] < 0:
                    queue.append(neighbor)
                    depth[neighbor] = depth[current] + 1

        # safety check for disconnected graphs
        if order_index < n:
            for i in range(n):
                if depth[i] < 0:
                    queue.append(i)
                    depth[i] = 0
                    break

    arr = np.array(order, dtype=np.int32)
    depth = np.array(depth, dtype=np.int32)

    return arr, depth


def get_dfs_order(edge_index, n, start):

    neighborhood = [[] for _ in range(n)]
    for i in range(edge_index.shape[1]):
        neighborhood[edge_index[0][i]].append(edge_index[1][i])

    depth = [-1] * n

    stack = [start]
    depth[start] = 0
    order = [0] * n
    order_index = 0
    while order_index < n:
        # complete stack
        while stack:
            current = stack.pop()
            order[order_index] = current
            order_index += 1
            for neighbor in neighborhood[current]:
                if depth[neighbor] < 0:
                    stack.append(neighbor)
                    depth[neighbor] = depth[current] + 1

        # safety check for disconnected graphs
        if order_index < n:
            for i in range(n):
                if depth[i] < 0:
                    stack.append(i)
                    depth[i] = 0
                    break

    arr = np.array(order, dtype=np.int32)
    depth = np.array(depth, dtype=np.int32)

    return arr, depth