import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
cimport cython

DTYPE = np.int32

@cython.boundscheck(False)
@cython.wraparound(False)
def get_bfs_order(int[:, :] edge_index, int n, int start):
    # n : number of vertices

    # get number of edges
    cdef int m = edge_index.shape[1]

    # 1 - compute outdegree of each vertex
    cdef int* outdegree = <int*>malloc(n * sizeof(int))
    for i in range(n):
        outdegree[i] = 0

    for i in range(m):
        outdegree[edge_index[0, i]] += 1

    # 2 - allocate memory for adjacency list
    cdef int **adj_list = <int**>malloc(n * sizeof(int *))
    for i in range(n):
        adj_list[i] = <int*>malloc(outdegree[i] * sizeof(int))

    # 3 - fill adjacency list
    for i in range(n):
        outdegree[i] = 0

    cdef int src
    cdef int dst
    for i in range(m):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        adj_list[src][outdegree[src]] = dst
        outdegree[src] += 1

    # ############ perform bfs ############

    # 1 - initialize queue
    cdef int *queue = <int*>malloc(n * sizeof(int))
    cdef int queue_start = 0
    cdef int queue_end = 0

    # 2 - initialize depth array
    # element i is shortest distance from start if visited
    # -1 otherwise
    cdef cnp.ndarray depth_np = np.full(n, -1, dtype=DTYPE)
    cdef int[:] depth = depth_np
    # cdef int *depth = <int*>malloc(n * sizeof(int))
    # for i in range(n):
    #     depth[i] = -1

    # 3 - perform bfs
    queue[queue_end] = start
    queue_end = 1
    depth[start] = 0

    # initialize order as a numpy array
    cdef cnp.ndarray order_np = np.zeros(n, dtype=DTYPE)
    cdef int[:] order = order_np
    cdef int order_index = 0
    cdef int current, neighbor
    while order_index < n:
        # complete queue
        while queue_start < queue_end:
            # perform dequeue
            current = queue[queue_start]
            queue_start += 1
            # append current to order array
            order[order_index] = current
            order_index += 1
            # enqueue neighbors
            for i in range(outdegree[current]):
                neighbor = adj_list[current][i]
                if depth[neighbor] < 0:
                    queue[queue_end] = neighbor
                    queue_end += 1
                    depth[neighbor] = depth[current] + 1

        # safety check for disconnected graphs
        if order_index < n:
            for i in range(n):
                if depth[i] < 0:
                    queue[queue_end] = i
                    queue_end += 1
                    depth[i] = 0
                    break


    # ############ free memory ############
    free(outdegree)
    for i in range(n):
        free(adj_list[i])
    free(adj_list)
    free(queue)

    return np.stack((order_np, depth_np), axis=0)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_dfs_order(int[:, :] edge_index, int n, int start):
    # n : number of vertices

    # get number of edges
    cdef int m = edge_index.shape[1]

    # 1 - compute outdegree of each vertex
    cdef int* outdegree = <int*>malloc(n * sizeof(int))
    for i in range(n):
        outdegree[i] = 0

    for i in range(m):
        outdegree[edge_index[0, i]] += 1

    # 2 - allocate memory for adjacency list
    cdef int **adj_list = <int**>malloc(n * sizeof(int *))
    for i in range(n):
        adj_list[i] = <int*>malloc(outdegree[i] * sizeof(int))

    # 3 - fill adjacency list
    for i in range(n):
        outdegree[i] = 0

    cdef int src
    cdef int dst
    for i in range(m):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        adj_list[src][outdegree[src]] = dst
        outdegree[src] += 1

    # ############ perform dfs ############

    # 1 - initialize stack
    cdef int *stack = <int*>malloc(n * sizeof(int))
    cdef int stack_top = 0

    # 2 - initialize depth array
    # element i is shortest distance from start if visited
    # -1 otherwise
    cdef cnp.ndarray depth_np = np.full(n, -1, dtype=DTYPE)
    cdef int[:] depth = depth_np

    # 3 - perform dfs
    stack[stack_top] = start
    stack_top += 1
    depth[start] = 0

    # initialize order as a numpy array
    cdef cnp.ndarray order_np = np.zeros(n, dtype=DTYPE)
    cdef int[:] order = order_np
    cdef int order_index = 0
    cdef int current, neighbor
    while order_index < n:
        # complete stack
        while stack_top > 0:
            # perform pop
            stack_top -= 1
            current = stack[stack_top]
            # append current to order array
            order[order_index] = current
            order_index += 1
            # push neighbors (in reverse order to maintain correct traversal)
            for i in range(outdegree[current] - 1, -1, -1):
                neighbor = adj_list[current][i]
                if depth[neighbor] < 0:
                    stack[stack_top] = neighbor
                    stack_top += 1
                    depth[neighbor] = depth[current] + 1
        
        # safety check for disconnected graphs
        if order_index < n:
            for i in range(n):
                if depth[i] < 0:
                    stack[stack_top] = i
                    stack_top += 1
                    depth[i] = 0
                    break
    

    # ############ free memory ############
    free(outdegree)
    for i in range(n):
        free(adj_list[i])
    free(adj_list)
    free(stack)

    return np.stack((order_np, depth_np), axis=0)