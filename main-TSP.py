# Travelling Salesman Problem
# Branch and Bound
import array
import numpy
import math
from sys import maxsize
import numpy as np

import RandomNumberGenerator

random = RandomNumberGenerator.RandomNumberGenerator(5546568)


# Function to generate adjacent matrix for the graph
def generate_graph(size: int):
    matrix = [[0 for x in range(size)] for y in range(size)]
    for i in range(0, size):
        for j in range(0, size):
            if [i] == [j]:
                matrix[i][j] = maxsize
            else:
                matrix[i][j] = random.nextInt(1, 30)
    return matrix


def greedy(matrix: array, start: int):
    # Nearest Neighbour Algorithm
    min_tour = 0
    path = [start]
    visited = [False] * size
    visited[start] = True
    current_node = start

    index = -1
    for i in range(0, size - 1):
        min_row = maxsize
        for j in range(0, size):
            if current_node == j: continue
            if matrix[current_node][j] < min_row and not visited[j]:
                min_row = matrix[current_node][j]
                index = j
        current_node = index
        visited[current_node] = True
        path.append(current_node)
        min_tour += min_row

    # Back to the 0th node
    last_index = path[-1]
    min_tour += matrix[last_index][0]
    path.append(path[0])

    return {'value': min_tour,
            'path': path}


def lowerbound(matrix: array, i: int, j:int, arr_min: array, visited: array):
    lb = matrix[i][j]
    visited[0] = False
    visited[j] = True
    sum = 0
    for i in range(0, size):
        if not visited[i]:
            sum += arr_min[i]
    visited[0] = True
    visited[j] = False

    return sum + lb


def tsp(adj: array, size: int):
    # Set upper bound -- initial best tour cost
    upperbound = greedy(adj, 0)
    upperbound = upperbound['value']
    # Find min of each column
    arr_min = np.min(adj, axis=0)
    arr_min = arr_min.tolist()

    # Iterative DFS
    stack = []
    prev = 0
    visited = [False] * size
    stack.append((0, False))
    lb = 0
    weight = []
    path = []
    cost = 0
    while len(stack) > 0:
        # flag indicates if this vertex is reached from 0
        (v, flag) = stack.pop()
        if flag:
            visited[prev] = False
            prev = 0
            weight = []
        if v != 0:
            lb = lowerbound(adj, prev, v, arr_min, visited)
            for k in range(0, len(weight)):
                lb += weight[k][0]
            weight.append((adj[prev][v], prev, v))
        prev = v
        if not visited[v]:
            visited[v] = True
            leaf = True
            for j in range(0, size):
                if not visited[j]:
                    leaf = False

                    if lb <= upperbound:  # else reduce
                        if v == 0:
                            stack.append((j, True))  # reached from 0
                        else:
                            stack.append((j, False))
            if leaf:
                newLb = lb + adj[v][0]
                if newLb < upperbound:
                    upperbound = newLb
                    path = [k for (i, j, k) in weight]
                    path.insert(0, 0)
                    path.append(0)
                visited = [False] * size
                visited[0] = True
                tmp = weight[0]
                prev = tmp[2]
                visited[prev] = True
                weight = [tmp]
    return {'value': lb,
            'path': path}


if __name__ == "__main__":
    size = 4  # number of graph vertices
    # Starting point is indicated by rows and destination by cols
    # i.e. travelling from node 0 to 1 is indicated by mat[0][1]
    mat = generate_graph(size)
    tsp = tsp(mat, size)
    nn = greedy(mat, 0)
    print('-----------------------------------------------------------')
    print('\t\t\t\tTSP')
    print('-----------------------------------------------------------')
    print('NEAREST NEIGHBOURS\t', nn)
    print('B&B\t', tsp)
