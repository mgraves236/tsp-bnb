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
            if matrix[current_node][j] < min_row and visited[j] == False:
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


def lowerbound(matrix: array, node: int, size: int):
    # Kruskal's Algorithm
    # Delete a vertex then find a minimum spanning tree

    reduced = np.delete(matrix, node, 0)
    reduced = reduced.tolist()
    reduced = np.delete(matrix, node, 1)
    reduced = reduced.tolist()

def tsp(adj: array):
    # Always start from node 0
    # set upper bound -- initial best tour cost
    upperbound = greedy(adj, 0)
    upperbound = upperbound['value']
    print(upperbound)
    lowerbound(adj, 0, size)


if __name__ == "__main__":
    size = 4  # number of graph vertices
    # starting point is indicated by rows and destination by cols
    # i.e. travelling from node 0 to 1 is indicated by mat[0][1]
    mat = generate_graph(size)

    print(mat)
    tsp(mat)
