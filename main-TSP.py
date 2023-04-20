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



def lowerbound(matrix: array, i: int, j:int, arr_min: array, visited: array):
    lb = matrix[i][j]
    print(lb)
    sum = 0
    for i in range(0, size):
        if not visited[i]:
            sum += arr_min[i]
            print(arr_min[i])
    return sum + lb

def tsp(adj: array, size: int):
    # Set upper bound -- initial best tour cost
    upperbound = greedy(adj, 0)
    upperbound = upperbound['value']
    print(adj)
    # Find min of each column
    arr_min = np.min(adj, axis=0)
    arr_min = arr_min.tolist()
    print(arr_min)

    # Iterative DFS
    stack = []
    visited = [False] * size
    stack.append(0)
    stack.append(1)
    visited[1] = True
    lb = lowerbound(adj, 0, 1, arr_min, visited)
    print(lb)
    # for i in range(0, size):
    #     stack = []
    #     visited = [False] * size
    #     stack.append(0)
    #     visited[0] = True
    #     stack.append(i)
    #     while len(stack) > 0:
    #         v = stack.pop()
    #         if not visited[v]:
    #             visited[v] = True






if __name__ == "__main__":
    size = 4  # number of graph vertices
    # Starting point is indicated by rows and destination by cols
    # i.e. travelling from node 0 to 1 is indicated by mat[0][1]
    mat = generate_graph(size)
    # mat2 = [[maxsize, 7, 8, maxsize, maxsize, maxsize],
    #         [7, maxsize, 3, 6, maxsize, 5],
    #         [8, 3, maxsize, 4, maxsize, 3],
    #         [maxsize, 6, 4, maxsize, 5, 2],
    #         [maxsize, maxsize, maxsize, 5, maxsize, 2],
    #         [maxsize, 5, 3, 2, 2, maxsize]]
    # k = kruskal(mat2, 6)
    # print(k)
    # print(mat)
    tsp(mat, size)
