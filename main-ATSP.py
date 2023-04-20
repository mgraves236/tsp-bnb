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


def find(i: int, parent: array):
    while parent[i] != i:
        i = parent[i]
    return i


def union(i: int, j: int, parent: array):
    a = find(i, parent)
    b = find(j, parent)
    parent[a] = b


def kruskal(matrix: array, size: int, edges: array):
    mincost = 0
    path = []
    parent = [i for i in range(size)]
    # Check if there are any predefined edges
    for i in range(0, size + 1):
        for j in range(0, size + 1):
            if edges[i][j] == True:
                print(1)


    # No of edges is size, but we start from 0
    for i in range(0, size-1):
        min = maxsize
        a = -1
        b = -1
        for j in range(0, size):
            for k in range(0, size):
                if find(j, parent) != find(k, parent) and matrix[j][k] < min:
                    min = matrix[j][k]
                    a = j
                    b = k
        union(a, b, parent)
        path.append(a)
        path.append(b)
        mincost += min
    return {'mincost': mincost,
            'path': path}


def least_cost_edges(matrix: array, i: int):
    least_cost = {'min1': maxsize, 'min2': maxsize}
    # Find first minimum
    for j in range(0, size):
        if i == j:
            continue
        elif matrix[i][j] < least_cost['min1'] and i != j:
            least_cost['min1'] = matrix[i][j]
        elif least_cost['min2'] > matrix[i][j] != least_cost['min1'] and i != j:
            least_cost['min2'] = matrix[i][j]
    return least_cost


def reduce(matrix: array, index: int):
    reduced = np.delete(matrix, index, 0)
    reduced = reduced.tolist()
    reduced = np.delete(reduced, index, 1)
    reduced = reduced.tolist()
    return reduced


def lowerbound(matrix: array, node: int, size: int, edges: array):
    # Kruskal's Algorithm
    # Create two undirected graphs from one directed
    # Delete a vertex then find a minimum spanning tree

    upper = [[0 for x in range(size)] for y in range(size)]
    lower = [[0 for x in range(size)] for y in range(size)]
    upper_edges = [[0 for x in range(size)] for y in range(size)]
    lower_edges = [[0 for x in range(size)] for y in range(size)]

    for i in range(0, size):
        for j in range(0, size):
            upper[i][j] = matrix[i][j]
            lower[i][j] = matrix[i][j]
    # Upper half
    for i in range(0, size):
        for j in range(0, size):
            upper[j][i] = matrix[i][j]
    # Lower half
    for i in range(0, size):
        for j in range(0, size):
            lower[i][j] = matrix[j][i]

    # Find lower bound for upper
    reduced = reduce(upper, node)
    least_edges = least_cost_edges(upper, node)
    lb_up = kruskal(reduced, size - 1, edges)
    lb_up = lb_up['mincost'] + least_edges['min1'] + least_edges['min2']
    # lb_up = lb_up['mincost'] + least_edges['min2']

    # Find lower bound for lower
    reduced = reduce(lower, node)
    least_edges = least_cost_edges(lower, node)
    lb_low = kruskal(reduced, size - 1, edges)
    lb_low = lb_low['mincost'] + least_edges['min1'] + least_edges['min2']
    # lb_low = lb_low['mincost'] + least_edges['min2']

    if lb_low > lb_up:
        return lb_up
    else:
        return lb_low


def tsp(adj: array, size: int):
    # Set upper bound -- initial best tour cost
    upperbound = greedy(adj, 0)
    upperbound = upperbound['value']
    print(upperbound)
    def_edges = [[0 for x in range(size)] for y in range(size)]
    for i in range(0, size):
        for j in range(0, size):
            def_edges[i][j] = False


    # Iterative DFS

    # for i in range(0, size):
    #     print(lowerbound(adj, i, size))
    matrix = [[0 for x in range(size)] for y in range(size)]

    for i in range(0, size):
        for j in range(0, size):
            matrix[i][j] = adj[i][j]
            matrix[i][j] = adj[i][j]
    stack = []
    visited = [False] * size
    stack.append(0)
    visited[0] = True
    stack.append(1)
    visited[1] = True
    def_edges[0][1] = True
    print(lowerbound(adj, 1, size, def_edges))



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
