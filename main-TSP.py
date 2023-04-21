# Travelling Salesman Problem
# Branch and Bound
import array
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


# Function to generate symmetric adjacent matrix for the graph
def generate_graph_sym(size: int):
    matrix = [[0 for x in range(size)] for y in range(size)]
    for i in range(0, size):
        for j in range(i, size):
            if [i] == [j]:
                matrix[i][j] = maxsize
            else:
                matrix[i][j] = random.nextInt(1, 30)
    for i in range(0, size):
        for j in range(i, size):
            matrix[j][i] = matrix[i][j]
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

    # back to the 0th node
    last_index = path[-1]
    min_tour += matrix[last_index][0]
    path.append(path[0])

    return {'value': min_tour,
            'path': path}


def lowerbound(matrix: array, i: int, j: int, arr_min: array, visited: array):
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


# Check if there are cycles
def find(i: int, parent: array):
    while parent[i] != i:
        i = parent[i]
    return i


def union(i: int, j: int, parent: array):
    a = find(i, parent)
    b = find(j, parent)
    parent[a] = b


def kruskal(matrix: array, size: int, edges: array = []):
    # Kruskal's Algorithm
    mincost = 0
    path = []
    parent = [i for i in range(size)]

    # no of edges is v - 1
    curr_size = size - 1
    if len(edges) != 0:
        for i in range(0, size):
            for j in range(i, size):
                if edges[i][j] and i != j:
                    curr_size -= 1  # one edge is set
                    union(i, j, parent)
                    path.append(i)
                    path.append(j)
                    mincost += matrix[i][j]
    # assign unassigned edges
    for i in range(0, curr_size):
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
    for j in range(0, size):
        if i == j:
            continue
        elif matrix[i][j] < least_cost['min1'] and i != j:
            least_cost['min1'] = matrix[i][j]
        elif least_cost['min2'] > matrix[i][j] != least_cost['min1'] and i != j:
            least_cost['min2'] = matrix[i][j]
    return least_cost


# Function to delete node from graph
def reduce(matrix: array, index: int):
    reduced = matrix.copy()
    reduced = np.delete(matrix, index, 0)
    reduced = reduced.tolist()
    reduced = np.delete(reduced, index, 1)
    reduced = reduced.tolist()
    return reduced


def lowerbound_k(matrix: array, node: int, size: int, edges: array):
    # Delete a vertex then find a minimum spanning tree
    reduced = reduce(matrix, node)
    least_edges = least_cost_edges(matrix, node)
    reduced_edges = reduce(edges, node)
    lb = kruskal(reduced, size - 1, reduced_edges)
    lb = lb['mincost'] + least_edges['min1'] + least_edges['min2']

    return lb


def tsp(adj: array, size: int):
    # set upper bound -- initial best tour cost
    upperbound = greedy(adj, 0)
    path = [i for i in upperbound['path']]
    upperbound = upperbound['value']
    # find min of each column
    arr_min = np.min(adj, axis=0)
    arr_min = arr_min.tolist()
    do_kruskal = False

    min_adj = [[maxsize for x in range(size)] for y in range(size)]
    # matrix with minimum cost
    for i in range(0, size):
        for j in range(0, size):
            if adj[i][j] > adj[j][i]:
                min_adj[i][j] = adj[j][i]
            else:
                min_adj[i][j] = adj[i][j]

    edges = [[False for x in range(size)] for y in range(size)]

    for i in range(0, size):
        edges[i][i] = True

    # Iterative DFS
    stack = []
    prev = 0
    prevLevel = 0
    visited = [False] * size
    stack.append((0, 0))
    lb = 0
    weight = []
    path = []
    doErase = False

    while len(stack) > 0:
        # print("1", stack)
        (v, l) = stack.pop()
        # print("pop ", v, l, doErase)
        if v != 0:

            if len(weight) != 0 and doErase:
                doErase = False
                erase = prevLevel - l + 1
                weight = weight[:-erase]

                if len(weight) != 0:
                    prev = weight[-1][2]
                else:
                    prev = 0
                visited = [False] * size
                # print("lll", weight)
                for (i, j, k) in weight:
                    visited[j] = True
                    visited[k] = True

            if do_kruskal:
                lb = lowerbound_k(min_adj, v, size, edges)
            else:
                lb = lowerbound(adj, prev, v, arr_min, visited)

            for k in range(0, len(weight)):
                lb += weight[k][0]
            weight.append((adj[prev][v], prev, v))
            # print("weights", weight)

        edges[prev][v] = True
        prev = v
        prevLevel = l
        if not visited[v]:
            visited[v] = True
            # print("v ", v)
            leaf = True
            for j in range(0, size):
                if not visited[j]:
                    leaf = False
                    # print("i ", j)
                    # TODO tutaj chyba mamy wracać do 0 itp.
                    if lb <= upperbound:  # else prune
                        stack.append((j, l + 1))
                    else:
                        # print("prune ", j)
                        doErase = True
            if leaf:
                # if do_kruskal:
                #     newLb = lb
                # else:
                #     newLb = lb + adj[v][0]

                newLb = 0
                for (a, b, c) in weight:
                    newLb += a
                newLb += adj[weight[-1][2]][0]

                # TODO to chyba nie tu..
                if newLb < upperbound:
                    upperbound = newLb
                    path = [k for (i, j, k) in weight]
                    path.insert(0, 0)
                    path.append(0)
                    # print("path ", path)

                doErase = True
                # print("Leaf ", v)
    print(path)
    return {'value': upperbound,
            'path': path}


if __name__ == "__main__":
    size = 5  # number of graph vertices
    # starting point is indicated by rows and destination by cols
    # i.e. travelling from node 0 to 1 is indicated by mat[0][1]
    mat = generate_graph(size)
    print('-----------------------------------------------------------')
    print('\t\t\t\t\tTSP')
    print('-----------------------------------------------------------')
    nn = greedy(mat, 0)
    print(mat)
    print('NEAREST NEIGHBOURS\t', nn)
    tsp = tsp(mat, size)
    print('B&B\t', tsp)

    # mat2 = [[maxsize, 7, 8, maxsize, maxsize, maxsize],
    #         [7, maxsize, 3, 6, maxsize, 5],
    #         [8, 3, maxsize, 4, maxsize, 3],
    #         [maxsize, 6, 4, maxsize, 5, 2],
    #         [maxsize, maxsize, maxsize, 5, maxsize, 2],
    #         [maxsize, 5, 3, 2, 2, maxsize]]
    # mat2 = [[maxsize, 2, maxsize, 6, maxsize],
    #         [2, maxsize, 3, 8, 5],
    #         [maxsize, 3, maxsize, maxsize, 7],
    #         [6, 8, maxsize, maxsize, 9],
    #         [maxsize, 5, 7, 9, maxsize]]
    #
    # k = kruskal(mat2, 5)
    # print(k)
    # mat2 = [[maxsize, 5, 5],
    #         [5, maxsize, 10],
    #         [5, 10, maxsize]]
    #
    # k = kruskal(mat2, 3)
    # print(k)