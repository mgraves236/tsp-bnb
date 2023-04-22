# Travelling Salesman Problem
# Branch and Bound
import array
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


# Check if two vertices are not in the same set
# else they will form a cycle
def find(i: int, parent: array):
    while parent[i] != i:
        i = parent[i]
    return i

# Connect two vertices with an edge
def union(i: int, j: int, parent: array):
    a = find(i, parent)
    b = find(j, parent)
    parent[a] = b


def kruskal(matrix: array, size: int, edges: array = []):
    # Kruskal's Algorithm
    mincost = 0
    path = []
    # all vertices in disjoint sets
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
    # print('least_cost', matrix)
    least_cost = {'min1': maxsize, 'min2': maxsize}
    for j in range(0, size):
        if i == j:
            continue
        elif matrix[i][j] < least_cost['min1'] and i != j:
            least_cost['min1'] = matrix[i][j]

    for j in range(0, size):
        if i == j:
            continue
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
    reduced_edges = reduce(edges, node)
    lb = kruskal(reduced, size - 1, reduced_edges)
    least_edges = least_cost_edges(matrix, node)
    lb = lb['mincost'] + least_edges['min1'] + least_edges['min2']

    return lb


def tsp(adj: array, size: int):
    # set upper bound -- initial best tour cost
    upperbound = greedy(adj, 0)
    path = upperbound['path']
    upperbound = upperbound['value']
    # find min of each column
    arr_min = np.min(adj, axis=0)
    arr_min = arr_min.tolist()
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
    prev_level = 0
    visited = [False] * size
    stack.append((0, 0))
    lb = 0
    weight = []
    do_erase = False

    while len(stack) > 0:
        (v, l) = stack.pop()
        if v != 0:

            if len(weight) != 0 and do_erase:
                do_erase = False
                erase = prev_level - l + 1
                weight = weight[:-erase]

                if len(weight) != 0:
                    prev = weight[-1][2]
                else:
                    prev = 0
                visited = [False] * size
                for (i, j, k) in weight:
                    visited[j] = True
                    visited[k] = True
            edges[prev][v] = True
            edges[v][prev] = True
            if do_kruskal:
                lb = lowerbound_k(min_adj, v, size, edges)
            else:
                lb = lowerbound(adj, prev, v, arr_min, visited)

            # add already fixed edges
            for k in range(0, len(weight)):
                lb += weight[k][0]
            weight.append((adj[prev][v], prev, v))


        prev = v
        prev_level = l
        if lb > upperbound: # prune
            do_erase = True
        else:
            if not visited[v]:
                visited[v] = True
                leaf = True
                for j in range(0, size):
                    if not visited[j]:

                        leaf = False
                        stack.append((j, l + 1))

                if leaf:
                    newLb = 0
                    # calculate real tour cost
                    for (a, b, c) in weight:
                        newLb += a
                    newLb += adj[weight[-1][2]][0]

                    if newLb < upperbound:
                        upperbound = newLb
                        path = [k for (i, j, k) in weight]
                        path.insert(0, 0)
                        path.append(0)

                    do_erase = True
    return {'value': upperbound,
            'path': path}


do_kruskal = False

if __name__ == "__main__":
    size = 4  # number of graph vertices
    # starting point is indicated by rows and destination by cols
    # i.e. travelling from node 0 to 1 is indicated by mat[0][1]
    mat = generate_graph_sym(size)
    print(mat)
    print('-----------------------------------------------------------')
    print('\t\t\t\t\tTSP')
    print('-----------------------------------------------------------')
    nn = greedy(mat, 0)
    print('NEAREST NEIGHBOURS\t', nn)
    # tsp_var = tsp(mat, size)
    # print('B&B\t', tsp_var)
    do_k = ''
    for i in range(0, 2):
        tsp_var = tsp(mat, size)
        if do_kruskal:
            do_k = 'Kruskal'
        print('B&B\t', do_k, tsp_var)
        do_kruskal = True



