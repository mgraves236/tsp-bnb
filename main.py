# Travelling Salesman Problem
# Branch and Bound
import array
from sys import maxsize

import RandomNumberGenerator

random = RandomNumberGenerator.RandomNumberGenerator(5954)


# function to generate adjacent matrix for the graph
def generate_graph(size: int):
    matrix = [[0 for x in range(size)] for y in range(size)]
    for i in range(0, size):
        for j in range(0, size):
            if [i] == [j]:
                matrix[i][j] = 0
            else:
                matrix[i][j] = random.nextInt(1, 30)
    return matrix


def least_cost_edges(matrix: array, i : int):
    # node # least cost edges # total cost
    least_cost = {'min1': maxsize, 'min2': maxsize}
    # find first minimum
    for j in range(0, size):
        if i == j:
            continue
        elif matrix[i][j] < least_cost['min1'] and i != j:
            least_cost['min1'] = matrix[i][j]
        elif least_cost['min2'] > matrix[i][j] != least_cost['min1'] and i != j:
            least_cost['min2'] = matrix[i][j]
    return least_cost


# lower bound calculated as >= sum_(v in V) (sum of the cost of the two
# least cost edges adjacent to v)

def lower_bound(adj : array, i: int, j: int, level: int):
    least_i = least_cost_edges(adj, i)
    least_j = least_cost_edges(adj, j)

    if level == 1:
        return (least_i['min1'] + least_j['min1']) * 1/2 + adj[i][j]

    if level > 1:
        return (least_i['min2'] + least_j['min1']) * 1/2 + adj[i][j]


def tsp(adj: array):
    # tour always starts at node 0
    curr_lower_bound = 0
    upper_bound = maxsize
    visited = [False] * size
    curr_path = []
    level = 0 # keep track of depth of the tree

    least = []
    sum = 0
    # always start from node 0
    for i in range(0, size):
        least = least_cost_edges(adj, i)
        print(least['min1'] + least['min2'])
        sum += least['min1'] + least['min2']
    sum = sum / 2
    curr_lower_bound = sum
    visited[0] = True
    curr_path.append(0)
    # lower_bound(adj, 0, 1, 1)

    if level == size:
        return



if __name__ == "__main__":
    size = 4  # number of graph vertices
    # starting point is indicated by rows and destination by cols
    # i.e. travelling from node 0 to 1 is indicated by mat[0][1]
    mat = generate_graph(size)

    print(mat)
    tsp(mat)
