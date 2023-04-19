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


# lower bound calculated as >= sum_(v in V) (sum of the cost of the two
# least cost edges adjacent to v)
def lower_bound(matrix: array, i):
    # node # least cost edges # total cost
    least_cost = {'min1': maxsize, 'min2': maxsize}
    # find first minimum
    for j in range(0, size):
        if i == j: continue
        elif matrix[i][j] <  least_cost['min1'] and i != j:
            least_cost['min1'] = matrix[i][j]
        elif least_cost['min2'] > matrix[i][j] != least_cost['min1'] and i != j:
            least_cost['min2'] = matrix[i][j]

    print(least_cost)


def tsp(adj: array):
    # tour always starts at node 0
    lower_bound(adj, 0)


if __name__ == "__main__":
    size = 4  # number of graph vertices
    # starting point is indicated by rows and destination by cols
    # i.e. travelling from node 0 to 1 is indicated by mat[0][1]
    mat = generate_graph(size)

    print(mat)
    tsp(mat)
