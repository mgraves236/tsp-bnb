from matplotlib import pyplot as plt

if __name__ == "__main__":

    n_arr = []
    time_arr = []
    with open('tests.txt') as data:
        for line in data:
            row = line.split()
            print(row)
            if row:
                time_arr.append(float(row[1]))
                n_arr.append(int(row[0]))
    n_arr_k = []
    time_arr_k = []
    with open('tests_kruskal.txt') as data:
        for line in data:
            row = line.split()
            print(row)
            if row:
                time_arr_k.append(float(row[1]))
                n_arr_k.append(int(row[0]))

    plt.title("Time vs. Sample size")
    plt.xlabel("n")
    plt.ylabel("time [s]")
    plt.plot(n_arr, time_arr, label="Min Lowerbound")
    plt.plot(n_arr_k, time_arr_k, label="Kruskal Lowerbound")
    plt.legend()
    plt.show()
