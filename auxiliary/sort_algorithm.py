import copy
import numpy as np


def arg_sort(table, axis=1, descending=True):

    tmp_table = table
    if axis == 0:
        tmp_table = tmp_table.T

    results_indices = []
    results_table = []
    for i in range(len(tmp_table)):
        indices, row = sort_one_row(tmp_table[i], descending)
        results_indices.append(indices)
        results_table.append(row)

    results_indices = np.asarray(results_indices, dtype=np.long)
    results_table = np.asarray(results_table, dtype=np.long)

    if axis == 0:
        results_indices = results_indices.T
        results_table = results_table.T

    return results_indices, results_table


def sort_one_row(row, descending=True):
    lens = len(row)
    indices = [i for i in range(lens)]
    tmp = copy.deepcopy(row)
    for i in range(lens):
        for j in range(lens-1):
            if tmp[j] > tmp[j+1]:
                if not descending:
                    swap(j, j+1, tmp)
                    swap(j, j+1, indices)
            elif tmp[j] < tmp[j+1]:
                if descending:
                    swap(j, j+1, tmp)
                    swap(j, j+1, indices)

    return indices, tmp


def swap(i, j, li):
    tmp = li[i]
    li[i] = li[j]
    li[j] = tmp


if __name__ == '__main__':
    pass
