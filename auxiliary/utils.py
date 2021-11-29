from __future__ import division
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
import xlwt
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm

from auxiliary.sort_algorithm import arg_sort


def calc_bleu(gold_list, pred_list):
    gold_list = [[sents] for sents in gold_list]
    bleu = corpus_bleu(gold_list, pred_list)
    return bleu*100


def get_examples(src):
    example_list = []
    with open(src, 'r') as f:
        lines = f.readlines()
        for line in lines:
            example_list.append(line.strip().split())

    return example_list


def cal(gold_src, pred_src):
    gold_list = get_examples(gold_src)
    pred_list = get_examples(pred_src)
    assert len(gold_list) == len(pred_list), "{}:{}".format(len(gold_list), len(pred_list))
    bleu = calc_bleu(gold_list, pred_list)
    return bleu


def _sort_tables(words, hier_meta):
    #
    home_start = int(hier_meta['home_ply_start'])
    home_end = int(hier_meta['home_ply_end']) + 1
    home_ply_kw_num = int(hier_meta['home_ply_kw_num'])
    home_ply_num = int(hier_meta['home_ply_num'])

    home_col_indices, home_row_indices = _sort_one_table(words, home_start, home_end, home_ply_num, home_ply_kw_num)

    assert len(home_col_indices) == home_ply_num*home_ply_kw_num
    assert len(home_row_indices) == home_ply_num*home_ply_kw_num
    #
    vis_ply_start = int(hier_meta['vis_ply_start'])
    vis_ply_end = int(hier_meta['vis_ply_end']) + 1
    vis_ply_kw_num = int(hier_meta['vis_ply_kw_num'])
    vis_ply_num = int(hier_meta['vis_ply_num'])

    vis_col_indices, vis_row_indices = _sort_one_table(words, vis_ply_start, vis_ply_end, vis_ply_num, vis_ply_kw_num)

    assert len(vis_col_indices) == vis_ply_num*vis_ply_kw_num
    assert len(vis_row_indices) == vis_ply_num * vis_ply_kw_num

    #
    team_start = int(hier_meta['team_start'])
    team_end = int(hier_meta['team_end']) + 1
    team_kw_num = int(hier_meta['team_kw_num'])
    team_num = int(hier_meta['team_num'])

    team_col_indices, team_row_indices = _sort_one_table(words, team_start, team_end, team_num, team_kw_num)
    assert len(team_col_indices) == team_num*team_kw_num
    assert len(team_row_indices) == team_num * team_kw_num

    return home_col_indices, home_row_indices, vis_col_indices, vis_row_indices, team_col_indices, team_row_indices


def _sort_one_table(words, start, end, row_num, kw_num, col_descending=True, row_descending=False):
    """

    :param words:
    :param start:
    :param end:
    :param row_num:
    :param kw_num:
    :param axis: 1 mean sort col
    :return:
    """
    table = list(words)[start:end]
    table = np.array(table).reshape(row_num, kw_num).transpose()

    for i in range(kw_num):
        for j in range(row_num):
            try:
                num = int(table[i][j])
            except:
                if table[i][j] == 'N/A':
                    num = -1
                else:
                    num = 0
            table[i][j] = num

    float_table = table.astype(np.float)
    # col_indices = np.argsort(-float_table, axis=1)
    """
    if col_descending:
        col_indices = np.argsort(-float_table, axis=1)
    else:
        col_indices = np.argsort(float_table, axis=1)
    """
    col_indices, _ = arg_sort(float_table, descending=col_descending)
    ranking = np.zeros(col_indices.shape, dtype=np.long)
    kw, n_row = col_indices.shape
    for i in range(kw):
        for j in range(n_row):
            pos = col_indices[i][j]
            ranking[i][pos] = j

    # row_indices = np.argsort(-ranking, axis=0)
    """
    if row_descending:  # 降序
        row_indices = np.argsort(-ranking, axis=0)
    else:
        row_indices = np.argsort(ranking, axis=0)
    """
    row_indices, _ = arg_sort(ranking, axis=0, descending=row_descending)
    col_indices = col_indices.transpose().reshape(row_num * kw_num)

    row_indices = row_indices.transpose().reshape(row_num * kw_num)

    col_indices = tuple(col_indices)
    row_indices = tuple(row_indices)
    return col_indices, row_indices


def export_excel(exported_excel, path):

    df = pd.DataFrame.from_dict(exported_excel, orient='index', columns=['BLEU'])
    df = df.reset_index().rename(columns={'index': 'epoch'})
    writer = pd.ExcelWriter(path)
    df.to_excel(path, encoding='utf-8')
    writer.save()


def plot_with_labels(lowDWeights, labels):
    plt.cla()
    # 降到二维了，分别给x和y
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    # 遍历每个点以及对应标签
    for x, y, s in zip(X, Y, labels):
        # c = cm.rainbow(int(255/9 * s))  # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区间
        plt.text(x, y, s, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer')
    plt.show()


def vectors_vision(tensors):
    data = tensors.data.numpy()
    tsne = TSNE(n_components=2)
    results = tsne.fit_transform(data) # 降维后的数据
    labels = [i+1 for i in range(len(data))]
    plot_with_labels(results, labels)



