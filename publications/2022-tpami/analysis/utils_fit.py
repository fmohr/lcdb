import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import operator
import math
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare
import networkx

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'
curve_models = ['last1', 'pow2', 'log2', 'exp2', 'lin2', 'ilog2', 'pow3', 'exp3', 'vap3', 'expp3', 'expd3', 'logpower3',
                'mmf4', 'wbl4', 'exp4', 'pow4']


def get_num_par(model_id):
    if model_id == 'last1':
        return 1
    if model_id in ['pow2', 'log2', 'exp2', 'lin2', 'ilog2']:
        return 2
    if model_id in ['pow3', 'exp3', 'vap3', 'expp3', 'expd3', 'logpower3']:
        return 3
    if model_id in ['mmf4', 'wbl4', 'exp4', 'pow4']:
        return 4


def get_fun_model_id(beta, model_id):
    num_par = get_num_par(model_id)
    fun = None

    # unpack parameters
    if num_par == 1:
        a = beta[0]
    if num_par == 2:
        a, b = beta[0], beta[1]
    if num_par == 3:
        a, b, c = beta[0], beta[1], beta[2]
    if num_par == 4:
        a, b, c, d = beta[0], beta[1], beta[2], beta[3]

    # define curve models
    if model_id == 'pow2':
        fun = lambda x: -a * x ** (-b)
    if model_id == 'pow3':
        fun = lambda x: a - b * x ** (-c)
    if model_id == 'log2':
        fun = lambda x: -a * np.log(x) + b
    if model_id == 'exp3':
        fun = lambda x: a * np.exp(-b * x) + c
    if model_id == 'exp2':
        fun = lambda x: a * np.exp(-b * x)
    if model_id == 'lin2':
        fun = lambda x: a * x + b
    if model_id == 'vap3':
        fun = lambda x: np.exp(a + b / x + c * np.log(x))
    if model_id == 'mmf4':
        fun = lambda x: (a * b + c * x ** d) / (b + x ** d)
    if model_id == 'wbl4':
        fun = lambda x: (c - b * np.exp(-a * (x ** d)))
    if model_id == 'exp4':
        fun = lambda x: c - np.exp(-a * (x ** d) + b)
    if model_id == 'expp3':
        # fun = lambda x: a * np.exp(-b*x) + c
        fun = lambda x: c - np.exp((x - b) ** a)
    if model_id == 'pow4':
        fun = lambda x: a - b * (x + d) ** (-c)  # has to closely match pow3
    if model_id == 'ilog2':
        fun = lambda x: b - (a / np.log(x))
    if model_id == 'expd3':
        fun = lambda x: c - (c - a) * np.exp(-b * x)
    if model_id == 'logpower3':
        fun = lambda x: a / (1 + (x / np.exp(b)) ** c)
    if model_id == 'last1':
        fun = lambda x: (a + x) - x  # casts the prediction to have the correct size
    return fun


def get_info_mean_curve(df_info, openmlid, learner):
    q = df_info.query('openmlid==@openmlid and learner==@learner')
    q = q.iloc[0, :]
    return [q.anchor_prediction, q.score]


def get_anchors_and_scores_mean_curve(df):
    rows = []
    for openmlid, df_dataset in tqdm(df.groupby("openmlid")):
        for learner, df_learner in df_dataset.groupby("learner"):
            sizes = sorted(pd.unique(df_learner["size_train"]))
            scores = []
            for (inner, outer), df_seeded in df_learner.groupby(["inner_seed", "outer_seed"]):
                sizes_seed, scores_seed = df_seeded["size_train"].values, df_seeded["score_valid"].values
                scores.append([scores_seed[list(sizes_seed).index(s)] if s in sizes_seed else np.nan for s in sizes])
            scores = np.array(scores)
            mean_scores = np.nanmean(scores, axis=0)
            rows.append([openmlid, learner, sizes, mean_scores])
    return pd.DataFrame(rows, columns=["openmlid", "learner", "anchor_prediction", "score"])


def get_info_mean_curve(df_info, openmlid, learner):
    q = df_info.query('openmlid==@openmlid and learner==@learner')
    q = q.iloc[0, :]
    return [q.anchor_prediction, q.score]


# Author: Hassan Ismail Fawaz <hassan.ismail-fawaz@uha.fr>
#         Germain Forestier <germain.forestier@uha.fr>
#         Jonathan Weber <jonathan.weber@uha.fr>
#         Lhassane Idoumghar <lhassane.idoumghar@uha.fr>
#         Pierre-Alain Muller <pierre-alain.muller@uha.fr>
# License: GPL3
def determine_plotting(p_values, average_ranks):
    methods = []
    for p in p_values:
        if p[3] == False:
            methods.append(p[0])
            methods.append(p[1])
    methods = np.unique(np.array(methods))

    visible = []
    for m in methods:
        visible.append(average_ranks[m])

    visible = np.array(visible)
    if len(visible) == 0:
        lowv = None
        highv = None
        dontplot = []
        return lowv, highv, dontplot
    lowv = np.floor(np.min(visible))
    highv = np.ceil(np.max(visible))

    dontplot = []
    for method in average_ranks.keys():
        rank = average_ranks[method]
        if rank > highv:
            dontplot.append(method)
        if rank < lowv:
            dontplot.append(method)

    return int(lowv), int(highv), dontplot


# inspired from orange3 https://docs.orange.biolab.si/3/data-mining-library/reference/evaluation.cd.html
def graph_ranks(avranks, names, p_values, cd=None, cdmethod=None, lowv=5, highv=15,
                width=100, textspace=1, reverse=True, filename=None, labels=False, dpi=10, dontplot=[], **kwargs):
    """
    Draws a CD graph, which is used to display  the differences in methods'
    performance. See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.

    Needs matplotlib to work.

    The image is ploted on `plt` imported using
    `import matplotlib.pyplot as plt`.

    Args:
        avranks (list of float): average ranks of methods.
        names (list of str): names of methods.
        cd (float): Critical difference used for statistically significance of
            difference between methods.
        cdmethod (int, optional): the method that is compared with other methods
            If omitted, show pairwise comparison of methods
        lowv (int, optional): the lowest shown rank
        highv (int, optional): the highest shown rank
        width (int, optional): default width in inches (default: 6)
        textspace (int, optional): space on figure sides (in inches) for the
            method names (default: 1)
        reverse (bool, optional):  if set to `True`, the lowest rank is on the
            right (default: `False`)
        filename (str, optional): output file name (with extension). If not
            given, the function does not write a file.
        labels (bool, optional): if set to `True`, the calculated avg rank
        values will be displayed
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        raise ImportError("Function graph_ranks requires matplotlib.")

    width = float(width)
    textspace = float(textspace)

    def nth(l, n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(l, n)
        return [a[n] for a in l]

    def lloc(l, n):
        """
        List location in list of list structure.
        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(l[0]) + n
        else:
            return n

    def mxrange(lr):
        """
        Multiple xranges. Can be used to traverse matrices.
        This function is very slow due to unknown number of
        parameters.

        >>> mxrange([3,5])
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        >>> mxrange([[3,5,1],[9,0,-3]])
        [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]

        """
        if not len(lr):
            yield ()
        else:
            # it can work with single numbers
            index = lr[0]
            if isinstance(index, int):
                index = [index]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))

    def print_figure(fig, *args, **kwargs):
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(*args, **kwargs)

    sums = avranks

    nnames = names
    ssums = sums

    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    lines = None

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

    fig = plt.figure(figsize=(width, height), dpi=dpi)
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    hf = 1. / height  # height factor
    wf = 1. / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=2)

    bigtick = 0.3
    smalltick = 0.15
    linewidth = 0.5
    linewidth_sign = 2.0

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2),
              (rankpos(a), cline)],
             linewidth=2)

    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a),
             ha="center", va="bottom", size=16)

    k = len(ssums)

    def filter_names(name):
        return name

    space_between_names = 0.24

    for i in range(math.ceil(k / 2)):
        current_name = filter_names(nnames[i])
        chei = cline + minnotsignificant + i * space_between_names
        if not current_name in dontplot:
            line([(rankpos(ssums[i]), cline),
                  (rankpos(ssums[i]), chei),
                  (textspace - 0.1, chei)],
                 linewidth=linewidth)
        if labels:
            text(textspace + 0.3, chei - 0.075, format(ssums[i], '.2f'), ha="right", va="center", size=10)
        text(textspace - 0.2, chei, filter_names(nnames[i]), ha="right", va="center", size=16)

    for i in range(math.ceil(k / 2), k):
        current_name = filter_names(nnames[i])
        chei = cline + minnotsignificant + (k - i - 1) * space_between_names
        if not current_name in dontplot:
            line([(rankpos(ssums[i]), cline),
                  (rankpos(ssums[i]), chei),
                  (textspace + scalewidth + 0.1, chei)],
                 linewidth=linewidth)
        if labels:
            text(textspace + scalewidth - 0.3, chei - 0.075, format(ssums[i], '.2f'), ha="left", va="center", size=10)
        text(textspace + scalewidth + 0.2, chei, filter_names(nnames[i]),
             ha="left", va="center", size=16)

    # no-significance lines
    def draw_lines(lines, side=0.05, height=0.1):
        start = cline + 0.2

        for l, r in lines:
            line([(rankpos(ssums[l]) - side, start),
                  (rankpos(ssums[r]) + side, start)],
                 linewidth=linewidth_sign)
            start += height
            print('drawing: ', l, r)

    # draw_lines(lines)
    start = cline + 0.2
    side = 0
    height = 0.1

    # draw no significant lines
    # get the cliques
    cliques = form_cliques(p_values, nnames)
    i = 1
    achieved_half = False
    # print(nnames)
    for clq in cliques:
        if len(clq) == 1:
            continue
        # print(clq)
        min_idx = np.array(clq).min()
        max_idx = np.array(clq).max()
        if min_idx >= len(nnames) / 2 and achieved_half == False:
            start = cline + 0.25
            achieved_half = True
        line([(rankpos(ssums[min_idx]) - side, start),
              (rankpos(ssums[max_idx]) + side, start)],
             linewidth=linewidth_sign, color='r')
        start += height


def form_cliques(p_values, nnames):
    """
    This method forms the cliques
    """
    # first form the numpy matrix data
    m = len(nnames)
    g_data = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        if p[3] == False:
            i = np.where(nnames == p[0])[0][0]
            j = np.where(nnames == p[1])[0][0]
            min_i = min(i, j)
            max_j = max(i, j)
            g_data[min_i, max_j] = 1

    g = networkx.Graph(g_data)
    return networkx.find_cliques(g)


def draw_cd_diagram(df_perf=None, alpha=0.05, title=None, labels=False):
    """
    Draws the critical difference diagram given the list of pairwise classifiers that are
    significant or not
    """
    p_values, average_ranks, _ = wilcoxon_holm(df_perf=df_perf, alpha=alpha)

    print(average_ranks)

    for p in p_values:
        print(p)

    graph_ranks(average_ranks.values, average_ranks.keys(), p_values,
                cd=None, reverse=True, width=30, textspace=3, labels=labels)

    font = {'family': 'sans-serif',
            'color': 'black',
            'weight': 'normal',
            'size': 22,
            }
    if title:
        plt.title(title, fontdict=font, y=0.9, x=0.5)
    plt.savefig('cd-diagram.pdf', bbox_inches='tight')

    return average_ranks


def wilcoxon_holm(alpha=0.05, df_perf=None, df_perf_nan=None):
    """
    Applies the wilcoxon signed rank test between each pair of algorithm and then use Holm
    to reject the null's hypothesis
    """

    df_perf_with_nans = df_perf_nan

    # print(pd.unique(df_perf['classifier_name']))
    # count the number of tested datasets per classifier
    df_counts = pd.DataFrame({'count': df_perf.groupby(
        ['classifier_name']).size()}).reset_index()
    # get the maximum number of tested datasets
    max_nb_datasets = df_counts['count'].max()
    # get the list of classifiers who have been tested on nb_max_datasets
    classifiers = list(df_counts.loc[df_counts['count'] == max_nb_datasets]
                       ['classifier_name'])
    # test the null hypothesis using friedman before doing a post-hoc analysis
    friedman_p_value = friedmanchisquare(*(
        np.array(df_perf.loc[df_perf['classifier_name'] == c]['accuracy'])
        for c in classifiers))[1]
    if friedman_p_value >= alpha:
        # then the null hypothesis over the entire classifiers cannot be rejected
        print('the null hypothesis over the entire classifiers cannot be rejected')
        exit()
    # get the number of classifiers
    m = len(classifiers)
    # init array that contains the p-values calculated by the Wilcoxon signed rank test
    p_values = []
    # loop through the algorithms to compare pairwise
    for i in range(m - 1):
        # get the name of classifier one
        classifier_1 = classifiers[i]
        # get the performance of classifier one
        perf_1 = np.array(df_perf_with_nans.loc[df_perf_with_nans['classifier_name'] == classifier_1]['accuracy']
                          , dtype=np.float64)
        for j in range(i + 1, m):
            # get the name of the second classifier
            classifier_2 = classifiers[j]
            # get the performance of classifier one
            perf_2 = np.array(df_perf_with_nans.loc[df_perf_with_nans['classifier_name'] == classifier_2]
                              ['accuracy'], dtype=np.float64)
            # calculate the p_value
            p_value = wilcoxon(perf_1, perf_2, zero_method='pratt', nan_policy='omit')[1]
            # appen to the list
            p_values.append((classifier_1, classifier_2, p_value, False))
    # get the number of hypothesis
    k = len(p_values)
    # sort the list in acsending manner of p-value
    p_values.sort(key=operator.itemgetter(2))

    # loop through the hypothesis
    for i in range(k):
        # correct alpha with holm
        new_alpha = float(alpha / (k - i))
        # test if significant after holm's correction of alpha
        if p_values[i][2] <= new_alpha:
            p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)
        else:
            # stop
            break
    # compute the average ranks to be returned (useful for drawing the cd diagram)
    # sort the dataframe of performances
    sorted_df_perf = df_perf.loc[df_perf['classifier_name'].isin(classifiers)]. \
        sort_values(['classifier_name', 'dataset_name'])
    # get the rank data
    rank_data = np.array(sorted_df_perf['accuracy']).reshape(m, max_nb_datasets)

    # create the data frame containg the accuracies
    df_ranks = pd.DataFrame(data=rank_data, index=np.sort(classifiers), columns=
    np.unique(sorted_df_perf['dataset_name']))

    # number of wins
    dfff = df_ranks.rank(ascending=False)
    # print(dfff[dfff == 1.0].sum(axis=1))

    # average the ranks
    average_ranks = df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)
    # return the p-values and the average ranks
    return p_values, average_ranks, max_nb_datasets


#

def remove_fails(df_total):
    fail_init = df_total['fails_init'] > 1000
    fail_fit = df_total['fails_fit'] > 100

    print('fail due to init %d' % fail_init.sum())
    print('fail due to fit %d' % fail_fit.sum())

    fail_ind = pd.concat([fail_init, fail_fit], axis=1)
    fail_ind = fail_ind.any(axis=1)
    print('fail total %d' % sum(fail_ind))

    df_fail = df_total[fail_ind]
    df_total_new = df_total[fail_ind == False]

    return [df_total_new, df_fail]


def remove_nan_and_inf(df_total):
    numeric = df_total.iloc[:, [0, 1, 2, 5, 6, 7, 8, 9]]
    # numeric.describe()

    ind_nan_or_inf = numeric.isin([np.inf, -np.inf, np.nan]).any(axis=1)
    print('number of rows with nans / infs:', ind_nan_or_inf.sum())
    print('columns with nans / infs:')
    print(numeric.isin([np.inf, -np.inf, np.nan]).any())

    return df_total[ind_nan_or_inf == False], df_total[ind_nan_or_inf == True]


def remove_performace_too_bad(df_total):
    numeric = df_total.iloc[:, [0, 1, 2, 5, 6, 7, 8, 9]]

    threshold = 100

    error_MSE_trn = numeric['MSE_trn'] > threshold
    error_MSE_tst = numeric['MSE_tst'] > threshold
    error_MSE_tst_last = numeric['MSE_tst_last'] > threshold
    # error_L1_trn = numeric['L1_trn'] > threshold
    # error_L1_tst = numeric['L1_tst'] > threshold
    # error_L1_tst_last = numeric['L1_tst_last'] > threshold

    errors_fit = pd.concat([error_MSE_trn, error_MSE_tst, error_MSE_tst_last], axis=1)
    print('performance too bad for analysis:')
    print(errors_fit.sum())
    print('number of offending rows:')
    print(errors_fit.any(axis=1).sum())

    errors_ind = errors_fit.any(axis=1)
    return df_total[errors_ind == False], df_total[errors_ind == True]


def convert_table2(df, performance_measure, logscale=False):
    curve_models = df['curve_model'].unique()

    rows = []
    info_rows = []
    for (openmlid, df_dataset) in tqdm(df.groupby('openmlid')):
        for (learner, df_learner) in df_dataset.groupby('learner'):
            for (n, df_n) in df_learner.groupby('n'):
                new_row = []
                bucket = df_n.iloc[0, :].percentage_bucket

                percentage = df_n.iloc[0, :].percentage

                # print(percentage)
                # print(type(percentage))

                info_rows.append([openmlid, learner, n, bucket, float(percentage)])
                for curve_model in curve_models:
                    row = df_n.query('curve_model == @curve_model')
                    score = np.nan
                    if len(row) > 0:
                        row = row.iloc[0, :]
                        score = row[performance_measure]
                    new_row.append(score)
                rows.append(new_row)

    a = np.array(rows)
    if logscale == True:
        a = np.log(a)
    a = pd.DataFrame(a, columns=curve_models)
    a_info = pd.DataFrame(info_rows, columns=['openmlid', 'learner', 'n', 'bucket', 'percentage'])
    b = pd.concat([a_info, a], axis=1)
    return b


def remove_rows_with_nan_or_inf(c):
    c = c.copy()
    ind_nan_or_inf = c.isin([np.inf, -np.inf, np.nan]).any(axis=1)
    c = c[ind_nan_or_inf == False]
    return c


def find_closest_rows(df, percentage_target):
    rows = []
    for (openmlid, df_dataset) in (df.groupby('openmlid')):
        for (learner, df_learner) in df_dataset.groupby('learner'):
            a = np.abs(np.array(df_learner['percentage'].values - percentage_target))
            ind_closest = np.argmin(a)
            rows.append(df_learner.iloc[ind_closest, :])
    a = pd.DataFrame(rows, columns=df.columns)
    return a


def prepare_data_for_cd(b, drop_nan=True):
    if drop_nan == True:
        c = remove_rows_with_nan_or_inf(b)
    else:
        c = b

    melted = pd.melt(c, id_vars=['openmlid', 'learner', 'n'], value_vars=curve_models)
    dataset_name = melted.agg('{0[openmlid]},{0[learner]},{0[n]}'.format, axis=1)
    melted['dataset_name'] = dataset_name

    df_cd = melted[['variable', 'dataset_name', 'value']]
    df_cd = df_cd.rename(columns={'variable': 'classifier_name', 'value': 'accuracy'})
    df_cd['accuracy'] = -df_cd['accuracy']
    return df_cd


def compute_ranks(df_cd):
    p_values, average_ranks, _ = wilcoxon_holm(df_perf=df_cd, df_perf_nan=df_cd, alpha=0.05)
    return average_ranks


def cd_plot(df_cd, df_cd_nan=None, lowv=1, highv=16, dontplot=[], title=None, auto=True):
    p_values, average_ranks, _ = wilcoxon_holm(df_perf=df_cd, df_perf_nan=df_cd_nan, alpha=0.05)

    if auto:
        lowv, highv, dontplot = determine_plotting(p_values, average_ranks)

    graph_ranks(average_ranks.values, average_ranks.keys(), p_values,
                cd=None, reverse=True, width=8, textspace=0, labels=True, dpi=600, lowv=lowv, highv=highv,
                dontplot=dontplot)
    font = {'family': 'sans-serif',
            'color': 'black',
            'weight': 'normal',
            'size': 22,
            }
    if title:
        plt.title(title, fontdict=font, y=0.9, x=0.5)
    return lowv, highv, dontplot, average_ranks


def rescale_rank_table(a):
    a = a.copy()
    b = a.iloc[:, [1, 2, 3, 4, 5, 6]] - np.min(a.iloc[:, [1, 2, 3, 4, 5, 6]].values)
    b = b / np.max(b.values)
    b = b * 15
    a.iloc[:, [1, 2, 3, 4, 5, 6]] = b
    return a


def print_pretty_rank_table_transpose(a):
    print('\\begin{table}[]')
    print('\\ttfamily')
    print('\\begin{tabular}{lllllllllllllllll}')
    my_list = ['curve', 'all', '5\%', '10\%', '20\%', '40\%', '80\%']
    a_res = rescale_rank_table(a)
    a_res = a_res.T
    b = a.copy().T
    first = True
    for i in range(0, len(b)):
        row = b.iloc[i, :]
        print(my_list[i] + '&')
        for (j, num) in enumerate(row.values):
            if i == 0:
                if num == 'baseline_last_constant':
                    num = 'last1'
                print('%12s' % num, ' ', end='')
            else:
                rank = num
                rank_res = a_res.iloc[i, j]
                print('\\cellcolor{r%d}{%.2f} ' % (rank_res, rank), end='')

            if j < 15:
                print('&', end='')
        print('\\\\')
    print('\\end{tabular}')
    print('\\end{table}')


def print_pretty_rank_table_transpose_trn_tst_all(a_trn, a_tst_all):
    print('\\begin{table}[]')
    print('\\ttfamily')
    print('\\begin{tabular}{lllllllllllllllll}')
    my_list = ['curve', 'all', '5\%', '10\%', '20\%', '40\%', '80\%']

    a_trn_res = rescale_rank_table(a_trn)
    a_tst_all_res = rescale_rank_table(a_tst_all)

    a_trn_res = a_trn_res.T
    a_tst_all_res = a_tst_all_res.T

    b_trn = a_trn_res.copy().T
    b_tst_all_res = a_tst_all_res.copy().T
    first = True
    for i in range(0, len(b_tst_all_res)):
        row = b_tst_all_res.iloc[i, :]
        print(my_list[i] + '&')
        for (j, num) in enumerate(row.values):
            if i == 0:
                if num == 'baseline_last_constant':
                    num = 'last1'
                print('%12s' % num, ' ', end='')
            else:
                rank = num
                rank_res = a_tst_all_res.iloc[i, j]
                print('\\cellcolor{r%d}{%.2f} ' % (rank_res, rank), end='')

            if j < 15:
                print('&', end='')
        print('\\\\')

    row = b_trn.iloc[1, :]
    print('trn all' + '&')
    for (j, num) in enumerate(row.values):
        rank = num
        rank_res = a_trn.iloc[i, j]
        print('\\cellcolor{r%d}{%.2f} ' % (rank_res, rank), end='')

        if j < 15:
            print('&', end='')
    print('\\\\')
    print('\\end{tabular}')
    print('\\end{table}')


def print_pretty_rank_table(a):
    print('\\begin{table}[]')
    print('\\ttfamily')
    print('\\begin{tabular}{lllllll}')
    print('curve & all & 5\% & 10\% & 20\% & 40\% & 80\% \\\\ ')
    a_res = rescale_rank_table(a)
    first = True
    for i in range(0, len(a)):
        row = a.iloc[i, :]
        for (j, num) in enumerate(row.values):
            if j == 0:
                if num == 'baseline_last_constant':
                    num = 'last1'
                print('%12s' % num, ' ', end='')
            else:
                rank = num
                rank_res = a_res.iloc[i, j]
                print('\\cellcolor{r%d}{%.2f} ' % (rank_res, rank), end='')

            if j < 6:
                print('&', end='')
        print('\\\\')
    print('\\end{tabular}')
    print('\\end{table}')


def filter_table(table):
    table_list = []
    percentages = [0.05, 0.1, 0.2, 0.4, 0.8]
    for p in percentages:
        table_filtered = find_closest_rows(table, p)
        table_list.append(table_filtered)
    return table_list


def convert_to_cd_tables(table_list):
    table_cd_list = []
    table_cd_list_nan = []
    for table in table_list:
        table_cd_list.append(prepare_data_for_cd(table, drop_nan=True))
        table_cd_list_nan.append(prepare_data_for_cd(table, drop_nan=False))
    return [table_cd_list, table_cd_list_nan]


def build_rank_table(rank_list):
    rows = []

    for ranks in rank_list:

        row_tmp = []
        for cm in curve_models:
            row_tmp.append(ranks[cm])
        rows.append(row_tmp)

    rows = np.array(rows)
    rows = rows.T
    a = pd.DataFrame(rows, columns=['all', '5%', '10%', '20%', '40%', '80%'])
    a.insert(0, 'model', curve_models)
    return a


def make_all_cd_plots(tables, tables_nan, title, ext):
    titles = ['all', '5%', '10%', '20%', '40%', '80%']
    plots = []
    for i in range(0, len(tables)):
        fn = title + ' ' + titles[i] + ext
        fn = fn.replace(' ', '_')
        cd_plot(tables[i], tables_nan[i], title=(title + ' ' + titles[i]))
        plt.savefig(fn, bbox_inches='tight')


def get_ranks_from_tables(tables_list):
    rank_list = []
    for table in tables_list:
        rank_list.append(compute_ranks(table))
    return rank_list


def from_tables_print_pretty_rank_table(tables):
    rank_list = get_ranks_from_tables(tables)
    rank_table = build_rank_table(rank_list)
    print_pretty_rank_table(rank_table)


def get_XY(row, df_anchors_and_scores):
    learner = row.learner
    openmlid = row.openmlid
    [X, Y] = get_info_mean_curve(df_anchors_and_scores, openmlid, learner)
    return [X, Y]


def get_XY2(row):
    X = row.anchor_prediction
    Y = row.score
    return [X, Y]


def set_ylim(row, df_anchors_and_scores, margin=0.05):
    [X, Y] = get_XY(row, df_anchors_and_scores)
    Y = 1 - Y
    Y_diff = np.max(Y) - np.min(Y)
    plt.ylim([np.min(Y), np.max(Y)])
    # plt.ylim([np.min(Y) - Y_diff*margin,np.max(Y) + Y_diff*margin])


def set_ylim2(row, margin=0.05):
    [X, Y] = get_XY2(row)
    Y = 1 - Y
    Y_diff = np.max(Y) - np.min(Y)
    plt.ylim([np.min(Y), np.max(Y)])
    # plt.ylim([np.min(Y) - Y_diff*margin,np.max(Y) + Y_diff*margin])


def plot_data(row, df_anchors_and_scores):
    [X, Y] = get_XY(row, df_anchors_and_scores)
    Y = 1 - Y

    plt.plot(X, Y, '*r', label='test anchors')
    set_ylim(row, df_anchors_and_scores)
    plt.xlabel('training set size')
    plt.ylabel('error rate')

    learner = row.learner
    openmlid = row.openmlid
    plt.title('%s dataset %d' % (learner, openmlid))


def plot_data2(row):
    [X, Y] = get_XY2(row)
    Y = 1 - Y

    plt.plot(X, Y, '*r', label='test anchors')
    set_ylim2(row)
    plt.xlabel('training set size')
    plt.ylabel('error rate')

    learner = row.learner
    openmlid = row.openmlid
    plt.title('%s dataset %d' % (learner, openmlid))


def plot_trn_data(row, df_anchors_and_scores):
    [X, Y] = get_XY(row, df_anchors_and_scores)
    Y = 1 - Y

    offset = np.argwhere(X == row.max_anchor_seen)[0][0]

    X_trn = X[:offset + 1]
    Y_trn = Y[:offset + 1]

    plt.plot(X_trn, Y_trn, 'ob', label='train anchors')
    set_ylim(row, df_anchors_and_scores)
    plt.xlabel('training set size')
    plt.ylabel('error rate')

    learner = row.learner
    openmlid = row.openmlid
    plt.title('%s dataset %d' % (learner, openmlid))


def plot_trn_data2(row):
    [X, Y] = get_XY2(row)
    Y = 1 - Y

    offset = np.argwhere(X == row.max_anchor_seen)[0][0]

    X_trn = X[:offset + 1]
    Y_trn = Y[:offset + 1]

    plt.plot(X_trn, Y_trn, 'ob', label='train anchors')
    set_ylim2(row)
    plt.xlabel('training set size')
    plt.ylabel('error rate')

    learner = row.learner
    openmlid = row.openmlid
    plt.title('%s dataset %d' % (learner, openmlid))


def get_curve_models(df, row):
    learner = row.learner
    openmlid = row.openmlid
    max_anchor_seen = row.max_anchor_seen

    df_models = df.query('learner == "%s" and openmlid == @openmlid and max_anchor_seen == @max_anchor_seen' % learner)

    return df_models


def get_curve_model(df, row, cm):
    df_models = get_curve_models(df, row)
    df_models2 = df_models.query('curve_model == "%s"' % cm)
    my_row = df_models2.iloc[0, :]
    return my_row


def plot_prediction(row, df_anchors_and_scores):
    curve_model = row.curve_model
    [X, Y] = get_XY(row, df_anchors_and_scores)
    Y = 1 - Y

    plt.plot(X, 1 - row.prediction, ':', label=curve_model)
    plt.legend()


def plot_prediction2(row):
    curve_model = row.curve_model
    [X, Y] = get_XY2(row)
    Y = 1 - Y

    plt.plot(X, 1 - row.prediction, ':', label=curve_model)
    plt.legend()


def plot_prediction_smooth(row, df_anchors_and_scores):
    curve_model = row.curve_model

    [X, Y] = get_XY(row, df_anchors_and_scores)
    Y = 1 - Y

    fun = get_fun_model_id(row.beta, curve_model)

    X_plot = np.arange(np.min(X), np.max(X))
    Y_hat = 1 - fun(X_plot)

    plt.plot(X_plot, Y_hat, '-', label=curve_model)
    plt.legend()


def plot_prediction_smooth2(row):
    curve_model = row.curve_model

    [X, Y] = get_XY2(row)
    Y = 1 - Y

    fun = get_fun_model_id(row.beta, curve_model)

    X_plot = np.arange(np.min(X), np.max(X))
    Y_hat = 1 - fun(X_plot)

    plt.plot(X_plot, Y_hat, '-', label=curve_model)
    plt.legend()


def load_from_parts():
    from os.path import exists
    df = pd.read_csv('database-accuracy.csv')
    fn_anchors_scores = 'anchor-accuracy.gz'
    if exists(fn_anchors_scores):
        df_anchors_and_scores = pd.read_pickle(fn_anchors_scores)
    else:
        df_anchors_and_scores = get_anchors_and_scores_mean_curve(df)
        df_anchors_and_scores.to_pickle(fn_anchors_scores, protocol=3)

    df_metrics_list = []
    df_extrapolations_list = []
    for i in range(0, 13):
        df_metrics_list.append(pd.read_pickle('metrics%d.gz' % i))
        df_extrapolations_list.append(pd.read_pickle('extrapolations%d.gz' % i))

    df_metrics = pd.concat(df_metrics_list, axis=0)
    df_extrapolations = pd.concat(df_extrapolations_list, axis=0)
    assert (len(df_metrics) == len(df_extrapolations))

    return [df_anchors_and_scores, df_metrics, df_extrapolations]


def remove_bad_fits(df_total2):
    # remove failed fits:
    # that either took more than 100 iterations to come to a reasonable solution
    # or that needed more than 1000 inits to find a reasonable initialisation
    [df_total_no_fail, df_fail] = remove_fails(df_total2)

    # check that removed correctly
    assert (len(df_total_no_fail) + len(df_fail) == len(df_total2))
    assert ((df_total_no_fail['fails_fit'] > 100).sum() == 0)  # check that indeed we got rid of them

    # remove rows with nan or inf for ANY L2 (trn, tst, tst last) loss
    df_no_fail_no_nan_or_inf, df_nan_or_inf = remove_nan_and_inf(df_total_no_fail)
    assert (len(df_no_fail_no_nan_or_inf) + len(df_nan_or_inf) == len(df_total_no_fail))

    # check that removed correctly
    numeric = df_no_fail_no_nan_or_inf.iloc[:, [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
    numeric = numeric.isin([np.inf, -np.inf, np.nan]).any(axis=1)
    assert (numeric.sum() == 0)

    # remove fits that have a MSE (trn, tst or test last) that is larger than 100
    df_no_fail_no_nan_or_inf_no_too_bad, df_too_bad = remove_performace_too_bad(df_no_fail_no_nan_or_inf)
    assert (len(df_no_fail_no_nan_or_inf_no_too_bad) + len(df_too_bad) == len(df_no_fail_no_nan_or_inf))

    # check that removed correctly
    numeric = df_no_fail_no_nan_or_inf_no_too_bad.iloc[:, [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
    threshold = 100

    error_MSE_trn = numeric['MSE_trn'] > threshold
    error_MSE_tst = numeric['MSE_tst'] > threshold
    error_MSE_tst_last = numeric['MSE_tst_last'] > threshold

    errors_fit = pd.concat([error_MSE_trn, error_MSE_tst, error_MSE_tst_last], axis=1)
    errors_ind = errors_fit.any(axis=1)
    assert (errors_ind.sum() == 0)

    # collect all that were removed:
    reason = ['fail'] * len(df_fail)
    df_fail2 = df_fail.copy()
    df_fail2.insert(0, 'reason', reason)
    reason = ['nan_or_inf'] * len(df_nan_or_inf)
    df_nan_or_inf2 = df_nan_or_inf.copy()
    df_nan_or_inf2.insert(0, 'reason', reason)
    reason = ['too_bad'] * len(df_too_bad)
    df_too_bad2 = df_too_bad.copy()
    df_too_bad2.insert(0, 'reason', reason)

    df_removed = pd.concat([df_fail2, df_nan_or_inf2, df_too_bad2], axis=0)
    df_clean = df_no_fail_no_nan_or_inf_no_too_bad

    return [df_clean, df_removed]


def recompute_percentages(df, df_anchors_and_scores):
    list_temp = []
    for (openmlid, df_dataset) in tqdm(df.groupby('openmlid')):
        for (learner, df_learner) in df_dataset.groupby('learner'):
            anchor_prediction, score = get_info_mean_curve(df_anchors_and_scores, openmlid, learner)
            max_anchor = np.max(anchor_prediction)
            new_percentage = df_learner['max_anchor_seen'].values / max_anchor
            df_learner['percentage'] = new_percentage
            list_temp.append(df_learner)

    df_new_percentage = pd.concat(list_temp, axis=0)

    return df_new_percentage


def failed_fits_statistics(df):
    rows = []
    for [curve_model, df_removed_curve] in df.groupby('curve_model'):
        too_bad = (df_removed_curve['reason'] == 'too_bad').sum()
        nan_or_inf = (df_removed_curve['reason'] == 'nan_or_inf').sum()
        fail = (df_removed_curve['reason'] == 'fail').sum()
        total = len(df_removed_curve)
        rows.append([curve_model, fail, nan_or_inf, too_bad, total])
    failfits = pd.DataFrame(rows, columns=['curve model', 'fail', 'nan or inf', 'too bad', 'total'])
    row_total = failfits.iloc[:, [1, 2, 3, 4]].sum()
    total = list(row_total)
    rows.append(['all', total[0], total[1], total[2], total[3]])
    failfits = pd.DataFrame(rows, columns=['curve_model', 'fail', 'nan_or_inf', 'too_bad', 'total'])
    return failfits


def create_percentage_buckets(df):
    percentage_buckets = [1, 0.8, 0.4, 0.2, 0.1, 0.05]
    percentage_buckets = np.array(percentage_buckets)

    bucket_list = [np.nan] * (len(df))
    bucket_list = np.array(bucket_list)

    for i in range(0, len(percentage_buckets)):
        p = percentage_buckets[i]
        inbucket = df['percentage'] < p
        bucket_list = np.where(inbucket.values, p, bucket_list)

    df_buckets = df.copy()
    df_buckets.insert(len(df.columns), 'percentage_bucket', bucket_list)
    # df_clean_buckets['percentage']

    return df_buckets


def prepare_total_dataframe(df_anchors_and_scores, df_metrics, df_extrapolations):
    df_extrapolations_no_curve_model = df_extrapolations.loc[:, df_extrapolations.columns != 'curve_model']
    df_total = pd.concat([df_extrapolations_no_curve_model, df_metrics], axis=1)
    df_total = df_total.rename(
        columns={'MSE trn': 'MSE_trn', 'MSE tst': 'MSE_tst', 'MSE tst last': 'MSE_tst_last', 'L1 trn': 'L1_trn',
                 'L1 tst': 'L1_tst', 'L1 tst last': 'L1_tst_last'})

    df_total2 = df_total.copy()

    df_total2 = df_total2.join(df_anchors_and_scores.set_index(['openmlid', 'learner']), on=['openmlid', 'learner'],
                               how='left')

    assert (len(df_total) == len(df_total2))

    df_total2_without_duplicate_columns = df_total2.drop(['percentage', 'max anchor seen'], axis=1)

    df_total3 = recompute_percentages(df_total2_without_duplicate_columns, df_anchors_and_scores)

    df_total4 = create_percentage_buckets(df_total3)

    return df_total4


def plot_metric(series, my_metric, ls='-', lw=1):
    curbin = 0.0000000005
    binlist = [0]
    while curbin < 1000000:
        binlist.append(curbin)
        curbin *= 2
    [hist, edges] = np.histogram(series, bins=binlist)
    plt.plot(edges[:-1], hist / np.sum(hist), ls, label=my_metric, linewidth=lw)
    plt.xscale('log')


def empirical_cdf(ax, series, my_metric, ls='-', label='', linewidth=1):
    a = series.values
    ax.plot(np.sort(a), np.linspace(0, 1, len(a), endpoint=False), ls, label=label, linewidth=linewidth)


def get_relevant_max_anchor(max_available_anchor):
    if max_available_anchor < 128:
        return max_available_anchor / 2
    sizes = np.array([2 ** int(np.ceil(k / 2)) for k in range(7, 100)])
    highest_anchor = sizes[max(np.where(sizes <= 0.2 * max_available_anchor)[0])]
    return highest_anchor
