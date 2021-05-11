import numpy as np
import pandas as pd
import seaborn as sns
from IPython.core.display import display
from matplotlib import pyplot as plt


def apk(actual, predicted, k=10):
    """
    #https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    Computes the average precision at k.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    #https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    Computes the mean average precision at k.
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def precision_recall_k(y_true, y_pred, k=10):
    precisions = []
    recalls = []
    for y_t, y_p in zip(y_true, y_pred):
        y_p = set(y_p[:k])
        relevant_retrieved = set(y_t).intersection(y_p)
        if relevant_retrieved:
            precision = len(relevant_retrieved) / len(y_p)
            recall = len(relevant_retrieved) / len(y_t)
        else:
            precision = 0.0
            recall = 0.0
        precisions.append(precision)
        recalls.append(recall)
    return np.mean(precisions), np.mean(recalls)


def mean_citation_similarity(sample_cases, preds):
    all_sims = []
    for item in sample_cases["citation_idx"]:
        for idx, sim in zip(item, preds):
            mean_sim = sim[idx].mean()
            all_sims.append(mean_sim)
    all_sims = np.mean(all_sims)
    return all_sims


def load_map(actual, predicted, k_range=range(1, 17, 3)):
    scores = []
    for k in k_range:
        scores.append(mapk(actual, predicted, k))
    return scores


def load_pk(actual, predicted, k_range=range(1, 17, 3)):
    scores = []
    for k in k_range:
        scores.append(precision_recall_k(actual, predicted, k))
    return scores


def plot_score(plot_df, title=""):
    display(plot_df.reset_index().rename({"index": "top-k"}, axis=1))
    plt.figure(figsize=(20, 10))
    sns.lineplot(data=plot_df, markers=True)
    plt.title(
        f"{title}", fontsize=20, ha="center",
    )
    plt.xlabel("top-k")
    sns.despine()
    plt.show()


def plot_scores(
    val_scores,
    test_scores,
    col_names=("val_map", "test_map"),
    index=range(1, 17, 3),
    title="",
):
    plot_df = pd.DataFrame(
        list(zip(val_scores, test_scores)), columns=col_names, index=index
    )
    display(plot_df.reset_index().rename({"index": "top-k"}, axis=1))
    plt.figure(figsize=(20, 10))
    sns.lineplot(data=plot_df, markers=True)
    plt.title(
        f"{title}", fontsize=20, ha="center",
    )
    plt.xlabel("top-k")
    sns.despine()
    plt.show()


def plot_pk_scores(
    val_scores,
    test_scores,
    col_names=("val_map", "test_map"),
    index=range(1, 17, 3),
    title="",
):
    val_precisions, val_recalls = zip(*val_scores)
    test_precisions, test_recalls = zip(*test_scores)

    plot_df = pd.DataFrame(
        list(zip(val_precisions, val_recalls, test_precisions, test_recalls)),
        columns=col_names,
        index=index,
    )
    display(plot_df.reset_index().rename({"index": "top-k"}, axis=1))
    plt.figure(figsize=(20, 10))
    sns.lineplot(data=plot_df, markers=True)
    plt.title(
        f"{title}", fontsize=20, ha="center",
    )
    plt.xlabel("top-k")
    sns.despine()
    plt.show()
