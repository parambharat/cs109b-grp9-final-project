import csv
import gzip
import json
import lzma
import os
import random
import zipfile
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from IPython.core.display import display
from gensim.parsing.preprocessing import DEFAULT_FILTERS, preprocess_string
from pandas import option_context
from tqdm import tqdm_notebook


from functools import partial
from multiprocessing import Pool


def load_cases(f_names, field=None):
    cases = []
    for f_name in f_names:
        with zipfile.ZipFile(f_name, "r") as zip_archive:
            xz_path = next(
                path
                for path in zip_archive.namelist()
                if path.endswith("/data.jsonl.xz")
            )
            with zip_archive.open(xz_path) as xz_archive, lzma.open(
                xz_archive
            ) as json_lines:
                for i, line in enumerate(json_lines):
                    record = json.loads(str(line, "utf-8"))
                    if field:
                        record = {field: record[field]}
                    cases.append(record)
        print(f"Loaded {i + 1} cases from {f_name.split('/')[-1]}")
    return pd.DataFrame(cases)


def read_citation_graph(f_name, case_ids):
    csvobj = csv.reader(gzip.open(f_name, mode="rt"), delimiter=",", quotechar="'")
    graph = []
    for item in csvobj:
        head = item[0]
        try:
            head = int(head)
        except ValueError:
            head = None
        if head in case_ids:
            graph.append(item)
    citation_graph = pd.DataFrame(graph)
    citation_graph = citation_graph.set_index(0)
    citation_graph = citation_graph.apply(lambda x: x.dropna().tolist(), axis=1)
    return citation_graph


def read_citiation_metadata(f_name, case_ids):
    csvobj = csv.DictReader(gzip.open(f_name, mode="rt"), delimiter=",")
    graph_meta = []
    for item in csvobj:
        head = item["id"]
        try:
            head = int(head)
        except ValueError:
            head = None
        if head in case_ids:
            graph_meta.append(item)
    graph_meta = pd.DataFrame(graph_meta).set_index("id")
    return graph_meta


def show_cases_per_reporter(cases_data, reporters_data):
    cases_by_reporters = cases_data.reporter_id.value_counts()
    reporter_names = (
        reporters_data.set_index("id")
        .loc[cases_by_reporters.index, "full_name"]
        .tolist()
    )
    cases_by_reporters = cases_by_reporters.reset_index()
    cases_by_reporters["reporters"] = reporter_names
    cases_by_reporters = cases_by_reporters.rename(
        {"reporter_id": "count", "index": "reporter_id"}, axis=1
    )
    cases_by_reporters["reporter_id"] = cases_by_reporters["reporter_id"].map(str)
    plt.figure(figsize=(20, 10))
    sns.barplot(
        data=cases_by_reporters,
        x="reporter_id",
        y="count",
        hue="reporters",
        dodge=False,
    )
    plt.xticks(None, rotation=45)
    plt.suptitle("Count of cases per reporters in the dataset")
    sns.despine()
    plt.show()


def show_cases_per_court(cases_data, courts_data):
    cases_by_courts = cases_data.court_id.value_counts()
    court_names = (
        courts_data.set_index("id").loc[cases_by_courts.index, "name"].tolist()
    )

    cases_by_courts = cases_by_courts.reset_index()
    cases_by_courts["court"] = court_names
    cases_by_courts = cases_by_courts.rename(
        {"court_id": "count", "index": "court_id"}, axis=1
    )
    cases_by_courts["court_id"] = cases_by_courts["court_id"].map(str)
    plt.figure(figsize=(20, 10))
    sns.barplot(data=cases_by_courts, x="court_id", y="count", hue="court", dodge=False)

    plt.suptitle("Count of cases per court in the dataset", fontsize=20, ha="center")
    plt.xticks(rotation=45)
    sns.despine()
    plt.show()


def show_cases_per_jurisdiction(cases_data, jurisdictions_data):
    cases_by_jurisdictions = cases_data.jurisdiction_id.value_counts()
    jurisdiction_names = (
        jurisdictions_data.set_index("id")
        .loc[cases_by_jurisdictions.index, "name"]
        .tolist()
    )

    cases_by_jurisdictions = cases_by_jurisdictions.reset_index()
    cases_by_jurisdictions["jurisdiction"] = jurisdiction_names
    cases_by_jurisdictions = cases_by_jurisdictions.rename(
        {"jurisdiction_id": "count", "index": "jurisdiction_id"}, axis=1
    )
    cases_by_jurisdictions["jurisdiction_id"] = cases_by_jurisdictions[
        "jurisdiction_id"
    ].map(str)
    plt.figure(figsize=(15, 5))
    sns.barplot(
        data=cases_by_jurisdictions,
        x="jurisdiction_id",
        y="count",
        hue="jurisdiction",
        dodge=False,
    )

    plt.suptitle(
        "Count of cases per jurisdiction in the dataset", fontsize=20, ha="center"
    )
    plt.xticks(rotation=45)
    sns.despine()
    plt.show()


def show_citation_lengths(cases_data):
    plt.figure(figsize=(20, 5))
    sns.distplot(cases_data.citation_ids.map(len), kde=False)
    plt.suptitle(
        "Distibution of citations from each case in the dataset",
        fontsize=20,
        ha="center",
    )
    plt.xlabel("Number of citations")
    plt.ylabel("Number of cases referring to citations")
    plt.show()
    print("\nSummary statistics of citations from each caselaw\n")
    display(pd.DataFrame(cases_data.citation_ids.map(len).describe()).T)


def show_cases_citing_most(citations_data):
    top_cited = citations_data.src.value_counts().head(100)
    plt.figure(figsize=(20, 10))
    ax = sns.barplot(top_cited.index, top_cited.values, order=top_cited.index,)
    ax.set(xlabel="Case ID", ylabel="Count")
    plt.xticks(rotation=90)
    plt.suptitle(
        "Top 100 cases containing the most citations", fontsize=20, ha="center"
    )
    plt.show()
    print()


def show_cases_per_year(cases_data):
    plt.figure(figsize=(20, 10))
    sns.distplot(
        pd.DatetimeIndex(cases_data.decision_date).year, kde=False,
    )
    plt.title("Distribution of caselaws over time in years", fontsize=20, ha="center")
    sns.despine()
    plt.show()


def show_text_col_lengths(cases_data, text_col):
    plt.figure(figsize=(20, 5))
    sns.distplot(cases_data[text_col].map(len), bins=100, kde=False)
    plt.suptitle(
        f"Distribution of character lengths of case {text_col}",
        fontsize=20,
        ha="center",
    )
    sns.despine()
    plt.show()
    print(f"\nSummary statistics for {text_col} lengths\n")
    display(pd.DataFrame(cases_data[text_col].map(len).describe()).T)
    print()


def load_and_display_dates(cases_data):
    # convert date column
    print("\nParsing dates from columns: 'decision_date'")
    print("Found record with date error")
    display(cases_data[cases_data.decision_date == "1914-02-29"])
    print()
    # convert date column
    cases_data["decision_date"] = pd.to_datetime(
        cases_data["decision_date"], format="%Y-%m-%d", errors="coerce"
    )
    cases_data = cases_data[cases_data["decision_date"].notnull()]
    show_cases_per_year(cases_data)
    return cases_data


def load_volumes(cases_data, subset_dir, save_files=False):
    volumes_data = pd.DataFrame(cases_data.volume.tolist(), index=cases_data.index)
    volumes_data.loc[:, "volume_number"] = volumes_data["volume_number"].astype(int)
    cases_data.loc[:, "volume_id"] = volumes_data["volume_number"]
    volumes_data = volumes_data.drop_duplicates("volume_number")

    if save_files:
        volumes_data.to_csv(
            f"{subset_dir}/volumes.csv",
            index=False,
            index_label=False,
            quoting=csv.QUOTE_ALL,
            quotechar='"',
        )
    return cases_data, volumes_data


def load_and_display_reporters(cases_data, subset_dir, save_files):
    print()
    # extract and normalize reporters
    reporters_data = pd.DataFrame(
        cases_data["reporter"].tolist(), index=cases_data.index
    )
    reporters_data.loc[:, "id"] = reporters_data["id"].astype(int)
    cases_data.loc[:, "reporter_id"] = reporters_data["id"]
    reporters_data = reporters_data.drop_duplicates("id")
    if save_files:
        reporters_data.to_csv(
            f"{subset_dir}/reporters.csv",
            index=False,
            index_label=False,
            quoting=csv.QUOTE_ALL,
            quotechar='"',
        )
    show_cases_per_reporter(cases_data, reporters_data)
    print()
    return cases_data, reporters_data


def load_and_display_courts(cases_data, subset_dir, save_files):
    print()
    # extract and normalize courts
    courts_data = pd.DataFrame(cases_data["court"].tolist(), index=cases_data.index)
    courts_data.loc[:, "id"] = courts_data["id"].astype(int)
    cases_data.loc[:, "court_id"] = courts_data["id"]
    courts_data = courts_data.drop_duplicates("id")
    if save_files:
        courts_data.to_csv(
            f"{subset_dir}/courts.csv",
            index=False,
            index_label=False,
            quoting=csv.QUOTE_ALL,
            quotechar='"',
        )
    show_cases_per_court(cases_data, courts_data)
    print()
    return cases_data, courts_data


def load_and_display_jurisdictions(cases_data, subset_dir, save_files):
    print()
    # extract and normalize jurisdictions
    jurisdictions_data = pd.DataFrame(
        cases_data["jurisdiction"].tolist(), index=cases_data.index
    )
    jurisdictions_data.loc[:, "id"] = jurisdictions_data["id"].astype(int)
    cases_data.loc[:, "jurisdiction_id"] = jurisdictions_data["id"]
    jurisdictions_data = jurisdictions_data.drop_duplicates("id")
    if save_files:
        jurisdictions_data.to_csv(
            f"{subset_dir}/jurisdictions.csv",
            index=False,
            index_label=False,
            quoting=csv.QUOTE_ALL,
            quotechar='"',
        )
    show_cases_per_jurisdiction(cases_data, jurisdictions_data)
    print()
    return cases_data, jurisdictions_data


def load_and_display_text_cols(cases_data):
    print()
    # extract case opinion and headmatter
    casebody_data = pd.DataFrame(
        cases_data.loc[:, "casebody"].map(lambda x: x.get("data")).tolist(),
        index=cases_data.index,
    )
    cases_data.loc[:, "head_matter"] = casebody_data.loc[:, "head_matter"]
    cases_data.loc[:, "opinion_text"] = casebody_data.loc[:, "opinions"].map(
        lambda x: "\n".join(y.get("text", "") for y in x)
    )
    cases_data["head_matter"] = cases_data["head_matter"]
    cases_data["opinion_text"] = cases_data["opinion_text"]
    show_text_col_lengths(cases_data, "head_matter")
    show_text_col_lengths(cases_data, "opinion_text")
    print()
    return cases_data


def load_and_display_citations(
    cases_data, citation_graph, subset_dir, save_files=False
):
    # read citation graph and link nodes
    print()
    print(f"reading citation graph from file: {citation_graph}")
    citation_graph = read_citation_graph(
        citation_graph, case_ids=frozenset(cases_data.index)
    )
    print(f"found and loaded {len(citation_graph)} nodes into citation_graph")

    # create a lookup for our cases
    citations_uids = frozenset(citation_graph.index)

    # remove citations that aren't in case data
    citation_graph = citation_graph.loc[:].map(
        lambda x: list(filter(lambda y: y in citations_uids, x))
    )

    # remove cases with no citations after truncation
    citation_graph = citation_graph[citation_graph.map(len) > 0]
    citation_graph.index = citation_graph.index.astype(int)

    cases_data.loc[citation_graph.index, "citation_ids"] = citation_graph.values
    cases_data = cases_data[cases_data.citation_ids.notnull()]

    citations_data = cases_data["citation_ids"].explode().reset_index()
    citations_data.columns = ["src", "dst"]
    if save_files:
        citations_data.to_csv(
            f"{subset_dir}/citations.csv", index=False, index_label=False,
        )

    show_citation_lengths(cases_data)

    show_cases_citing_most(citations_data)
    print()
    return cases_data


def flatten(lst):
    return (item for sublist in lst for item in sublist)


def preprocess_text(cases_data, f_name, overwrite=False):
    if not os.path.isfile(f_name):
        nlp = spacy.load("en_core_web_sm")
        case_text = (
            cases_data["head_matter"]
            + "\n"
            + cases_data["opinions"].map(
                lambda x: " ".join(y.get("text", "") for y in x)
            )
        )
        if overwrite:
            with open(f_name, "w+") as outfile:
                tags = frozenset(["ADJ", "ADV", "NOUN", "PRON", "PROPN"])
                for doc in tqdm_notebook(
                    nlp.pipe(case_text, batch_size=10, n_process=-1),
                    total=len(case_text),
                ):
                    tokens = []
                    for tok in doc:
                        if tok.pos_ in tags and tok.is_alpha:
                            tokens.append(tok.lemma_.lower())
                    tokens = " ".join(tokens)
                    tokens = preprocess_string(tokens, DEFAULT_FILTERS[:-1])
                    outline = " ".join(tokens) + "\n"
                    outfile.write(outline)
    print(f"Loading preprocessed case text from {f_name}")
    case_lines = pd.read_csv(f_name, header=None, names=["text"])
    case_lines.index = cases_data.index
    return case_lines


def show_preprocessed_cases(case_lines):
    # sample preprocessed text
    print("\nSample set of preprocessed case text\n")
    with option_context("display.max_colwidth", 120):
        display(case_lines.head())
    print()


def show_word_count_by_jurisdiction(cases_data, case_lines, jurisdictions_data):
    print()
    grp_vocab = {}
    groups = cases_data.groupby("jurisdiction_id").groups.items()
    for grp, idx in groups:
        grp_data = case_lines.loc[idx]
        top_10 = Counter(flatten(grp_data.text.str.split().tolist())).most_common(10)
        grp_vocab[grp] = top_10
    num_plots = len(grp_vocab)
    fig, ax = plt.subplots(nrows=num_plots, ncols=1, figsize=(20, 20))
    for i, (k, v) in enumerate(grp_vocab.items()):
        title = jurisdictions_data[jurisdictions_data.id == k]["name_long"].values[0]
        word, count = zip(*v)
        sns.barplot(x=np.array(word), y=np.array(count), ax=ax[i])
        ax[i].set_title(f"Jurisdiction:{title}")
    plt.xlabel("Word")
    plt.ylabel("Count")
    plt.suptitle(
        "Top 10 words by Jurisdiction", fontsize=20, ha="center",
    )
    sns.despine()
    plt.show()
    print()


def show_word_count_by_court(cases_data, case_lines, courts_data):
    print()
    grp_vocab = {}
    groups = cases_data.groupby("court_id").groups.items()
    for grp, idx in groups:
        grp_data = case_lines.loc[idx]
        top_10 = Counter(flatten(grp_data.text.str.split().tolist())).most_common(10)
        grp_vocab[grp] = top_10
    num_plots = len(grp_vocab)
    fig, ax = plt.subplots(nrows=num_plots, ncols=1, figsize=(20, 48))
    for i, (k, v) in enumerate(grp_vocab.items()):
        title = courts_data[courts_data.id == k]["name"].values[0]
        word, count = zip(*v)
        sns.barplot(x=np.array(word), y=np.array(count), ax=ax[i])
        ax[i].set_title(f"Court:{title}")
    plt.xlabel("Word")
    plt.ylabel("Count")
    plt.suptitle(
        "Top 10 words by Court", fontsize=20, ha="center",
    )
    sns.despine()
    plt.show()
    print()


def clean_cases(case_list, citation_uids):
    return [int(item) for item in case_list if int(item) in citation_uids]


def case_list2idxlist(case_list, citation_uids, case2idx):
    idx, case_list = case_list
    case_list = clean_cases(case_list, citation_uids)
    if case_list:
        idx_list = [case2idx.get(case) for case in case_list]
        idx_list = list(filter(lambda x: x is not None, idx_list))
        return idx, case_list, idx_list
    else:
        return idx, case_list, case_list


def map_case2idx(cases_data):
    print()
    print("Mapping case_ids to indices")
    case2idx = {v: k for k, v in cases_data["id"].items()}
    citation_uids = frozenset(cases_data["id"].tolist())

    lookup_case2idx = partial(
        case_list2idxlist, citation_uids=citation_uids, case2idx=case2idx
    )

    pool = Pool(14)
    results = pool.imap_unordered(
        lookup_case2idx, cases_data.citation_ids.items(), chunksize=1000
    )

    results = [item for item in tqdm_notebook(results, total=len(cases_data))]

    results = pd.DataFrame(
        results, columns=["idx", "citation_ids", "citation_idx"]
    ).set_index("idx")
    results = results.loc[cases_data.index]
    cases_data.loc[:, "citation_ids"] = results["citation_ids"]
    cases_data.loc[:, "citation_idx"] = results["citation_idx"]
    return case2idx, cases_data


def get_negative_samples(pos_samples, citation_uids):
    idx, pos_samples = pos_samples
    neg_samples = random.sample(
        citation_uids.difference(pos_samples), len(pos_samples) if pos_samples else 1
    )
    return idx, neg_samples


def load_neg_case_idx(cases_df):
    print()
    print("Loading negative case indices")
    citation_uids = frozenset(cases_df.index)
    pos_citations = cases_df["citation_idx"]
    pool = Pool(14)
    sample_negative = partial(get_negative_samples, citation_uids=citation_uids)
    neg_samples = pool.imap_unordered(
        sample_negative, pos_citations.items(), chunksize=2500
    )
    neg_samples = [item for item in tqdm_notebook(neg_samples, total=len(cases_df))]
    neg_samples = pd.DataFrame(neg_samples, columns=["idx", "citation"]).set_index(
        "idx"
    )
    neg_samples = neg_samples.loc[cases_df.index]
    cases_df.loc[:, "neg_citation_idx"] = neg_samples["citation"]
    return cases_df


def truncate_cases_data(cases_data, subset_dir, save_files=False):
    case_info_columns = [
        "id",
        "jurisdiction_id",
        "court_id",
        "decision_date",
        "head_matter",
        "opinion_text",
        "citation_ids",
        "citation_idx",
        "neg_citation_idx",
    ]
    cases_data = cases_data[case_info_columns]

    if save_files:
        cases_data.to_json(f"{subset_dir}/case_info.json", lines=True, orient="records")

    print(f"\nTotal number of cases in the processed dataset : {len(cases_data)}")
    print(
        f"Total number of columns in the processed dataset : {len(cases_data.columns)}\n"
    )
    return cases_data


def make_train_test_splits(cases_data):
    base_data = cases_data.copy()[["id", "citation_idx", "neg_citation_idx"]]
    train_df = base_data.iloc[: int(len(cases_data) * 0.8)]
    test_df = base_data.loc[base_data.index.difference(train_df.index)]
    val_df = test_df.iloc[: int(len(test_df) * 0.5)]
    test_df = test_df.loc[test_df.index.difference(val_df.index)]
    return train_df, val_df, test_df


def flatten_df_with_labels(df):
    pos_citations = (
        pd.DataFrame(df["citation_idx"].explode())
        .reset_index()
        .rename({"index": "idx"}, axis=1)
    )
    pos_citations["label"] = 0
    neg_citations = (
        pd.DataFrame(df["neg_citation_idx"].explode())
        .reset_index()
        .rename({"index": "idx", "neg_citation_idx": "citation_idx"}, axis=1)
    )
    neg_citations["label"] = 1
    citations = (
        pd.concat([pos_citations, neg_citations]).sample(frac=1, replace=False).dropna()
    )
    return citations


def sort_by_date(cases_data, case_lines):
    cases_data = cases_data.sort_values(["decision_date", "id"])
    case_lines = case_lines.loc[cases_data.index]
    cases_data = cases_data.reset_index(drop=True)
    case_lines = case_lines.reset_index(drop=True)
    return cases_data, case_lines


def load_train_test_splits(cases_data, subset_dir, save_files=False):
    if subset_dir and cases_data is None:

        train_flat = pd.read_csv(f"{subset_dir}/train_map.csv",)
        val_flat = pd.read_csv(f"{subset_dir}/val_map.csv",)
        test_flat = pd.read_csv(f"{subset_dir}/test_map.csv",)

        train_df = (
            train_flat[train_flat.label == 1].groupby("idx").agg({"citation_idx": list})
        )
        val_df = (
            val_flat[val_flat.label == 1].groupby("idx").agg({"citation_idx": list})
        )
        test_df = (
            test_flat[test_flat.label == 1].groupby("idx").agg({"citation_idx": list})
        )
    else:

        train_df, val_df, test_df = make_train_test_splits(cases_data)

        train_flat = flatten_df_with_labels(train_df)
        val_flat = flatten_df_with_labels(val_df)
        test_flat = flatten_df_with_labels(test_df)
        if save_files:
            train_flat.to_csv(
                f"{subset_dir}/train_map.csv", index=False, index_label=False
            )
            val_flat.to_csv(f"{subset_dir}/val_map.csv", index=False, index_label=False)
            test_flat.to_csv(
                f"{subset_dir}/test_map.csv", index=False, index_label=False
            )

    print(f"Number of cases in train_dataset: {len(train_df)}")
    print(f"Number of cases in valiadtion_dataset: {len(val_df)}")
    print(f"Number of cases in test_dataset: {len(test_df)}")

    return train_df, val_df, test_df


def analyze_and_clean(text_files, subset_dir, citation_graph, save_files=False):
    print(f"\nLoading cases from text files at {subset_dir}\n")

    cases_data = load_cases(text_files)

    print(f"\nTotal number of cases in the raw dataset : {len(cases_data)}")
    print(f"Total number of columns in the raw dataset : {len(cases_data.columns)}\n")

    # view the first case
    sample_case = cases_data.iloc[0, :].to_json()
    # fix for escape chars
    sample_case = json.dumps(json.loads(sample_case), indent=2)
    print("\n" + "#" * 40 + " SAMPLE_RECORD " + "#" * 40 + "\n")
    print(sample_case)
    print("\n" + "#" * 90 + "\n")

    drop_cols = ["docket_number", "preview", "cites_to", "citations"]
    cases_data = cases_data.drop(drop_cols, axis=1)

    # convert the id to int for indexing
    cases_data["id"] = cases_data["id"].astype(int)
    cases_data = cases_data.set_index("id")

    cases_data = load_and_display_dates(cases_data)

    cases_data, volumes_data = load_volumes(cases_data, subset_dir, save_files)

    cases_data, reporters_data = load_and_display_reporters(
        cases_data, subset_dir, save_files
    )
    cases_data, courts_data = load_and_display_courts(
        cases_data, subset_dir, save_files
    )
    cases_data, jurisdictions_data = load_and_display_jurisdictions(
        cases_data, subset_dir, save_files
    )

    cases_data = load_and_display_text_cols(cases_data)

    cases_data = load_and_display_citations(
        cases_data, citation_graph, subset_dir, save_files
    )
    cases_data = cases_data.reset_index()
    case_lines = preprocess_text(
        cases_data, f_name=f"{subset_dir}/caselines.txt", overwrite=False
    )

    show_preprocessed_cases(case_lines)

    show_word_count_by_jurisdiction(cases_data, case_lines, jurisdictions_data)
    show_word_count_by_court(cases_data, case_lines, courts_data)

    case2idx, cases_data = map_case2idx(cases_data)
    cases_data = load_neg_case_idx(cases_data)

    case_info_columns = [
        "jurisdiction_id",
        "court_id",
        "decision_date",
        "head_matter",
        "opinion_text",
        "citation_ids",
        "citation_idx",
        "neg_citation_idx",
    ]
    cases_data = cases_data[case_info_columns]
    if save_files:
        cases_data.to_json(f"{subset_dir}/case_info.json", lines=True, orient="records")

    print(f"\nTotal number of cases in the processed dataset : {len(cases_data)}")
    print(
        f"Total number of columns in the processed dataset : {len(cases_data.columns)}\n"
    )

    return cases_data, case_lines
