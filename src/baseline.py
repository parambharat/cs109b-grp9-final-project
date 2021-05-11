from multiprocessing import Pool

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity


def split_text(text):
    idx, text = text
    return idx, text.split()


def load_case_texts(case_lines):
    pool = Pool(14)
    case_texts = pool.imap_unordered(
        split_text, case_lines.text.items(), chunksize=1000
    )
    case_texts = [item for item in case_texts]
    case_texts = pd.DataFrame(case_texts, columns=["idx", "text"]).set_index("idx")
    case_texts = case_texts.loc[case_lines.index]
    return case_texts


class CaselawCorpus:
    def __init__(self, iterator, dictionary, **kwargs):
        self.iterator = iterator
        self.dictionary = dictionary

    def __iter__(self):
        for line in self.iterator:
            # assume there's one document per line, tokens separated by whitespace
            yield self.dictionary.doc2bow(line)

    def __len__(self):
        return len(self.iterator)


class CosineSimilarityIndexer:
    def __init__(self, no_below=5, no_above=0.75):
        self.dictionary = None
        self.corpus = None
        self.model = None
        self.index = None
        self.no_below = no_below
        self.no_above = no_above

    def fit(self, texts, **kwargs):
        self.dictionary = Dictionary(texts)
        self.dictionary.filter_extremes(self.no_below, self.no_above, keep_n=50000)
        self.dictionary.compactify()

        self.corpus = CaselawCorpus(texts, self.dictionary, **kwargs)
        self.model = TfidfModel(self.corpus)
        self.index = SparseMatrixSimilarity(
            self.model[self.corpus],
            num_features=len(self.dictionary),
            num_terms=len(self.dictionary),
            num_docs=len(self.corpus),
            maintain_sparsity=True,
        )
        return self

    def predict(self, texts):
        pred_corpus = CaselawCorpus(texts, self.dictionary)
        pred_vecs = self.model[pred_corpus]
        pred_sims = self.index[pred_vecs]
        return pred_sims


def sort_preds(cases_data, sample_idxs, sample_preds, use_heuristic=True):
    sample_cases = cases_data.loc[sample_idxs]
    top_preds = []
    for (idx, row), sample_pred in zip(sample_cases.iterrows(), sample_preds):
        row = row.to_dict()
        jur_id = row["jurisdiction_id"]
        court_id = row["court_id"]
        decision_date = row["decision_date"]
        pred = sample_pred.argsort()[::-1]
        if use_heuristic:
            pred = pred[
                (cases_data.jurisdiction_id == jur_id)
                & (cases_data.court_id == court_id)
                & (cases_data.decision_date > decision_date)
            ]
        top_preds.append(pred)

    return top_preds
