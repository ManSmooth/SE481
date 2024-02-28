import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
import os
import json
from pathlib import Path
from flask import Flask, request
from elasticsearch import Elasticsearch
import time
from sklearn.preprocessing import minmax_scale
import re


class PR:
    def __init__(self, alpha):
        self.crawled_folder = Path(os.path.abspath("")) / "crawled/"
        self.alpha = alpha
        self.url_extract()

    def url_extract(self):
        url_maps = {}
        all_urls = set([])
        for file in os.listdir(self.crawled_folder):
            if file.endswith(".json"):
                try:
                    j = json.load(open(os.path.join(self.crawled_folder, file)))
                    all_urls.add(j["url"])
                    all_urls.update(set(j["url_lists"]))
                    url_maps[j["url"]] = list(set(j["url_lists"]))
                except json.JSONDecodeError:
                    print(file)
        all_urls = list(all_urls)
        self.url_maps = url_maps
        self.all_urls = all_urls

    def pr_calc(self):
        url_maps, all_urls = self.url_maps, self.all_urls
        print(f"{len(all_urls)=}")
        url_idx = {v: i for (i, v) in enumerate(all_urls)}
        size = len(all_urls)
        url_matrix = sparse.lil_array((size, size), dtype=int)
        for url in url_maps:
            if len(url_maps[url]) > 0 and len(all_urls) > 0:
                url_matrix[
                    url_idx[url], [url_idx[sub_url] for sub_url in url_maps[url]]
                ] = 1
        # return url_matrix
        print(f"bytes@prepad: {url_matrix.data.nbytes}")
        rows = np.where(url_matrix.sum(1) == 0)[0]
        url_matrix[rows, :] = np.ones(size, int)
        print(f"bytes@postpad: {url_matrix.data.nbytes}")
        url_matrix = url_matrix * sparse.coo_array(1 / url_matrix.sum(axis=1)).T
        print(f"bytes@multiply: {url_matrix.data.nbytes}")

        x0 = np.repeat(1 / len(all_urls), len(all_urls)).T
        v = np.repeat(1 / len(all_urls), len(all_urls)).T

        prev_Px = x0
        Px = self.alpha * x0 @ url_matrix + (1 - self.alpha) * v
        i = 0
        while any(abs(np.asarray(prev_Px).flatten() - np.asarray(Px).flatten()) > 1e-8):
            i += 1
            prev_Px = Px
            Px = self.alpha * Px @ url_matrix + (1 - self.alpha) * v

        print(
            "Converged in {0} iterations: {1}".format(
                i, np.around(np.asarray(Px).flatten().astype(float), 5)
            )
        )

        self.pr_result = pd.Series(minmax_scale(Px), index=all_urls)


class BM25(object):
    def __init__(self, fitted_vectorizer: TfidfVectorizer, b=0.75, k1=1.6):
        self.fitted_vectorizer = fitted_vectorizer
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """Fit IDF to documents X"""
        self.y = super(TfidfVectorizer, self.fitted_vectorizer).transform(X)
        self.avdl = self.y.sum(1).mean()

    def transform(self, q):
        """Calculate BM25 between query q and documents X"""
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        len_y = self.y.sum(1).A1
        (q,) = super(TfidfVectorizer, self.fitted_vectorizer).transform(q)
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        y = self.y.tocsc()[:, q.indices]
        denom = y + (k1 * (1 - b + b * len_y / avdl))[:, None]
        idf = self.fitted_vectorizer._tfidf.idf_[None, q.indices] - 1.0
        numer = y.multiply(np.broadcast_to(idf, y.shape)) * (k1 + 1)
        return (numer / denom).sum(1).A1


def custom_preprocessor(s: str):
    lemmatizer = WordNetLemmatizer()
    s = re.sub(r"[^A-Za-z]", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(" +", " ", s)
    s = s.lower()
    s = word_tokenize(s)
    s = set(s).difference(set(stopwords.words("english")))
    s = [word for word in s if len(word) > 2]
    s = [lemmatizer.lemmatize(w) for w in s]
    s = " ".join(s)
    return s


class ManualIndexer:
    def __init__(self):
        self.crawled_folder = Path(os.path.abspath("")) / "crawled/"
        self.stored_file = "indexer/manual_indexer.pickle"
        if not Path(self.stored_file).parent.exists():
            Path.mkdir(Path(self.stored_file).parent)
        if os.path.isfile(self.stored_file):
            with open(self.stored_file, "rb") as f:
                cached_dict = pickle.load(f)
            self.__dict__.update(cached_dict)
        else:
            self.run_indexer()

    def run_indexer(self):
        documents = []
        for file in os.listdir(self.crawled_folder):
            if file.endswith(".json"):
                j = json.load(open(os.path.join(self.crawled_folder, file)))
                documents.append(j)
        self.documents = pd.DataFrame.from_dict(documents)
        tfidf_vectorizor = TfidfVectorizer(
            preprocessor=custom_preprocessor, stop_words=stopwords.words("english")
        )
        tfidf_vectorizor.fit(
            self.documents.apply(lambda s: " ".join(s[["title", "text"]]), axis=1)
        )
        self.bm25 = BM25(tfidf_vectorizor)
        self.bm25.fit(
            self.documents.apply(lambda s: " ".join(s[["title", "text"]]), axis=1)
        )
        self.pr = PR(0.8)
        self.pr.pr_calc()
        with open(self.stored_file, "wb") as f:
            pickle.dump(self.__dict__, f)

    def search(self, query):
        scores = minmax_scale(self.bm25.transform([query]))
        df = pd.DataFrame(scores, columns=["score"])
        result_df = self.documents.join(df)
        result_df["score"] = result_df.apply(
            lambda x: self.pr.pr_result[x["url"]] * x["score"], axis=1
        )
        result_df = result_df.sort_values("score", ascending=False).head(20)

        # Atrocious Python
        result_df["text_highlight"] = result_df["text"].apply(
            lambda x: [
                f"...{x[span[0] - 24: span[0]]}<b>{x[span[0] : span[1]]}</b>{x[span[1]: span[1] + 24]}..."
                for span in [
                    word_span
                    for word_spans in [
                        [
                            m.span()
                            for m in re.finditer(rf"\b{q_word}\b", x, re.IGNORECASE)
                        ]
                        for q_word in query.split()
                    ]
                    for word_span in word_spans
                ]
            ]
        )
        return result_df


app = Flask(__name__)
app.manual_indexer = ManualIndexer()
app.es_client = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "+oEqEIt7p6lC_=rI1HIC"),
    ca_certs="./http_ca.crt",
)


@app.route(r"/search_es", methods=["GET"])
def search_es():
    start = time.time()
    response_object = {"status": "success"}
    argList = request.args.to_dict(flat=False)
    query_term = argList["query"][0]
    results = app.es_client.search(
        index="simple",
        source_excludes=["url_lists"],
        size=100,
        query={
            "script_score": {
                "query": {"match": {"text": query_term}},
                "script": {"source": "_score * doc['pagerank'].value"},
            }
        },
    )
    end = time.time()
    total_hit = results["hits"]["total"]["value"]
    scores = minmax_scale(app.manual_indexer.bm25.transform([query_term]))
    df = pd.DataFrame(scores, columns=["score"])
    bm25_scores = app.manual_indexer.documents.join(df).set_index("url")["score"]
    results_df = pd.DataFrame(
        [
            [
                hit["_source"]["title"],
                hit["_source"]["url"],
                hit["_source"]["text"][:100],
                (
                    hit["_score"] * bm25_scores[hit["_source"]["url"]]
                    if hit["_source"]["url"] in bm25_scores
                    else 0.01
                ),
            ]
            for hit in results["hits"]["hits"]
        ],
        columns=["title", "url", "text", "score"],
    ).sort_values("score", ascending=False)
    response_object["total_hit"] = total_hit
    response_object["results"] = results_df.to_dict("records")
    response_object["elapse"] = end - start
    return response_object


@app.route(r"/search_manual", methods=["GET"])
def search_manual():
    start = time.time()
    response_object = {"status": "success"}
    argList = request.args.to_dict(flat=False)
    query_term = argList["query"][0]
    results = app.manual_indexer.search(query_term)
    end = time.time()
    results = results[results["score"] > 0]
    results_df = (
        results.sort_values("score", ascending=False)
        .drop("url_lists", axis=1)
        .head(100)
    )
    results_df["text"] = results_df["text"].apply(lambda x: x[:100] + '...')
    response_object["hits"] = len(results)
    response_object["results"] = results_df.to_dict("records")
    response_object["elapse"] = end - start
    return response_object


if __name__ == "__main__":
    app.run(debug=False)
