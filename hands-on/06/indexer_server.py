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
        with open(self.stored_file, "wb") as f:
            pickle.dump(self.__dict__, f)

    def search(self, query):
        scores = self.bm25.transform([query])
        df = pd.DataFrame(scores, columns=["score"])
        return self.documents.join(df)

app = Flask(__name__)
app.manual_indexer = ManualIndexer()
app.es_client = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "+oEqEIt7p6lC_=rI1HIC"),
    ca_certs="./http_ca.crt",
)
@app.route(r"/search_es", methods=['GET'])
def search_es():
    start = time.time()
    response_object = {'status': 'success'}
    argList = request.args.to_dict(flat=False)
    query_term=argList['query'][0]
    results = app.es_client.search(index='simple', source_excludes=['url_lists'], size=100, query={"match": { "text": query_term}})
    end = time.time()
    total_hit = results['hits']['total']['value']
    results_df = pd.DataFrame([[hit["_source"]['title'], hit["_source"]['url'], hit["_source"]['text'][:100], hit["_score"]] for hit in results['hits']['hits']], columns=['title', 'url', 'text', 'score'])
    response_object['total_hit'] = total_hit
    response_object['results'] = results_df.to_dict('records')
    response_object['elapse'] = end - start
    return response_object

@app.route(r"/search_manual", methods=['GET'])
def search_manual():
    start = time.time()
    response_object = {'status': 'success'}
    argList = request.args.to_dict(flat=False)
    query_term=argList['query'][0]
    results = app.manual_indexer.search(query_term)
    end = time.time()
    results_df = results.sort_values("score", ascending=False).drop("url_lists", axis=1).head(20)
    response_object['results'] = results_df.to_dict('records')
    response_object['elapse'] = end - start
    return response_object

if __name__ == "__main__":
    app.run(debug=False)

