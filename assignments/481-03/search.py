from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import re
import nltk
import string
import pandas as pd
import numpy as np
import os
import joblib
import argparse

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)


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


# Pickleable Preprocessor
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


def extract_description(df: pd.DataFrame):
    s = df["job_description"]
    s = s.apply(
        lambda s: s.lower()
        .translate(
            str.maketrans(
                string.punctuation + "\xa0", " " * (len(string.punctuation) + 1)
            )
        )
        .translate(str.maketrans(string.whitespace, " " * len(string.whitespace)))
    )
    s = s.apply(lambda s: re.sub(" +", " ", s))
    return s


def search(
    terms: str, tf_idf: np.matrix, vectorizer, cleaned_description
) -> pd.DataFrame:
    Q = vectorizer.transform([terms])
    cos_sim = cosine_similarity(tf_idf, Q, dense_output=False)
    df = pd.DataFrame.sparse.from_spmatrix(cos_sim, columns=["score"]).sort_values(
        "score", ascending=False
    )
    df["description"] = df.apply(lambda x: cleaned_description.loc[x.index])
    return df


def searchBM25(terms: str, vectorizer: BM25, cleaned_description) -> pd.DataFrame:
    scores = vectorizer.transform([terms])
    df = pd.DataFrame(scores, columns=["score"]).sort_values("score", ascending=False)
    df["description"] = df.apply(lambda x: cleaned_description.loc[x.index])
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search keywords")
    parser.add_argument("-r", action="store_false", help="force regenerate pickles")
    parser.add_argument("Q", metavar="Q", type=str, nargs="+", help="queries to search")

    args = parser.parse_args()
    cleaned_description: pd.Series = None
    tf_idf: sparse.spmatrix = None
    tf_idf_vocab: np.ndarray = None
    tf_idf_vectorizer: TfidfVectorizer = None
    bm25_vectorizer: BM25 = None
    if (
        os.path.exists("./resources/cleaned_descriptions.joblib")
        and os.path.exists("./resources/bigram_tf_idf.joblib")
        and os.path.exists("./resources/bigram_tf_idf_names.joblib")
        and os.path.exists("./resources/bigram_tf_idf_vectorizer.joblib")
        and os.path.exists("./resources/bm25_vectorizer.joblib")
        and args.r
    ):
        print("Joblibs found.")
        with open("./resources/cleaned_descriptions.joblib", "rb") as f:
            cleaned_description = joblib.load(f)
        with open("./resources/bigram_tf_idf.joblib", "rb") as f:
            tf_idf = joblib.load(f)
        with open("./resources/bigram_tf_idf_names.joblib", "rb") as f:
            tf_idf_vocab = joblib.load(f)
        with open("./resources/bigram_tf_idf_vectorizer.joblib", "rb") as f:
            tf_idf_vectorizer = joblib.load(f)
        with open("./resources/bm25_vectorizer.joblib", "rb") as f:
            bm25_vectorizer = joblib.load(f)
    else:
        print("Joblib(s) missing, recreating.")
        m1 = pd.read_csv("./resources/m1.csv")
        cleaned_description = extract_description(m1)
        cleaned_description.drop_duplicates(inplace=True)
        cleaned_description.reset_index(drop=True, inplace=True)
        stop_dict = set(stopwords.words("english"))
        tf_idf_vectorizer = TfidfVectorizer(
            preprocessor=custom_preprocessor,
            use_idf=True,
            ngram_range=(1, 2),
        )
        print("Transforming")
        tf_idf = tf_idf_vectorizer.fit_transform(cleaned_description)
        tf_idf_vocab = tf_idf_vectorizer.get_feature_names_out()
        bm25_vectorizer = BM25(tf_idf_vectorizer)
        bm25_vectorizer.fit(cleaned_description)
        print("Pickling")
        with open("./resources/cleaned_descriptions.joblib", "wb") as f:
            joblib.dump(cleaned_description, f)
        with open("./resources/bigram_tf_idf.joblib", "wb") as f:
            joblib.dump(tf_idf, f)
        with open("./resources/bigram_tf_idf_names.joblib", "wb") as f:
            joblib.dump(tf_idf_vocab, f)
        with open("./resources/bigram_tf_idf_vectorizer.joblib", "wb") as f:
            joblib.dump(tf_idf_vectorizer, f)
        with open("./resources/bm25_vectorizer.joblib", "wb") as f:
            joblib.dump(bm25_vectorizer, f)
    for query in args.Q:
        if query == "-r":
            continue
        pd.set_option("max_colwidth", 128)
        print(f"\n{query}")
        print(f"\nTF IDF\n--------------------------------")
        print(search(query, tf_idf, tf_idf_vectorizer, cleaned_description).head(5))
        print(f"\nBM25\n--------------------------------")
        print(searchBM25(query, bm25_vectorizer, cleaned_description).head(5))
