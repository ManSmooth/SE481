import pandas as pd
import string
import requests
from bs4 import BeautifulSoup
import os
import numpy as np

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
path = r"./resources/software_developer_united_states_1971_20191023_1.csv"


def extract_description(df: pd.DataFrame):
    s = df["job_description"]
    s = s.apply(
        lambda s: s.lower()
        .translate(str.maketrans("", "", string.punctuation + "\xa0"))
        .translate(str.maketrans(string.whitespace, " " * len(string.whitespace)))
    )
    return s


def tokenize(s: pd.Series):
    return s.apply(lambda s: [x.strip() for x in s.split()])


def transformation_pipe(df: pd.DataFrame):
    s = extract_description(df)
    s = tokenize(s)
    return s


# https://stackoverflow.com/questions/480214/how-do-i-remove-duplicates-from-a-list-while-preserving-order
def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def parse_db():
    res = requests.get(r"https://db-engines.com/en/ranking").content
    soup = BeautifulSoup(res, "html.parser")
    table = soup.find("table", {"class": "dbi"})
    ths = table.find_all("th", {"class": "pad-l"})
    all_db = [
        "".join(s.find("a").find_all(string=True, recursive=False)).strip() for s in ths
    ]
    all_db = f7(all_db)[:10]
    db_list = [[ss.strip() for ss in s.lower().split()] for s in all_db]
    return db_list


if __name__ == "__main__":
    lang = [
        ["java"],
        ["python"],
        ["c"],
        ["kotlin"],
        ["swift"],
        ["rust"],
        ["ruby"],
        ["scala"],
        ["julia"],
        ["lua"],
    ]
    print(os.getcwd())
    rows = pd.read_csv(path)
    descriptions = transformation_pipe(rows)
    dbs = parse_db()
    all_terms = lang + dbs
    query_map = pd.DataFrame(
        descriptions.apply(
            lambda s: [1 if np.all([d in s for d in db]) else 0 for db in all_terms]
        ).values.tolist(),
        columns=[" ".join(d) for d in all_terms],
    )

    print("db with java")
    db_with_java = query_map[query_map["java"] == 1]
    print(db_with_java[[" ".join(d) for d in dbs]].sum().sort_values(ascending=False))

    print("db with oracle")
    db_with_oracle = query_map[query_map["oracle"] == 1]
    print(
        db_with_oracle[[" ".join(d) for d in dbs if d != ["oracle"]]]
        .sum()
        .sort_values(ascending=False)
    )

    print("lang with python")
    lang_with_python = query_map[query_map["python"] == 1]
    print(lang_with_python[[" ".join(d) for d in lang if d != ["python"]]].sum().sort_values(ascending=False))
