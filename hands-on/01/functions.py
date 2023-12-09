import pandas as pd
import string
import requests
from bs4 import BeautifulSoup

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
