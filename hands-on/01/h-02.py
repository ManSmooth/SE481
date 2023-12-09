import pandas as pd
import os
import numpy as np
from functions import *

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
path = r"./resources/software_developer_united_states_1971_20191023_1.csv"

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

    print(query_map.head())
