import pandas as pd
import os
import numpy as np
from functions import *

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
path = r"./resources/software_developer_united_states_1971_20191023_1.csv"

if __name__ == "__main__":
    print(os.getcwd())
    rows = pd.read_csv(path)
    dbs = parse_db()
    descriptions = transformation_pipe(rows)
    db_series = pd.Series(dbs)
    res_frame = pd.DataFrame(db_series, columns=["db"])
    res_frame["occurance"] = res_frame.apply(
        lambda row: descriptions.apply(
            lambda s: np.all([x in s for x in row["db"]])
        ).sum(),
        axis=1,
    )
    res_frame["occurance_percent"] = res_frame["occurance"] / len(descriptions)
    res_frame["occurance+python"] = res_frame.apply(
        lambda row: descriptions.apply(
            lambda s: np.all([x in s for x in row["db"]]) and "python" in s
        ).sum(),
        axis=1,
    )
    res_frame["occurance+python_to_occurance_percent"] = (
        res_frame["occurance+python"] / res_frame["occurance"]
    )
    print(res_frame.sort_values)
