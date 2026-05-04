import pandas as pd
import os

PATH = "/home/kai/siliconmind_oss_76k/data/"
files = os.listdir(PATH)

dfs = []

for f in files:
    with open(os.path.join(PATH, f), "rb") as file:
        df = pd.read_parquet(file)
        dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

df.to_json(
    "merged.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)