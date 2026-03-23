import pandas as pd

inp = r"D:\PyProjects\Dataset\unsplash-research-dataset-full-latest\work\dataset\metadata\manifest_aesthetic.parquet"
outp = r"D:\PyProjects\Dataset\unsplash-research-dataset-full-latest\work\dataset\metadata\manifest_aesthetic_keep.parquet"

df = pd.read_parquet(inp)

ok = df[df["aesthetic_ok"] == True].copy()
thr = ok["aesthetic_score"].quantile(0.30)   # 先去掉 bottom 30%
ok["aesthetic_pass"] = ok["aesthetic_score"] >= thr

keep = ok[ok["aesthetic_pass"] == True].copy()
keep.to_parquet(outp, index=False)

print("threshold =", thr)
print("keep rows =", len(keep))