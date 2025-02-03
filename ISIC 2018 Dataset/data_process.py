import pandas as pd

meta = pd.read_csv("./metadata.csv") # your dir here
meta = meta.dropna(subset=["estimated_ita"])
meta["skin_type"] = pd.cut(
    meta["estimated_ita"],
    bins=[-float("inf"), 10, 19, 28, 41, 55, float("inf")],
    labels=[6, 5, 4, 3, 2, 1],
).astype(int)
# Lesion: identify NV
lesion_map = {"MEL": 0, "NV": 1, "BCC": 0, "AKIEC": 0, "BKL": 0, "DF": 0, "VASC": 0}
meta["Y"] = meta["lesion"].map(lesion_map)
preds = pd.read_csv("./predictions_01DataShift00Compare41.csv") # your dir here
preds.rename(columns={"predicted_lesion": "Yhat"}, inplace=True)
preds["Phat"] = preds["NV"]
merged = pd.merge(meta, preds, on="image", how="inner")
merged["Yhat"] = merged["Yhat"].map(lesion_map)
merged = merged[["image", "skin_type", "Y", "Yhat", "Phat"]]
merged["Phat"] = merged.apply(
    lambda row: 1 - row["Phat"] if row["Yhat"] == 0 else row["Phat"], axis=1
)

# Light skin vs dark skin
merged["skin_type"] = pd.cut(
    merged["skin_type"], bins=[-float("inf"), 2, float("inf")], labels=[0, 1]
)
merged.rename(columns={"skin_type": "dark_skin"}, inplace=True)
merged.to_csv("./isic18_preds.csv", index=False) # your dir here
