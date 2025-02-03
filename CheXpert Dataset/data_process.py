import pandas as pd

data_dir = "" # your dir here
df_demo = pd.DataFrame(
    pd.read_excel(data_dir + "CHEXPERT DEMO.xlsx", engine="openpyxl")
)
df_demo = df_demo.rename(columns={"PRIMARY_RACE": "race"})
df_demo = df_demo.rename(columns={"PATIENT": "patient_id"})
df_demo = df_demo.rename(columns={"GENDER": "sex"})
df_demo = df_demo.rename(columns={"AGE_AT_CXR": "age"})
df_demo = df_demo.rename(columns={"ETHNICITY": "ethnicity"})
df_demo = df_demo.drop(["sex", "age"], axis=1)
df_demo.head()
df_data_split = pd.read_csv(data_dir + "chexpert_split_2021_08_20.csv").set_index(
    "index"
)
df_img_data = pd.read_csv(data_dir + "train_cheXbert.csv")
df_img_data = pd.concat([df_img_data, df_data_split], axis=1)
df_img_data = df_img_data[~df_img_data.split.isna()]
split = df_img_data.Path.str.split("/", expand=True)
df_img_data["patient_id"] = split[2]
df_img_data = df_img_data.rename(columns={"Age": "age"})
df_img_data = df_img_data.rename(columns={"Sex": "sex"})
df_img_data.head()
df_cxr = df_demo.merge(df_img_data, on="patient_id")

columns = df_cxr.columns.tolist()
target_index = columns.index("Enlarged Cardiomediastinum")
columns.remove("No Finding")
columns.remove("Unnamed: 0")
columns.insert(target_index, "No Finding")
df_cxr = df_cxr[columns]
df_cxr.head()

white = "White"
asian = "Asian"
black = "Black"
mask = df_cxr.race.str.contains("Black", na=False)
df_cxr.loc[mask, "race"] = black
mask = df_cxr.race.str.contains("White", na=False)
df_cxr.loc[mask, "race"] = white
mask = df_cxr.race.str.contains("Asian", na=False)
df_cxr.loc[mask, "race"] = asian
df_cxr["race"].unique()
df_cxr = df_cxr[df_cxr.race.isin([asian, black, white])]
df_cxr = df_cxr[df_cxr.ethnicity.isin(["Non-Hispanic/Non-Latino", "Not Hispanic"])]
df_cxr = df_cxr[df_cxr["Frontal/Lateral"] == "Frontal"]
df_cxr["race_label"] = df_cxr["race"]
df_cxr.loc[df_cxr["race_label"] == white, "race_label"] = 0
df_cxr.loc[df_cxr["race_label"] == asian, "race_label"] = 1
df_cxr.loc[df_cxr["race_label"] == black, "race_label"] = 2
df_cxr["sex_label"] = df_cxr["sex"]
df_cxr.loc[df_cxr["sex_label"] == "Male", "sex_label"] = 0
df_cxr.loc[df_cxr["sex_label"] == "Female", "sex_label"] = 1

all_labels = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

df_cxr["disease"] = df_cxr[all_labels[0]]
df_cxr.loc[df_cxr[all_labels[0]] == 1, "disease"] = all_labels[0]
df_cxr.loc[df_cxr[all_labels[10]] == 1, "disease"] = all_labels[10]
df_cxr.loc[df_cxr["disease"].isna(), "disease"] = "Other"

df_cxr["disease_label"] = df_cxr["disease"]
df_cxr.loc[df_cxr["disease_label"] == all_labels[0], "disease_label"] = 0
df_cxr.loc[df_cxr["disease_label"] == all_labels[10], "disease_label"] = 1
df_cxr.loc[df_cxr["disease_label"] == "Other", "disease_label"] = 2

df_train = df_cxr[df_cxr.split == "train"]
df_val = df_cxr[df_cxr.split == "validate"]
df_test = df_cxr[df_cxr.split == "test"]

test_predictions = pd.read_csv("./predictions.test.csv") # your dir here
rename_dict1 = {
    col: f"Y_{col.split('_')[1]}"
    for col in test_predictions.columns
    if col.startswith("target_")
}
rename_dict2 = {
    col: f"Phat_{col.split('_')[1]}"
    for col in test_predictions.columns
    if col.startswith("class_")
}
test_predictions.rename(columns=rename_dict1, inplace=True)
test_predictions.rename(columns=rename_dict2, inplace=True)

preds = pd.concat(
    [df_test.reset_index(drop=True), test_predictions.reset_index(drop=True)], axis=1
)

preds = preds[
    [
        "patient_id",
        "race",
        "race_label",
        "sex",
        "sex_label",
        "Pleural Effusion",
        "disease",
        "disease_label",
        "Phat_10",
        "Y_10",
    ]
]
preds = preds.rename(columns={"Phat_10": "Phat", "Y_10": "Y"})
preds.to_csv("./chexpert_preds.csv", index=False) # your dir here
