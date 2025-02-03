import pandas as pd
from func import (
    get_runs,
    evaluation,
    test,
    figure2,
    figure3,
    adjustable_alpha,
    plot_alpha,
    simulate_data,
    visualize_simulation,
    alpha_calsize,
    plot_3d,
    plot_phat_distribution,
)

##### Data prep
## ICM/NICM
icm_preds = pd.read_csv("./icmnicm_preds.csv")
icm_runs_dir = (
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/icmnicm/runs/"
)
get_runs(icm_preds, "Female", 100, 0.5, icm_runs_dir)


## CheXpert
chexpert_data_dir = "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/chexpert/chexpert_preds.csv"
chexpert_preds = pd.read_csv(chexpert_data_dir)
chexpert_runs_dir = (
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/chexpert/runs/"
)
get_runs(chexpert_preds, "race_label", 100, 0.5, chexpert_runs_dir)

## ISIC18
isic18_data_dir = "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/isic18/isic18_preds.csv"
isic18_preds = pd.read_csv(isic18_data_dir)
isic18_preds["Phat"] = isic18_preds.apply(
    lambda row: 1 - row["Phat"] if row["Yhat"] == 0 else row["Phat"], axis=1
)
isic18_runs_dir = (
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/isic18/runs/"
)
get_runs(isic18_preds, "dark_skin", 100, 0.5, isic18_runs_dir)

##### Unfairness mitigation methods
## ICM/NICM
icm_save_dir = "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/icmnicm/"
evaluation(0.08, 0.2, icm_runs_dir, icm_save_dir, "Female", 1)

## CheXpert
chexpert_save_dir = (
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/chexpert/"
)
evaluation(0.1, 0.2, chexpert_runs_dir, chexpert_save_dir, "race_label", 50)

## ISIC18
isic18_save_dir = (
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/isic18/"
)
evaluation(0.1, 0.2, isic18_runs_dir, isic18_save_dir, "dark_skin", 5)

##### Visualization and hypothesis tests
## ICM/NICM
icm_base = pd.read_csv(
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/icmnicm/base.csv"
)
icm_base_cal = pd.read_csv(
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/icmnicm/base_cal.csv"
)
icm_roc = pd.read_csv(
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/icmnicm/roc.csv"
)
icm_icp_ag = pd.read_csv(
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/icmnicm/icp_ag.csv"
)
icm_icp_sp = pd.read_csv(
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/icmnicm/icp_sp.csv"
)
figure2(
    [icm_base, icm_base_cal, icm_roc, icm_icp_ag, icm_icp_sp],
    ["Base", "Base_cal", "ROC", "ICP_org", "FairICP"],
    ["ACC", "TPR", "FPR"],
    ["Male", "Female"],
    "Metrics by Group (ICM/NICM)",
)
figure3(
    icm_base,
    [icm_base_cal, icm_roc, icm_icp_ag, icm_icp_sp],
    ["Base_cal", "ROC", "ICP_org", "FairICP"],
    ["ACC", "TPR", "FPR"],
    {"0": "Male", "1": "Female"},
)
test(icm_base, icm_base_cal)
test(icm_base, icm_roc)
test(icm_base, icm_icp_ag)
test(icm_base, icm_icp_sp)


## Chexpert
chexpert_base = pd.read_csv(
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/chexpert/base.csv"
)
chexpert_base_cal = pd.read_csv(
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/chexpert/base_cal.csv"
)
chexpert_roc = pd.read_csv(
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/chexpert/roc.csv"
)
chexpert_icp_ag = pd.read_csv(
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/chexpert/icp_ag.csv"
)
chexpert_icp_sp = pd.read_csv(
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/chexpert/icp_sp.csv"
)
figure2(
    [chexpert_base, chexpert_base_cal, chexpert_roc, chexpert_icp_ag, chexpert_icp_sp],
    ["Base", "Base_cal", "ROC", "ICP_org", "FairICP"],
    ["ACC", "TPR", "FPR"],
    ["White", "Asian", "Black"],
    "Metrics by Group (CheXpert)",
)
figure3(
    chexpert_base,
    [chexpert_base_cal, chexpert_roc, chexpert_icp_ag, chexpert_icp_sp],
    ["Base_cal", "ROC", "ICP_org", "FairICP"],
    ["ACC", "TPR", "FPR"],
    {"0": "White", "1": "Asian", "2": "Black"},
)
test(chexpert_base, chexpert_base_cal)
test(chexpert_base, chexpert_roc)
test(chexpert_base, chexpert_icp_ag)
test(chexpert_base, chexpert_icp_sp)


## ISIC18 - ICP
isic18_base = pd.read_csv(
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/isic18/base.csv"
)
isic18_base_cal = pd.read_csv(
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/isic18/base_cal.csv"
)
isic18_roc = pd.read_csv(
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/isic18/roc.csv"
)
isic18_icp_ag = pd.read_csv(
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/isic18/icp_ag.csv"
)
isic18_icp_sp = pd.read_csv(
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/isic18/icp_sp.csv"
)
figure2(
    [isic18_base, isic18_base_cal, isic18_roc, isic18_icp_ag, isic18_icp_sp],
    ["Base", "Base_cal", "ROC", "ICP_org", "FairICP"],
    ["ACC", "TPR", "FPR"],
    ["Light", "Dark"],
    "Metrics by Group (ISIC18)",
)
figure3(
    isic18_base,
    [isic18_base_cal, isic18_roc, isic18_icp_ag, isic18_icp_sp],
    ["Base_cal", "ROC", "ICP_org", "FairICP"],
    ["ACC", "TPR", "FPR"],
    {"0": "Light", "1": "Dark"},
)
test(isic18_base, isic18_base_cal)
test(isic18_base, isic18_roc)
test(isic18_base, isic18_icp_ag)
test(isic18_base, isic18_icp_sp)

##### ICP lever one: confidence level
## Chexpert
FairICP_alphas = adjustable_alpha(
    chexpert_runs_dir,
    0.2,
    "race_label",
    50,
    0.1,
    0.2,
    "Specific",
    50,
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/chexpert/",
)
ICP_org_alphas = adjustable_alpha(
    chexpert_runs_dir,
    0.2,
    "race_label",
    50,
    0.1,
    0.2,
    "Agnostic",
    50,
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/chexpert/",
)
FairICP_alphas = pd.read_csv(
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/chexpert/FairICP_alphas.csv"
)
FairICP_alphas["Confidence"] = 1 - FairICP_alphas["Alpha"]
ICP_org_alphas = pd.read_csv(
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/chexpert/ICP_org_alphas.csv"
)
ICP_org_alphas["Confidence"] = 1 - ICP_org_alphas["Alpha"]
plot_alpha(FairICP_alphas, ICP_org_alphas)

##### ICP lever two: calibration size
## Simulation
save_dir = "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/simulation/"
simulation_df = simulate_data(
    n_samples_group1=200000,
    n_samples_group2=200000,
    acc_group1=0.80,
    acc_group2=0.75,
    correct_mean_group1=0.88,
    correct_sd_group1=0.03,
    incorrect_mean_group1=0.72,
    incorrect_sd_group1=0.04,
    correct_mean_group2=0.82,
    correct_sd_group2=0.04,
    incorrect_mean_group2=0.75,
    incorrect_sd_group2=0.04,
)
visualize_simulation(simulation_df)
simulation_df.loc[simulation_df["Yhat_agnostic"] == 0, "Phat"] = (
    1 - simulation_df["Phat"]
)
simulation_df.to_csv(save_dir + "simulation_df.csv", index=False)
alpha_calsize(
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/simulation/simulation_df.csv",
    "Group",
    "Specific",
    0.2,
    1,
    0.1,
    0.21,
    [10, 100, 1000, 10000, 100000],
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/simulation/",
    10,
)
df = pd.read_csv(
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/simulation/alpha_calsize.csv"
)
plot_3d(df, "cal_size", "confidence", "Cost", "group")


##### Plot Phat distributions
df1 = pd.read_csv(
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/icmnicm/runs/run_1.csv"
)
df2 = pd.read_csv(
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/chexpert/runs/run_1.csv"
)
df3 = pd.read_csv(
    "/home/sunx/data/aiiih/projects/sunx/clinical_projects/ICP/data/isic18/runs/run_1.csv"
)
plot_phat_distribution(df1, df2, df3)
