import torch
import os
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import binom
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import config
from dataset import icmDataModule
from model import icm_pred

device = "cuda" if torch.cuda.is_available() else "cpu"


### Data prep
def predicted_labels(df, attribute_col):
    """
    Get group-agnostic or group-specific thresholds for maximal accuracy.
    """
    # Separate calibration and test data
    calibration_data = df[df["split"] == "calibration"]
    test_data = df[df["split"] == "test"]

    # Group-agnostic threshold optimization
    best_agnostic_threshold = None
    best_agnostic_accuracy = 0

    for threshold in np.linspace(0, 1, 101):
        preds = (calibration_data["Phat"] >= threshold).astype(int)
        accuracy = accuracy_score(calibration_data["Y"], preds)
        if accuracy > best_agnostic_accuracy:
            best_agnostic_accuracy = accuracy
            best_agnostic_threshold = threshold

    # Group-specific threshold optimization
    best_group_thresholds = {}
    for group in calibration_data[attribute_col].unique():
        group_data = calibration_data[calibration_data[attribute_col] == group]
        best_group_threshold = None
        best_group_accuracy = 0

        for threshold in np.linspace(0, 1, 101):
            preds = (group_data["Phat"] >= threshold).astype(int)
            accuracy = accuracy_score(group_data["Y"], preds)
            if accuracy > best_group_accuracy:
                best_group_accuracy = accuracy
                best_group_threshold = threshold

        best_group_thresholds[group] = best_group_threshold

    # Apply thresholds to the test set
    def apply_specific_threshold(row):
        return int(row["Phat"] >= best_group_thresholds[row[attribute_col]])

    calibration_data["Yhat_agnostic"] = (
        calibration_data["Phat"] >= best_agnostic_threshold
    ).astype(int)
    calibration_data["Yhat_specific"] = calibration_data.apply(
        apply_specific_threshold, axis=1
    )
    test_data["Yhat_agnostic"] = (test_data["Phat"] >= best_agnostic_threshold).astype(
        int
    )
    test_data["Yhat_specific"] = test_data.apply(apply_specific_threshold, axis=1)

    return calibration_data, test_data


def get_runs(test_predictions, attribute_col, N, test_ratio, runs_dir):
    for i in range(N):
        # Add a 'run' column to identify the iteration
        test_predictions["run"] = i

        # Randomly split the entire DataFrame into calibration and test sets
        cal_idx, test_idx = train_test_split(
            test_predictions.index, test_size=test_ratio, random_state=i
        )

        # Assign the splits
        test_predictions["split"] = "calibration"
        test_predictions.loc[test_idx, "split"] = "test"

        # Predict labels using calibration data
        cal, test = predicted_labels(test_predictions, attribute_col)
        result = pd.concat([cal, test], ignore_index=True)

        # Save the result
        result.to_csv(runs_dir + f"run_{i+1}.csv", index=False)


### Post-process methods
# ICP
def selective_risk(lam, cal_yhats, cal_phats, cal_labels):
    """
    Computes the misclassification rate for a given threshold -- the proportion of misclassified selected samples among all selected samples.

    'lam': threshold.
    'cal_yhats': prediction labels of the calibration set.
    'cal_phats': maximum prediction probabilities of the calibration set.
    'cal_labels': true labels of the calibration set.
    """
    selected_indices = cal_phats >= lam
    return (cal_yhats[selected_indices] != cal_labels[selected_indices]).sum() / (
        selected_indices
    ).sum()


def nlambda(lam, cal_phats):
    """
    Counts the number of predictions above a threshold.

    'lam': threshold.
    'cal_phats': maximum prediction probabilities of the calibration set.
    """
    return (cal_phats > lam).sum()


def invert_for_ub(p, lam, delta, cal_yhats, cal_phats, cal_labels):
    """
    Given p, what is the gap between observed binomial cdf and delta?

    Computes the cumulative probability of obtaining up to "observed number of misclassification cases" in "num_selected" trials, each individual case has misclassification probability p.

    'p': misclassification probability of individual trial.
    'delta': user selected maximum accepted misclassification rate.
    """
    misclassification_rate = selective_risk(lam, cal_yhats, cal_phats, cal_labels)
    num_selected = nlambda(lam, cal_phats)
    num_misclassification = misclassification_rate * num_selected
    return binom.cdf(num_misclassification, num_selected, p) - delta


def p_ub(lam, delta, cal_yhats, cal_phats, cal_labels):
    """
    Finds the maximum allowable misclassification rate (p) for a given threshold (λ) such that the binomial cumulative probability remains within delta. Binomial CDF is monotone increasing with respect to p when nlambda is fixed.
    Uses the brentq root-finding algorithm to determine the value of p within the interval [0, 0.9999] where invert_for_ub(p) equals 0 (i.e., the misclassification rate satisfies the confidence constraint).
    """
    return brentq(
        invert_for_ub, 0, 0.9999, args=(lam, delta, cal_yhats, cal_phats, cal_labels)
    )


def optimal_lambda(df, alpha, delta, num):
    """Compute the minimum lambda hat that its misclassification rate upper bound is less than alpha."""
    yhats = df["Yhat_agnostic"]
    phats = df["Phat"]
    labels = df["Y"]
    min_phat = phats.min()

    lambdas = np.linspace(min_phat, 1, 1000)
    lambdas = np.array(
        [lam for lam in lambdas if nlambda(lam, phats) >= num]
    )  # Make sure there's some data in the top bin.

    lhat = None
    for candidate in lambdas:
        lhat = candidate
        ub = p_ub(candidate, delta, yhats, phats, labels)
        if ub < alpha:
            break

    return lhat


def ICP(df, alpha, delta, num, attribute_col, icp_type):
    # Helper function
    def test_performance(df):
        performance = {}

        # Split into retained and abstained groups
        retained = df[df["Abs"] == 0]
        abstained = df[df["Abs"] == 1]

        # Retained metrics
        re_ratio = len(retained) / len(df) if len(df) > 0 else 0
        acc_re = (
            accuracy_score(retained["Y"], retained["Yhat_agnostic"])
            if not df.empty
            else 0
        )
        tp = ((retained["Y"] == 1) & (retained["Yhat_agnostic"] == 1)).sum()
        tn = ((retained["Y"] == 0) & (retained["Yhat_agnostic"] == 0)).sum()
        fp = ((retained["Y"] == 0) & (retained["Yhat_agnostic"] == 1)).sum()
        fn = ((retained["Y"] == 1) & (retained["Yhat_agnostic"] == 0)).sum()
        tpr_re = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_re = fp / (fp + tn) if (fp + tn) > 0 else 0

        # Abstention metrics
        abs_ratio = len(abstained) / len(df) if len(df) > 0 else 0
        acc_abs = (
            accuracy_score(abstained["Y"], abstained["Yhat_agnostic"])
            if not abstained.empty
            else 0
        )
        cost = abs_ratio * acc_abs

        performance["Re_Ratio"] = re_ratio
        performance["ACC"] = acc_re
        performance["TPR"] = tpr_re
        performance["FPR"] = fpr_re
        performance["Abs_Ratio"] = abs_ratio
        performance["ACC_abs"] = acc_abs
        performance["Cost"] = cost

        return performance

    # Change the Phat to 1-Phat depend on yhat_col
    run = df.copy()
    run.loc[run["Yhat_agnostic"] == 0, "Phat"] = 1 - run["Phat"]

    cal_df = run[run["split"] == "calibration"]
    test_df = run[run["split"] == "test"]

    # Return the ICP thresholds for each groups in the calibration set
    if icp_type == "Specific":
        lhat_dict = {
            group: optimal_lambda(
                cal_df[cal_df[attribute_col] == group], alpha, delta, num
            )
            for group in cal_df[attribute_col].unique()
        }
        test_df["Abs"] = 0
        for group, threshold in lhat_dict.items():
            mask = test_df[attribute_col] == group
            test_df.loc[mask, "Abs"] = test_df.loc[mask, "Phat"].apply(
                lambda x: 1 if x < threshold else 0
            )

    elif icp_type == "Agnostic":
        lhat_ag = optimal_lambda(cal_df, alpha, delta, num)
        lhat_dict = {"Agnostic": lhat_ag}
        test_df["Abs"] = test_df["Phat"].apply(lambda x: 1 if x < lhat_ag else 0)

    # Test set performance evaluation
    unique_groups = test_df[attribute_col].unique()
    performance_records = []
    for group in unique_groups:
        sub = test_df[test_df[attribute_col] == group]
        performance = test_performance(sub)
        performance["Group"] = group
        performance_records.append(performance)

    performance_df = pd.DataFrame(performance_records)

    return lhat_dict, performance_df


# Base
def Base(run, attribute_col, yhat_col):
    test_df = run[run["split"] == "test"]

    def baseline_performance(df, yhat_col):
        performance = {}
        acc = accuracy_score(df["Y"], df[yhat_col]) if not df.empty else 0
        tp = ((df["Y"] == 1) & (df[yhat_col] == 1)).sum()
        tn = ((df["Y"] == 0) & (df[yhat_col] == 0)).sum()
        fp = ((df["Y"] == 0) & (df[yhat_col] == 1)).sum()
        fn = ((df["Y"] == 1) & (df[yhat_col] == 0)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        performance["ACC"] = acc
        performance["TPR"] = tpr
        performance["FPR"] = fpr
        performance["Label Type"] = yhat_col
        return performance

    unique_groups = test_df[attribute_col].unique()
    performance_records = []
    for group in unique_groups:
        sub = test_df[test_df[attribute_col] == group]
        performance = baseline_performance(sub, yhat_col)
        performance["Group"] = group
        performance_records.append(performance)
    performance_df = pd.DataFrame(performance_records)
    return performance_df


# Reject Option based Classification (ROC)
def ROC(run, attribute_col, yhat_col):
    cal_df = run[run["split"] == "calibration"]
    test_df = run[run["split"] == "test"]

    # Determine favored and deprived groups based on baseline PPR
    unique_groups = cal_df[attribute_col].unique()
    baseline_ppr = {
        group: cal_df.loc[cal_df[attribute_col] == group, yhat_col].mean()
        for group in unique_groups
    }
    favored_group = max(baseline_ppr, key=baseline_ppr.get)
    deprived_groups = [group for group in unique_groups if group != favored_group]

    # Calculate max[p(C+|X), 1 − p(C+|X)]
    cal_df["Max_Phat"] = cal_df["Phat"].apply(lambda x: max(x, 1 - x))

    # Critical region (0.5 < θ < 1)
    thetas = np.linspace(0.5, 1, 500)
    optimal_theta = None
    max_acc = 0

    # Iterate over candidate lambdas to find the optimal threshold
    for theta in np.flip(thetas):
        # Split the data into rejected and standard groups
        rejected_cal = cal_df[cal_df["Max_Phat"] <= theta].copy()
        standard_cal = cal_df[cal_df["Max_Phat"] > theta].copy()

        # Update predictions in the rejected group
        rejected_cal.loc[
            rejected_cal[attribute_col].isin(deprived_groups), yhat_col
        ] = 1
        rejected_cal.loc[rejected_cal[attribute_col] == favored_group, yhat_col] = 0

        # Combine groups and calculate accuracies
        new_cal = pd.concat([rejected_cal, standard_cal])
        new_acc = accuracy_score(new_cal["Y"], new_cal[yhat_col])

        # Update optimal lambda
        if new_acc > max_acc:
            max_acc = new_acc
            optimal_theta = theta

    # Test
    test_df["Max_Phat"] = test_df["Phat"].apply(lambda x: max(x, 1 - x))
    rejected_test = test_df[test_df["Max_Phat"] <= optimal_theta].copy()
    standard_test = test_df[test_df["Max_Phat"] > optimal_theta].copy()
    rejected_test.loc[rejected_test[attribute_col].isin(deprived_groups), yhat_col] = 1
    rejected_test.loc[rejected_test[attribute_col] == favored_group, yhat_col] = 0
    new_test = pd.concat([rejected_test, standard_test])

    # Test set performance
    def test_performance(df, theta, yhat_col):
        performance = {}
        acc = accuracy_score(df["Y"], df[yhat_col]) if not df.empty else 0
        tp = ((df["Y"] == 1) & (df[yhat_col] == 1)).sum()
        tn = ((df["Y"] == 0) & (df[yhat_col] == 0)).sum()
        fp = ((df["Y"] == 0) & (df[yhat_col] == 1)).sum()
        fn = ((df["Y"] == 1) & (df[yhat_col] == 0)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        # Rejection ratio
        num_rej = df[df["Max_Phat"] <= theta]
        rej_ratio = len(num_rej) / len(df)

        # Save results
        performance["ACC"] = acc
        performance["TPR"] = tpr
        performance["FPR"] = fpr
        performance["Rej_Ratio"] = rej_ratio
        performance["Label Type"] = yhat_col

        return performance

    performance_records = []
    for group in unique_groups:
        sub = new_test[new_test[attribute_col] == group]
        performance = test_performance(sub, optimal_theta, yhat_col)
        performance["Group"] = group
        performance_records.append(performance)

    performance_df = pd.DataFrame(performance_records)

    return performance_df


def evaluation(alpha, delta, runs_dir, save_dir, attribute_col, num):
    """
    Given the runs data and specified hyperparameters ['alpha', 'delta', 'num'], evaluate 5 unfairness mitigation frameworks.
    """
    base = []
    base_cal = []
    roc = []
    icp_ag = []
    icp_sp = []

    for i, filename in enumerate(os.listdir(runs_dir), start=1):
        if filename.endswith(".csv"):
            print(f"Processing run {i}")
            filepath = os.path.join(runs_dir, filename)
            run = pd.read_csv(filepath)
            b = Base(run, attribute_col, "Yhat_agnostic")
            b_cal = Base(run, attribute_col, "Yhat_specific")
            r = ROC(run, attribute_col, "Yhat_agnostic")
            _, i_ag = ICP(run, alpha, delta, num, attribute_col, "Agnostic")
            _, i_sp = ICP(run, alpha, delta, num, attribute_col, "Specific")
            base.append(b)
            base_cal.append(b_cal)
            roc.append(r)
            icp_ag.append(i_ag)
            icp_sp.append(i_sp)

    base_ = pd.concat(base, ignore_index=True)
    base_cal_ = pd.concat(base_cal, ignore_index=True)
    roc_ = pd.concat(roc, ignore_index=True)
    icp_ag_ = pd.concat(icp_ag, ignore_index=True)
    icp_sp_ = pd.concat(icp_sp, ignore_index=True)

    base_.to_csv(save_dir + "base.csv", index=False)
    base_cal_.to_csv(save_dir + "base_cal.csv", index=False)
    roc_.to_csv(save_dir + "roc.csv", index=False)
    icp_ag_.to_csv(save_dir + "icp_ag.csv", index=False)
    icp_sp_.to_csv(save_dir + "icp_sp.csv", index=False)


### Performances & Bias Mitigation
def figure2(dfs, labels, metrics, group_names, super_title):

    # Number of subplots
    num_metrics = len(metrics)

    # Create the figure with a single row of subplots
    fig, axes = plt.subplots(
        1,
        num_metrics,
        figsize=(6 * num_metrics, 6),
        sharey=False,
        constrained_layout=True,
    )

    # Ensure axes is always 1D (even if there is only one metric)
    if num_metrics == 1:
        axes = np.expand_dims(axes, axis=0)

    # Combine all DataFrames with a label
    combined_data = []
    for df, label in zip(dfs, labels):
        df = df.copy()
        df["Source"] = label
        combined_data.append(df)
    combined_df = pd.concat(combined_data)

    # Compute mean and standard error for each metric per group per DataFrame
    stats = []
    for metric in metrics:
        grouped = combined_df.groupby(["Group", "Source"])[metric]
        stats.append(
            grouped.agg(["mean", "sem"])
            .reset_index()
            .rename(columns={"mean": "Mean", "sem": "SE", "Group": "Group"})
        )
    stats_df = pd.concat(stats, keys=metrics, names=["Metric"])

    # Plot each metric
    for col_idx, metric in enumerate(metrics):
        ax = axes[col_idx]
        metric_data = stats_df.loc[metric]

        # Plot side-by-side boxplots
        x_ticks = np.arange(len(metric_data["Group"].unique()))
        width = 0.12  # Adjust bar width for multiple sources
        for j, source in enumerate(labels):
            source_data = metric_data[metric_data["Source"] == source]
            means = source_data["Mean"]
            errors = source_data["SE"]

            ax.bar(
                x_ticks + j * width,
                means,
                yerr=errors,
                width=width,
                label=source,
                capsize=5,
            )

        # Set axis labels and title
        ax.set_title(f"{metric}", fontsize=18)
        ax.set_xticks(x_ticks + (width * (len(labels) - 1) / 2))
        ax.set_xticklabels(
            group_names, ha="center", fontsize=14
        )  # Center x-ticks and make them bigger
        ax.grid(axis="y")

        # Crop the bottom of the box plot by adjusting y-limits, ensuring ymin is 0
        min_value = max(
            metric_data["Mean"].min() - 0.05, 0
        )  # Ensure minimum y-value is 0
        ax.set_ylim(bottom=min_value)

    # Title
    fig.suptitle(super_title, fontsize=20)

    # Add a single legend at the bottom of the figure
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=len(labels),
        fontsize=12,
        bbox_to_anchor=(0.5, -0.1),  # Move legend downward to avoid overlap
    )

    # Adjust the layout and display
    plt.show()


def figure3(base_df, alter_dfs, labels, metrics, group_names):

    # Combine all DataFrames with a label
    combined_data = []
    for df, label in zip(alter_dfs, labels):
        df = df.copy()
        df = df[metrics + ["Group"]]
        diff_df = base_df.copy()
        diff_df = diff_df[metrics + ["Group"]]
        diff_df[metrics] = df[metrics] - base_df[metrics]
        diff_df["Source"] = label
        combined_data.append(diff_df)

    # Combine all processed DataFrames
    combined_df = pd.concat(combined_data)

    # Compute mean and standard error for each metric per group per DataFrame
    stats = []
    for metric in metrics:
        grouped = combined_df.groupby(["Group", "Source"])[metric]
        stats.append(
            grouped.agg(["mean", "sem"])
            .reset_index()
            .rename(columns={"mean": "Mean", "sem": "SE", "Group": "Group"})
        )
    stats_df = pd.concat(stats, keys=metrics, names=["Metric"])
    stats_df["Source"] = pd.Categorical(
        stats_df["Source"], categories=labels, ordered=True
    )

    # Create subplots for each metric
    num_metrics = len(metrics)
    fig, axes = plt.subplots(
        1,
        num_metrics,
        figsize=(6 * num_metrics, 6),
        sharey=False,
        constrained_layout=True,
    )

    # Ensure axes is always 1D (even if there is only one metric)
    if num_metrics == 1:
        axes = np.expand_dims(axes, axis=0)

    # Plotting each metric's results
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        metric_data = stats_df.xs(metric, level="Metric", drop_level=False)

        # Seaborn barplot for each group
        barplot = sns.barplot(
            data=metric_data,
            x="Source",
            y="Mean",
            hue="Group",
            ci=None,
            ax=ax,
            dodge=True,
        )

        # Add SE as whiskers using errorbar
        for patch, (mean, se) in zip(
            barplot.patches, metric_data[["Mean", "SE"]].values
        ):
            height = patch.get_height()
            y_position = patch.get_y() + height

            # Draw error bars (SE as whiskers)
            ax.errorbar(
                patch.get_x() + patch.get_width() / 2,  # X position of the bar
                y_position,  # Y position is the top of the bar
                yerr=se,  # Standard error as whiskers
                fmt="none",  # No marker
                color="black",  # Color of whiskers
                capsize=5,  # Whisker cap size
                elinewidth=2,  # Whisker line width
            )

        # Reverse and plot below 0 values (handle negative values)
        for patch in ax.patches:
            height = patch.get_height()
            if height < 0:
                patch.set_height(
                    abs(height)
                )  # Reverse the direction of negative values
                patch.set_y(patch.get_y() + height)

        # Modify the legend to use `group_names` dictionary for labeling groups
        handles, labels = ax.get_legend_handles_labels()
        # Create a dictionary of the new legend labels
        new_labels = [group_names.get(label, label) for label in labels]

        # Set the new labels to the legend
        ax.legend(handles, new_labels, title="Group")

        ax.set_title(f"{metric} Improvement")
        ax.set_xlabel("Unfairness Mitigation Method")
        ax.set_ylabel("Mean Improvement")
        ax.axhline(0, color="black", linewidth=1)

    plt.show()


def test(base, alter):
    """
    Perform paired t tests.
    By CLT, as N increases, the average accuracy follows normal distribution.
    """

    def df_to_dict_by_runs(df):
        """Converts the DataFrame into a dictionary grouped by unique runs for each group (e.g., "Method 1", "Method 2")."""
        unique_groups = df["Group"].unique()
        runs = df.groupby(df.index // len(unique_groups))  # Group by runs

        result_dict = {}

        # Add individual group results
        for group in unique_groups:
            result_dict[str(group)] = {"ACC": [], "TPR": [], "FPR": []}

        # Process each run to collect results for comparisons and individual metrics
        for _, run in runs:
            group_values = run.set_index("Group")[["ACC", "TPR", "FPR"]]
            for group, values in group_values.iterrows():
                for metric in ["ACC", "TPR", "FPR"]:
                    result_dict[str(group)][metric].append(values[metric])

        return result_dict

    def one_sided_paired_ttest(alter, base, metric, side):
        t_stat, p_value = ttest_rel(alter[metric], base[metric], alternative=side)
        t_stat = round(t_stat, 4)
        print(f"t-stat: {t_stat}, p-value: {p_value}")
        if p_value < 0.05:
            if side == "greater":
                print(f"Method 2's {metric} significantly greater than Method 1.")
            elif side == "less":
                print(f"Method 2's {metric} significantly less than Method 1.")
        else:
            print(f"No statistical evidence.")

    base_dict = df_to_dict_by_runs(base)
    alter_dict = df_to_dict_by_runs(alter)

    # Tests start here
    for group in base_dict:
        print(f"Comparing group: {group}")
        # Improvement from baseline?
        one_sided_paired_ttest(alter_dict[group], base_dict[group], "ACC", "greater")
        one_sided_paired_ttest(alter_dict[group], base_dict[group], "TPR", "greater")
        one_sided_paired_ttest(alter_dict[group], base_dict[group], "FPR", "less")


### Decision Threshold Optimization
def adjustable_alpha(
    runs_dir,
    delta,
    attribute_col,
    num,
    min_alpha,
    max_alpha,
    icp_type,
    max_files,
    out_dir,
):

    result = {}
    alpha_lst = [round(val, 2) for val in np.arange(min_alpha, max_alpha, 0.01)]
    run_count = 0
    empty_metrics = {
        "Lambda": [],
        "Re_Ratio": [],
        "ACC": [],
        "TPR": [],
        "FPR": [],
        "Abs_Ratio": [],
        "ACC_abs": [],
        "Cost": [],
    }

    for i, filename in enumerate(os.listdir(runs_dir), start=1):
        if not filename.endswith(".csv"):
            continue
        filepath = os.path.join(runs_dir, filename)
        run = pd.read_csv(filepath)
        grouped_data = run.groupby(attribute_col)

        for alpha in alpha_lst:
            print(f"Processing run {i}, alpha {alpha}")
            l, p = ICP(run, alpha, delta, num, attribute_col, icp_type)

            for group, group_data in grouped_data:
                if group not in result:
                    result[group] = {
                        alpha: {k: v[:] for k, v in empty_metrics.items()}
                        for alpha in alpha_lst
                    }

                # Update metrics for the group and alpha
                if icp_type == "Specific":
                    result[group][alpha]["Lambda"].append(l[group])
                elif icp_type == "Agnostic":
                    result[group][alpha]["Lambda"].append(l["Agnostic"])

                # Extend values in a single operation
                group_p = p[p["Group"] == group]
                result[group][alpha]["Re_Ratio"].extend(group_p["Re_Ratio"].tolist())
                result[group][alpha]["ACC"].extend(group_p["ACC"].tolist())
                result[group][alpha]["TPR"].extend(group_p["TPR"].tolist())
                result[group][alpha]["FPR"].extend(group_p["FPR"].tolist())
                result[group][alpha]["Abs_Ratio"].extend(group_p["Abs_Ratio"].tolist())
                result[group][alpha]["ACC_abs"].extend(group_p["ACC_abs"].tolist())
                result[group][alpha]["Cost"].extend(group_p["Cost"].tolist())

        run_count += 1
        if run_count >= max_files:
            print(f"Processed {max_files} files, stopping.")
            break

    rows = []
    for group, alpha_data in result.items():
        for alpha, metrics in alpha_data.items():
            for i in range(len(metrics["Lambda"])):
                rows.append(
                    {
                        "Group": group,
                        "Alpha": alpha,
                        "Lambda": metrics["Lambda"][i],
                        "Re_Ratio": metrics["Re_Ratio"][i],
                        "ACC": metrics["ACC"][i],
                        "TPR": metrics["TPR"][i],
                        "FPR": metrics["FPR"][i],
                        "Abs_Ratio": metrics["Abs_Ratio"][i],
                        "ACC_abs": metrics["ACC_abs"][i],
                        "Cost": metrics["Cost"][i],
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir + "alphas.csv", index=False)

    return result


def plot_alpha(df1, df2):
    # Set up the figure with subplots for each metric
    metrics = ["ACC", "TPR", "FPR", "ACC_abs", "Abs_Ratio", "Cost"]
    titles = [
        "Selected ACC",
        "Selected TPR",
        "Selected FPR",
        "Abstained ACC",
        "Abstention Ratio",
        "ICP Implementation Cost",
    ]
    plt_idx = ["a", "b", "c", "d", "e", "f"]
    group_idx = ["White", "Asian", "Black"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Define a color palette for the groups
    colors = sns.color_palette("Set1", len(group_idx))  # Adjust palette as needed

    # Group by 'Group' to plot each group with different colors
    groups = df1["Group"].unique()

    for i, metric in enumerate(metrics):
        ax = axes[i // 3, i % 3]

        for j, group in enumerate(groups):
            group1_data = df1[df1["Group"] == group]
            group2_data = df2[df2["Group"] == group]

            # Group by confidence and compute mean and standard error for each metric
            confidence_group1 = group1_data.groupby("Confidence")[metric].agg(
                ["mean", "sem"]
            )
            confidence_group2 = group2_data.groupby("Confidence")[metric].agg(
                ["mean", "sem"]
            )

            # Use the color from the palette for the current group
            group_color = colors[j]

            # Plot the data: error bars are given by standard error
            ax.errorbar(
                confidence_group1.index,
                confidence_group1["mean"],
                yerr=confidence_group1["sem"],
                fmt="o",
                label=f"{group_idx[j]} - FairICP",
                capsize=5,
                elinewidth=2,
                linestyle="-",
                marker="o",
                color=group_color,
            )
            ax.errorbar(
                confidence_group2.index,
                confidence_group2["mean"],
                yerr=confidence_group2["sem"],
                fmt="o",
                label=f"{group_idx[j]} - ICP_org",
                capsize=5,
                elinewidth=2,
                linestyle="--",
                marker="o",
                color=group_color,
            )

        # Set all unique confidence values as x-ticks
        unique_confidences = sorted(df1["Confidence"].unique())
        ax.set_xticks(unique_confidences)

        # Labeling and aesthetics
        ax.set_title(f"({plt_idx[i]}): {titles[i]}")
        ax.set_xlabel("Confidence Level")
        ax.set_ylabel(metric)
        ax.legend()

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()


def simulate_data(
    n_samples_group1,
    n_samples_group2,
    acc_group1,
    acc_group2,
    correct_mean_group1,
    correct_sd_group1,
    incorrect_mean_group1,
    incorrect_sd_group1,
    correct_mean_group2,
    correct_sd_group2,
    incorrect_mean_group2,
    incorrect_sd_group2,
):
    # Helper function to generate data for a single group
    def generate_group_data(
        n_samples,
        accuracy,
        correct_mean,
        correct_sd,
        incorrect_mean,
        incorrect_sd,
        group_label,
    ):
        n_correct = int(n_samples * accuracy)
        n_incorrect = n_samples - n_correct

        # Generate phat for correct predictions
        correct_phat = np.random.normal(
            loc=correct_mean, scale=correct_sd, size=n_correct
        )
        correct_phat = np.clip(
            correct_phat, 0.5, 1
        )  # Ensure phat values are in [0.5, 1]

        # Assign Y and Yhat for correct predictions
        correct_yhat = np.random.choice([1, 0], size=n_correct, p=[0.5, 0.5])
        correct_y = correct_yhat  # For correct predictions, Y == Yhat

        # Generate phat for incorrect predictions
        incorrect_phat = np.random.normal(
            loc=incorrect_mean, scale=incorrect_sd, size=n_incorrect
        )
        incorrect_phat = np.clip(
            incorrect_phat, 0.5, 1
        )  # Ensure phat values are in [0.5, 1]

        # Assign Y and Yhat for incorrect predictions
        incorrect_yhat = np.random.choice([1, 0], size=n_incorrect, p=[0.5, 0.5])
        incorrect_y = 1 - incorrect_yhat  # For incorrect predictions, Y != Yhat

        # Combine correct and incorrect predictions
        phat = np.concatenate([correct_phat, incorrect_phat])
        yhat = np.concatenate([correct_yhat, incorrect_yhat])
        y = np.concatenate([correct_y, incorrect_y])

        # Shuffle the data
        indices = np.random.permutation(n_samples)
        phat = phat[indices]
        yhat = yhat[indices]
        y = y[indices]

        # Create DataFrame
        return pd.DataFrame(
            {
                "Group": [group_label] * n_samples,
                "Phat": phat,
                "Yhat_agnostic": yhat,
                "Y": y,
            }
        )

    # Generate data for both groups
    group1_data = generate_group_data(
        n_samples_group1,
        acc_group1,
        correct_mean_group1,
        correct_sd_group1,
        incorrect_mean_group1,
        incorrect_sd_group1,
        group_label=0,
    )
    group2_data = generate_group_data(
        n_samples_group2,
        acc_group2,
        correct_mean_group2,
        correct_sd_group2,
        incorrect_mean_group2,
        incorrect_sd_group2,
        group_label=1,
    )

    # Combine and return
    return pd.concat([group1_data, group2_data], ignore_index=True)


def visualize_simulation(data):
    """
    Plot histograms of `phat` for correct and incorrect predictions for each group.

    Args:
        data (DataFrame): DataFrame containing columns 'Group', 'Phat', 'Y', and 'Yhat'.
    """
    # Get unique groups
    groups = data["Group"].unique()
    group_names = {0: "H", 1: "L"}

    # Set up subplots
    fig, axes = plt.subplots(1, len(groups), figsize=(15, 5), sharey=True)

    if len(groups) == 1:
        axes = [axes]  # Ensure axes is iterable if there's only one group

    # Loop through each group
    for i, group in enumerate(groups):
        group_data = data[data["Group"] == group]

        # Separate correct and incorrect predictions
        correct = group_data[group_data["Y"] == group_data["Yhat_agnostic"]]
        incorrect = group_data[group_data["Y"] != group_data["Yhat_agnostic"]]

        # Plot histograms
        sns.histplot(
            correct["Phat"],
            bins=30,
            color="green",
            alpha=0.7,
            label="Correct",
            kde=True,
            ax=axes[i],
        )
        sns.histplot(
            incorrect["Phat"],
            bins=30,
            color="turquoise",
            alpha=0.7,
            label="Incorrect",
            kde=True,
            ax=axes[i],
        )

        # Aesthetics
        axes[i].set_title(
            f"Group {group_names[group]} - Prediction pobability distribution"
        )
        axes[i].set_xlabel("Prediction probability")
        axes[i].set_ylabel("Frequency")
        axes[i].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()


def alpha_calsize(
    df_dir,
    attribute_col,
    icp_type,
    delta,
    num,
    min_alpha,
    max_alpha,
    calsize_lst,
    out_dir,
    n_runs,
):
    # Simulation df
    sim_df = pd.read_csv(df_dir)
    sim_df["split"] = "test"

    # Hyperparameter grid
    alpha_lst = [round(val, 2) for val in np.arange(min_alpha, max_alpha, 0.02)]
    alpha_grid, calsize_grid = np.meshgrid(alpha_lst, calsize_lst)
    parameter_grid = np.column_stack((alpha_grid.flatten(), calsize_grid.flatten()))

    # Initialize results
    result = {}

    # Process each hyperparameter combination
    for alpha, calsize in parameter_grid:
        calsize = int(calsize)
        print(f"Alpha {alpha}, Cal Size {calsize}")
        metrics_collection = []

        for _ in range(n_runs):
            sim_df["split"] = "test"  # Reset split column to 'test'

            # Adjust calibration size
            if icp_type == "Agnostic":
                cal_indices = np.random.choice(
                    sim_df.index, size=calsize, replace=False
                )
                sim_df.loc[cal_indices, "split"] = "calibration"

            elif icp_type == "Specific":
                for group_name, group_indices in sim_df.groupby(
                    attribute_col
                ).groups.items():
                    group_size = len(group_indices)
                    selected_indices = np.random.choice(
                        group_indices, size=min(calsize, group_size), replace=False
                    )
                    sim_df.loc[selected_indices, "split"] = "calibration"

            # Perform ICP
            l, p = ICP(sim_df, alpha, delta, num, attribute_col, icp_type)

            # Collect metrics for this run
            run_metrics = {}
            for group, group_data in sim_df.groupby(attribute_col):
                group_p = p[p["Group"] == group]
                run_metrics[group] = {
                    "Lambda": l[group] if icp_type == "Specific" else l["Agnostic"],
                    "Re_Ratio": np.mean(group_p["Re_Ratio"]),
                    "ACC": np.mean(group_p["ACC"]),
                    "TPR": np.mean(group_p["TPR"]),
                    "FPR": np.mean(group_p["FPR"]),
                    "Abs_Ratio": np.mean(group_p["Abs_Ratio"]),
                    "ACC_abs": np.mean(group_p["ACC_abs"]),
                    "Cost": np.mean(group_p["Cost"]),
                }
            metrics_collection.append(run_metrics)

        # Calculate means across runs for this combination
        key = f"{alpha}/{calsize}"
        result[key] = {}
        for group in metrics_collection[0].keys():  # Iterate over groups
            group_means = {
                metric: np.mean([run[group][metric] for run in metrics_collection])
                for metric in metrics_collection[0][group].keys()
            }
            result[key][group] = group_means

    # Convert results to a DataFrame
    rows = []
    for col_name, col_data in result.items():
        alpha, cal_size = col_name.split("/")
        alpha = float(alpha)
        cal_size = int(cal_size)

        # Flatten the dictionary and convert the metrics into columns
        for group, metrics in col_data.items():
            row = {"alpha": alpha, "cal_size": cal_size, "group": group}
            row.update(metrics)
            rows.append(row)

    # Create the DataFrame from the rows
    df = pd.DataFrame(rows)
    df["cal_size"] = np.log10(df["cal_size"])
    df.to_csv(out_dir + "alpha_calsize.csv", index=False)


def plot_3d(df, x_col, y_col, z_col, group_col, figsize=(10, 10), degree=4):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Convert to confidence
    df["confidence"] = 1 - df["alpha"]

    # Group by the specified column
    groups = df.groupby(group_col)
    group_names = {0: "H", 1: "L"}

    # Define a polynomial surface fitting function
    def poly_surface(xy, *coeffs):
        x, y = xy
        terms = [coeffs[0]]  # Intercept
        idx = 1
        for d in range(1, degree + 1):
            for i in range(d + 1):
                terms.append(coeffs[idx] * (x ** (d - i)) * (y**i))
                idx += 1
        return sum(terms)

    for name, group in groups:
        # Prepare data for the current group
        X = group[x_col].values
        Y = group[y_col].values
        Z = group[z_col].values

        # Generate initial coefficients and fit the model
        num_coeffs = (
            (degree + 1) * (degree + 2) // 2
        )  # Number of coefficients for given degree
        initial_coeffs = np.zeros(num_coeffs)
        popt, _ = curve_fit(poly_surface, (X, Y), Z, p0=initial_coeffs)

        # Create a mesh grid for the surface
        x_range = np.linspace(X.min(), X.max(), 50)
        y_range = np.linspace(Y.min(), Y.max(), 50)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        Z_grid = poly_surface((X_grid.ravel(), Y_grid.ravel()), *popt).reshape(
            X_grid.shape
        )

        # Scatter plot for the current group
        ax.scatter(X, Y, Z)

        # Surface plot for the fitted surface
        ax.plot_surface(
            X_grid, Y_grid, Z_grid, alpha=0.5, label=f"Group {group_names[name]}"
        )

    # Limit x-ticks and y-ticks to unique values in the DataFrame
    x_ticks = sorted(df[x_col].unique())
    ax.set_xticks(x_ticks)
    x_tick_rename = {1: 10, 2: 100, 3: 1000, 4: 10000, 5: 100000}
    x_tick_labels = [x_tick_rename.get(x, str(x)) for x in x_ticks]
    ax.set_xticklabels(x_tick_labels)

    ax.set_yticks(sorted(df[y_col].unique()))

    # Label the axes
    ax.set_xlabel("Calibration Size")
    ax.set_ylabel("Confidence Level")
    ax.set_zlabel("Cost", rotation=90)

    # Add legend
    ax.legend()

    # Show plot
    plt.tight_layout()
    plt.show()


### Error Analysis
def plot_phat_distribution(df1, df2, df3):

    def plot_single(df, ax, title):
        df["Phat"] = df.apply(
            lambda row: row["Phat"] if row["Yhat_agnostic"] == 1 else 1 - row["Phat"],
            axis=1,
        )
        correct_preds = df[df["Y"] == df["Yhat_agnostic"]]
        incorrect_preds = df[df["Y"] != df["Yhat_agnostic"]]

        ax.hist(
            correct_preds["Phat"],
            bins=30,
            alpha=0.6,
            label="Correct Predictions",
            color="blue",
        )
        ax.hist(
            incorrect_preds["Phat"],
            bins=30,
            alpha=0.6,
            label="Incorrect Predictions",
            color="red",
        )

        ax.set_title(title)
        ax.set_xlabel("Phat")
        ax.set_ylabel("Frequency")
        ax.legend()

    # Set up the subplots: 3 rows and 1 column
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot for df1
    plot_single(df1, axes[0], "ICM/NICM Phat Distribution")
    # Plot for df2
    plot_single(df2, axes[1], "CheXpert Phat Distribution")
    # Plot for df3
    plot_single(df3, axes[2], "ISIC18 Phat Distribution")

    # Adjust layout
    plt.tight_layout()
    plt.show()
