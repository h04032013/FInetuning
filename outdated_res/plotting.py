import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

output_path="/n/netscratch/dam_lab/Lab/hdiaz/ft_project/npt_responses/pretrained_graded.json"
figures_path ="/n/netscratch/dam_lab/Lab/hdiaz/ft_project/figures"

# Load your JSON file
with open(output_path, "r") as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert score columns to numeric
score_cols = ["correctness", "clarity", "reasoning", "notation", "total"]
#df[score_cols] = df[score_cols].apply(pd.to_numeric)

for col in score_cols:
    df[col] = df[col].astype(str).str.extract(r"(\d+)")[0].astype(float)

# Set seaborn style
sns.set_theme(style="whitegrid")

# --- Plot 1: Accuracy by Method ---
accuracy_data = {
    "Method": ["api_equiv", "is_equiv", "Rubric Correct (score=5)"],
    "Accuracy": [
        df["api_equiv"].mean(),
        df["is_equiv"].mean(),
        (df["correctness"] == 5).mean()
    ]
}
acc_df = pd.DataFrame(accuracy_data)

plt.figure(figsize=(8, 5))
sns.barplot(data=acc_df, x="Method", y="Accuracy")
plt.title("Accuracy by Method")
plt.ylim(0, 1)
plt.ylabel("Proportion Correct")
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "accuracy_by_method3.png"))
plt.close()

# --- Plot 2: Score Distributions ---
for score in score_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[score], bins=5, kde=False)
    plt.title(f"Distribution of {score.capitalize()} Scores")
    plt.xlabel(score.capitalize())
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f"{score}_distribution3.png"))
    plt.close()

# --- Plot 3: Total Score by Equivalence ---
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="is_equiv", y="total")
plt.title("Total Score by is_equiv")
plt.xlabel("is_equiv")
plt.ylabel("Total Score")
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "total_by_is_equiv3.png"))
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="api_equiv", y="total")
plt.title("Total Score by api_equiv")
plt.xlabel("api_equiv")
plt.ylabel("Total Score")
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "total_by_api_equiv3.png"))
plt.close()

# --- Plot 4: Performance by Problem Type ---
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="type", y="total", ci=None)
plt.title("Average Total Score by Problem Type")
plt.ylabel("Average Total Score")
plt.xlabel("Problem Type")
plt.tight_layout()
plt.savefig(os.path.join(os.path.join(figures_path, "score_by_problem_type3.png")))
plt.close()

# --- Plot 5: Equivalence Agreement Heatmap ---
equiv_comparison = df.groupby(["api_equiv", "is_equiv"]).size().reset_index(name='count')
pivot_table = equiv_comparison.pivot(index="api_equiv", columns="is_equiv", values="count").fillna(0)

plt.figure(figsize=(6, 5))
sns.heatmap(pivot_table, annot=True, fmt="g", cmap="Blues", cbar=False)
plt.title("Equivalence Check Agreement (api_equiv vs is_equiv)")
plt.xlabel("is_equiv")
plt.ylabel("api_equiv")
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "equiv_agreement_heatmap3.png"))
plt.close()

# --- Plot 6: Heatmap of Score by Level and Type ---
score_heatmap = df.groupby(["level", "type"])["total"].mean().unstack()

plt.figure(figsize=(8, 6))
sns.heatmap(score_heatmap, annot=True, cmap="YlGnBu", fmt=".1f")
plt.title("Average Total Score by Level and Type")
plt.ylabel("Level")
plt.xlabel("Problem Type")
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "score_by_level_and_type3.png"))
plt.close()

print("All plots saved as PNG files.")
