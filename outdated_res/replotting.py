import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import os 
from scipy.stats import zscore


output_path = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/npt_responses/pretrained_graded.json"
figures_path = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/figures"

# Load your JSON data from a file
with open(output_path, "r") as f:
    data = json.load(f)

# Convert JSON to DataFrame
df = pd.DataFrame(data)

for col in ['correctness', 'clarity', 'reasoning', 'notation']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing values in score column

# Standardize the scores
df_z = df[['correctness', 'clarity', 'reasoning', 'notation']].apply(zscore)

# Add z-scores back to the original dataframe
for col in df_z.columns:
    df[f'z_{col}'] = df_z[col]

# Compute difference between reasoning and clarity
df['z_diff_rs_crr'] = df['z_reasoning'] - df['z_correctness']

# Find outliers: large difference (e.g., |diff| > 1.5 standard deviations)
outliers = df[abs(df['z_diff_rs_crr']) > 1.5]

# Show first few outliers
print(outliers[['unique_id', 'correctness', 'clarity', 'reasoning', 'notation', 'z_diff_rs_crr']].head())
outliers.to_json("/n/netscratch/dam_lab/Lab/hdiaz/ft_project/npt_responses/outliers_RSCR.json", index=False)
# Show results
print(outliers[['unique_id', 'correctness', 'reasoning', 'z_diff_rs_crr']].head())

