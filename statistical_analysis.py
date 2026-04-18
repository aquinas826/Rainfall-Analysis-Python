import pandas as pd
from scipy import stats

df = pd.read_csv("Sub_Division_IMD_2017.csv")

# Example t-test between two numeric columns
stat, p = stats.ttest_ind(df[df.columns[1]], df[df.columns[2]])
print(f"T-test result: stat={stat:.4f}, p={p:.4f}")
