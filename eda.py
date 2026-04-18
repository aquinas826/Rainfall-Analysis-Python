import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("Sub_Division_IMD_2017.csv")

# Histogram
plt.hist(df[df.columns[1]], bins=30, color="skyblue")
plt.title("Histogram of Rainfall")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Violin plot
sns.violinplot(x=df.columns[1], y=df.columns[2], data=df)
plt.title("Rainfall Distribution Visualization")
plt.show()

# Pairplot
sns.pairplot(df[df.select_dtypes(include=['float64','int64']).columns[:5]])
plt.show()
