import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv("Sub_Division_IMD_2017.csv")

# Handle missing values
df = df.fillna(df.median(numeric_only=True))

# Remove duplicates
df = df.drop_duplicates()

# Normalize numerical features
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=['float64','int64']).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Encode categorical variables
encoder = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = encoder.fit_transform(df[col])

print("Preprocessing completed")
