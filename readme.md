# Rainfall-Analysis-Python


Problem Statement
Rainfall variability is a critical factor in India, influencing agriculture, water supply, and socio-economic planning. This project uses the IMD Sub-Division Rainfall Dataset (2017) to explore rainfall behavior, detect anomalies, and test predictive modeling with Linear Regression. The workflow is structured around five subjective objectives, each producing clear outputs for screenshots and reporting.

###Objectives, Solutions, and Conclusions

#Objective 1: Understand Rainfall Variability
*Solution: 
I began by cleaning the dataset. Missing values were replaced with medians, duplicates removed, numerical features normalized, categorical variables encoded, and outliers eliminated using the IQR method. This ensured the dataset was consistent and reliable.  
code:
df = df.fillna(df.median(numeric_only=True))
df = df.drop_duplicates()
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
encoder = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = encoder.fit_transform(df[col])
*Conclusion:
After preprocessing, the dataset was free of inconsistencies and ready for deeper analysis. The shape reduction confirmed that extreme values were successfully removed.


#Objective 2: Interpret Seasonal and Annual Rainfall Trends
*Solution:
Exploratory Data Analysis (EDA) was performed to summarize rainfall values and visualize seasonal trends. Histograms showed distribution, while correlation heatmaps revealed relationships between rainfall variables.  
code:
print(df.describe())
plt.hist(df[numeric_cols[0]], bins=30, color="skyblue")
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
*Conclusion:
Seasonal and annual rainfall trends varied across subdivisions. Correlation analysis highlighted strong interdependencies, confirming the importance of seasonal rainfall in climate studies.

#Objective 3: Evaluate Socio-Economic Implications of Anomalies
*Solution:  
A t-test was used to compare rainfall variables, checking for significant differences that could indicate anomalies with socio-economic impacts.  
code:
stat, p = stats.ttest_ind(df[numeric_cols[0]], df[numeric_cols[1]])

*Conclusion: 
The t-test revealed significant differences in rainfall patterns, suggesting possible impacts on agriculture and water availability in affected regions.

#Objective 4: Explore Innovative Visualization Techniques
*Solution:
Advanced plots were applied to uncover hidden patterns. Violin plots showed distribution spread, while pairplots highlighted relationships among multiple rainfall features.  
code:
sns.violinplot(x=numeric_cols[0], y=numeric_cols[1], data=df)
sns.pairplot(df[numeric_cols[:5]])

*Conclusion:
These visualizations provided richer insights into rainfall distribution, making the analysis more intuitive and visually engaging.

#Objective 5: Assess ML Potential in Predicting Rainfall
*Solution:
Linear Regression was applied to predict rainfall. The model was evaluated using R², MSE, and RMSE metrics, and a scatter plot compared actual vs predicted values.  
code:
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
print("R²:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

*Conclusion:
Linear Regression showed predictive potential, with measurable accuracy. The scatter plot confirmed the model’s ability to approximate rainfall trends.


#Final Notes
This project demonstrates how preprocessing, statistical analysis, visualization, and machine learning can be combined to analyze rainfall variability. Each objective produced clear outputs, making the workflow reproducible and suitable for academic reporting and GitHub documentation.
