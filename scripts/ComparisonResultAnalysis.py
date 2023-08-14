import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('evaluation_comparison_result.csv')

# error
df['error_1'] = (df['repo_sim (No. 1)'] - df['repo_sim (RepoSim4Py)']).abs()
df['error_2'] = (df['repo_sim (No. 2)'] - df['repo_sim (RepoSim4Py)']).abs()
df['error_3'] = (df['repo_sim (No. 3)'] - df['repo_sim (RepoSim4Py)']).abs()

# mean error
mean_error_1 = df['error_1'].mean()  # 0.13
mean_error_2 = df['error_2'].mean()  # 0.17
mean_error_3 = df['error_3'].mean()  # 0.20

# max error
max_error_1 = df['error_1'].max()  # 0.44
max_error_2 = df['error_2'].max()  # 0.36
max_error_3 = df['error_3'].max()  # 0.65


# 1. Error Distribution Chart
fig, axs = plt.subplots(1,3,figsize=(15,5))
sns.histplot(df['error_1'], ax=axs[0])
sns.histplot(df['error_2'], ax=axs[1])
sns.histplot(df['error_3'], ax=axs[2])
plt.show()

# 2. error correlation
print(df[['error_1','error_2','error_3']].corr())

# 3. Scatter Plots of Error vs. Similarity
fig, axs = plt.subplots(1,3,figsize=(15,5))
axs[0].scatter(df['repo_sim (RepoSim4Py)'], df['error_1'])
axs[0].set_xlabel('Similarity Score')
axs[0].set_ylabel('Error of Model 1')

axs[1].scatter(df['repo_sim (RepoSim4Py)'], df['error_2'])
axs[1].set_xlabel('Similarity Score')
axs[1].set_ylabel('Error of Model 2')

axs[2].scatter(df['repo_sim (RepoSim4Py)'], df['error_3'])
axs[2].set_xlabel('Similarity Score')
axs[2].set_ylabel('Error of Model 3')
plt.show()