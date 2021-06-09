import seaborn as sns;
import matplotlib.pyplot as plt
import numpy as np

co_matrix = np.zeros((4, 4))
# [28500, 28473, 28532, 28495], label clas
plt.rc('font', family='Times New Roman')
sns.set(font="Times New Roman", font_scale=1.3)
ax = sns.heatmap(co_matrix / np.diag(co_matrix), xticklabels=["Politics", "Sports", "Business", "Tech"],
                 yticklabels=["Politics", "Sports", "Business", "Tech"], annot=True, fmt=".2f", cmap="YlGnBu_r")
plt.xlabel("Class Label")
plt.ylabel("Class Label")
# World (0), Sports (1), Business (2), Sci/Tech (3).
plt.yticks(rotation=60)
