import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

dataset_path = "dataset/iris.csv"

df = pd.read_csv(dataset_path)

print(df.shape)
print(df.head())

kvalues = [2,3,4]

X = df.iloc[:,:2].values

plt.figure(figsize=(10,6))
plt.subplot(2,2,1)
plt.scatter(X[:,0],X[:,1],c='gray')
plt.title("Before clustering")

for k in [2,3,4]:
    plt.subplot(2,2,k)
    km = KMeans(k).fit_predict(X)
    plt.scatter(X[:,0],X[:,1],km)
    plt.title(f'K = {k}')

plt.tight_layout()
plt.show()


#tryna do it from scratch
