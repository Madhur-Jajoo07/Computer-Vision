#%%
import pandas as pd
import numpy as np
#%%
def read_csv(path):
    data= pd.read_csv(path,index_col=False)
    data.drop('Unnamed: 0',axis = 1)
    y = data.iloc[:,-1:]
    data_y = []
    for i in range(len(y)) :
        data_y.append(int(y.loc[i][0][1]))
    data_x = data.iloc[:,:-1]
    data_x.drop('Unnamed: 0',axis = 1)
    return data_x , data_y

# %%
data_train_x , data_train_y = read_csv('train_q1.csv')
data_test_x , data_test_y = read_csv('test_q1.csv')

# %%
def svd(k):
    u = np.empty((data_train_x.shape[0], k))
    sigma = np.empty(k)
    v_transpose = np.empty((k, data_train_x.shape[1]))
    block_size = 10000 # 60k * 60k is computationally hard so taking the top 10k
    for i in range(0, k, block_size):
        j = min(i + block_size, k)
        u_block, sigma_block, v_t_block = np.linalg.svd(data_train_x, full_matrices=False)
        u[:, i:j] = u_block[:, :j - i]
        sigma[i:j] = sigma_block[:j - i]
        v_transpose[i:j, :] = v_t_block[:j - i, :]
    return u,sigma,v_transpose

# %%
reconstruction_errors = []
k_values=range(1,11)
for k in range(1,11):
    u, sigma, v_transpose = svd(k)
    u = u[:, :k]
    sigma = np.diag(sigma[:k])
    v_transpose = v_transpose[:k, :]
    reconstructed = np.dot(u, np.dot(sigma, v_transpose))
    error = np.linalg.norm(data_train_x - reconstructed)
    reconstruction_errors.append(error)
#%%
import matplotlib.pyplot as plt
plt.plot(k_values, reconstruction_errors, marker='o')
plt.title('Reconstruction Error vs. k')
plt.xlabel('k (Number of Singular Vectors)')
plt.ylabel('Reconstruction Error')
plt.grid()
plt.show()

# %%
