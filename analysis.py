import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

def construct_biadjacency_matrix(filtered_data):
    biadjacency_matrix = filtered_data.pivot_table(index='RecipeID', columns='Ingredient', aggfunc='size', fill_value=0)
    return biadjacency_matrix

def perform_svd(biadjacency_matrix):
    sparse_matrix = csr_matrix(biadjacency_matrix, dtype=float)  # Ensure the matrix is of floating-point type
    U, Sigma, VT = svds(sparse_matrix, k=min(sparse_matrix.shape)-1)
    return U, Sigma, VT

def plot_singular_values(Sigma, title='Singular Values'):
    plt.plot(Sigma[::-1])
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.show()

def analyze_singular_vectors(U, VT, biadjacency_matrix):
    u1 = U[:, -1]
    v1 = VT[-1, :]
    key_recipes = np.argsort(np.abs(u1))[::-1]
    key_ingredients = np.argsort(np.abs(v1))[::-1]
    return key_recipes, key_ingredients

def check_randomness(biadjacency_matrix):
    p_recipes = biadjacency_matrix.sum(axis=1) / biadjacency_matrix.sum().sum()
    q_ingredients = biadjacency_matrix.sum(axis=0) / biadjacency_matrix.sum().sum()
    N = biadjacency_matrix.sum().sum()
    random_matrix = np.random.binomial(1, p_recipes.values[:, None] * q_ingredients.values, size=biadjacency_matrix.shape)
    return random_matrix

def apply_idf_weighting(biadjacency_matrix):
    n = biadjacency_matrix.shape[0]
    idf = np.log(n / (biadjacency_matrix.sum(axis=0) + 1))
    idf_weighted_matrix = biadjacency_matrix * idf
    return idf_weighted_matrix

def perform_svd_idf(idf_weighted_matrix):
    sparse_matrix = csr_matrix(idf_weighted_matrix, dtype=float)  # Ensure the matrix is of floating-point type
    U_idf, Sigma_idf, VT_idf = svds(sparse_matrix, k=min(sparse_matrix.shape)-1)
    return U_idf, Sigma_idf, VT_idf
