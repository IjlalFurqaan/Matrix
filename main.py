from data_preparation import load_data
from analysis import (construct_biadjacency_matrix, perform_svd, plot_singular_values, analyze_singular_vectors,
                      check_randomness, apply_idf_weighting, perform_svd_idf)

# URL of the dataset
data_url = 'http://nlp.uned.es/lihlith-project/cook/Cookbook.kgraph.v1.4.txt'

# Load and prepare data
filtered_data = load_data(data_url)

# Construct biadjacency matrix
biadjacency_matrix = construct_biadjacency_matrix(filtered_data)

# Perform SVD and analyze singular values/vectors
U, Sigma, VT = perform_svd(biadjacency_matrix)
plot_singular_values(Sigma, title='Singular Values of the Biadjacency Matrix')
key_recipes, key_ingredients = analyze_singular_vectors(U, VT, biadjacency_matrix)

# Print key recipes and ingredients
print("Key Recipes:", biadjacency_matrix.index[key_recipes][:10])
print("Key Ingredients:", biadjacency_matrix.columns[key_ingredients][:10])

# Check randomness
random_matrix = check_randomness(biadjacency_matrix)
U_rand, Sigma_rand, VT_rand = perform_svd(random_matrix)
plot_singular_values(Sigma_rand, title='Singular Values of the Random Biadjacency Matrix')

# Apply IDF-weighting and analyze results
idf_weighted_matrix = apply_idf_weighting(biadjacency_matrix)
U_idf, Sigma_idf, VT_idf = perform_svd_idf(idf_weighted_matrix)
plot_singular_values(Sigma_idf, title='Singular Values of the IDF-Weighted Biadjacency Matrix')

# Print key recipes and ingredients for IDF-weighted matrix
key_recipes_idf, key_ingredients_idf = analyze_singular_vectors(U_idf, VT_idf, idf_weighted_matrix)
print("Key Recipes (IDF-Weighted):", idf_weighted_matrix.index[key_recipes_idf][:10])
print("Key Ingredients (IDF-Weighted):", idf_weighted_matrix.columns[key_ingredients_idf][:10])
