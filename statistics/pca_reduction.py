from sklearn.decomposition import PCA


def pca(train_input_as_matrix, test_input_as_matrix):
    print(f'Train shape before pca: {train_input_as_matrix.shape}')
    print(f'Test shape before pca: {test_input_as_matrix.shape}')
    pca_method = PCA(n_components=30)
    train_input_as_matrix = pca_method.fit_transform(train_input_as_matrix)
    test_input_as_matrix = pca_method.transform(test_input_as_matrix)
    print(f'Train shape after pca: {train_input_as_matrix.shape}')
    print(f'Test shape after pca: {test_input_as_matrix.shape}')
    return train_input_as_matrix, test_input_as_matrix
