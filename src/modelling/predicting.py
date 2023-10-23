def train_model(x_train: csr_matrix, y_train: np.ndarray):
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    return lr