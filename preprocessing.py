import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


def generate_data_delay(dataset, testnum, input_length, output_length, n_delays):
    """
    Generate data with intervals between input and output.

    Args:
        dataset: Input dataset. Shape: (input_length, )
        testnum: Sample num for test.
        input_length: Input length for each sample.
        output_length: Output length for each sample.
        n_delays: Interval between input and output.

    Returns: [x_train, x_test, y_train, y_test].
        x_train shape: (batch_size, input_length)
        x_test shape: (batch_size, input_length)
        y_train shape: (batch_size, output_length)
        y_test shape: (batch_size, output_length)

    """

    repeat_train_data = True
    assert testnum % output_length == 0, ('testnum must be divided by '
                                          'output_length')
    dataset_temp = np.zeros([
        len(dataset) - input_length - n_delays - output_length + 1,
        input_length + n_delays + output_length])
    for i in range(
            len(dataset) - input_length - n_delays - output_length + 1):
        dataset_temp[i, :] = dataset[i:i + input_length + n_delays +
                                     output_length]
    if repeat_train_data:
        train_all = dataset_temp[:dataset_temp.shape[0] - testnum, :]
    else:
        train_all = dataset_temp[range(0, dataset_temp.shape[0] - testnum, 25), :]
    test_all = dataset_temp[range(
        dataset_temp.shape[0] - testnum - 1 + output_length, dataset_temp.shape[0],
        output_length), :]
    # np.random.shuffle(train_all)
    x_train = train_all[:, :-output_length - n_delays].reshape(train_all.shape[0], -1)
    y_train = train_all[:, -output_length:].reshape(train_all.shape[0], -1)
    x_test = test_all[:, :-output_length - n_delays]
    y_test = test_all[:, -output_length:]
    if testnum > 0:
        x_test = x_test.reshape(test_all.shape[0], -1)
        y_test = y_test.reshape(test_all.shape[0], -1)
    return x_train, x_test, y_train, y_test


def generate_data(dataset, data_normalizer, input_length, output_length,
                  testnum, window):
    """
    Generate dataset.
    Args:
        dataset: Input dataset.
        data_normalizer: Normalization method.
        input_length: Input length.
        output_length: Output length.
        testnum: Number of sample used for validation.
        window: Sample interval between the last input and the first output.

    Returns: (x_train, y_train, x_test, y_test, normalizer).
        x_train: Shape: (batch, input_length, input_dim)
        y_train: Shape: (batch, output_length, output_dim)
        x_test: Shape: (batch, input_length, input_dim)
        y_test: Shape: (batch, output_length, output_dim)
        normalizer: An instance of Scaler.
    """
    dataset = dataset.reshape(-1, 1).astype(float)
    assert data_normalizer in ['minmax', 'standard', 'robust'], (
        'Normalizer type should be in minmax, standard and robust.')
    if data_normalizer == 'minmax':
        normalizer = MinMaxScaler(feature_range=(0, 1))
    elif data_normalizer == 'standard':
        normalizer = StandardScaler()
    else:
        normalizer = RobustScaler()
    normalizer.fit(dataset[len(dataset) - testnum, 0].reshape(-1, 1))
    dataset_transform = normalizer.transform(dataset).flatten()
    x_train, x_test, y_train, y_test = generate_data_delay(
        dataset_transform, testnum, input_length, output_length, window)
    x_train = np.reshape(x_train, (-1, input_length, 1))
    x_test = np.reshape(x_test, (-1, input_length, 1))
    y_train = np.reshape(y_train, (-1, output_length, 1))
    y_test = np.reshape(y_test, (-1, output_length, 1))
    return x_train, y_train, x_test, y_test, normalizer
