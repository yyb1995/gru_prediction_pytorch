import matplotlib.pyplot as plt
import torch
import time
import numpy as np

from torch.utils.data import random_split, Dataset, Subset, DataLoader


class CustomDataset(Dataset):
    """
    Dataset class for whole dataset. x: input data. y: output data
    """
    def __init__(self, x, y):
        # np.float32 is equal to torch.float
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def train_epoch(model, train_dataloader, optimizer, loss_function, device):
    """
    Implement a single epoch.

    Args:
        model: Model to use.
        train_dataloader: Training dataloader.
        optimizer: Training Optimizer.
        loss_function: Loss function.
        device: 'cpu' or 'cuda'.

    Returns: Loss of a sample after a training epoch.

    """
    model.train()
    average_loss = 0
    sample_sum = 0

    for batch in train_dataloader:
        # Get input and target sequence
        x_train, y_train = batch
        sample_sum += x_train.shape[0]

        x_train, y_train = map(lambda x: x.to(device), [x_train, y_train])

        # Forward step
        optimizer.zero_grad()
        output = model(x_train)
        loss = loss_function(output, y_train)

        # Loss backward
        loss.backward()

        # Update learning rate. Default learning rate is same
        optimizer.step()

        # Calculate loss in a training epoch
        average_loss += loss.item()

    average_loss = average_loss / sample_sum
    return average_loss


def eval_epoch(model, eval_dataloader, loss_function, device):
    """
    Implement an evaluation epoch.

    Args:
        model: Model to use.
        eval_dataloader: Evaluation dataloader.
        loss_function: Loss function.
        device: 'cpu' or 'cuda'.

    Returns: y_predict_list: Prediction list. Shape: (batch, output_size, output_dim)
             y_real_list: Real list. Shape: (batch, output_size, output_dim)
             average_loss: Validation loss over a sample.

    """
    model.eval()
    sample_sum = 0
    average_loss = 0
    y_predict_list = []
    y_real_list = []

    # Validation doesn't update gradients.
    with torch.no_grad():
        for batch in eval_dataloader:
            # Get input and target sequence.
            x_test, y_test = batch
            sample_sum += x_test.shape[0]

            x_test, y_test = map(lambda x: x.to(device), [x_test, y_test])

            output = model(x_test)
            y_predict_list.append(output)
            y_real_list.append(y_test)
            # Save prediction result

            # Calculate loss if needed
            loss = loss_function(output.flatten(), y_test.flatten())
            average_loss += loss.item()

        # Get final prediction and real tensor
        y_real_list = torch.cat(y_real_list, dim=0)
        y_predict_list = torch.cat(y_predict_list, dim=0)

        average_loss = average_loss / sample_sum

    return y_predict_list, y_real_list, average_loss


def train(model, x, y, epoch, batch_size, optimizer, loss_function, device, validation_split=0,
          shuffle=False):
    """
    Implement train process.

    Args:
        model: Model used for training.
        x: Input for training. Shape: (batch, input_length, input_dim).
        y: Output for training. Shape: (batch, output_length, output_dim).
        epoch: Training epoch.
        batch_size: Training batch size.
        optimizer: Training optimizer.
        loss_function: Training loss function.
        device: 'cpu' or 'cuda'.
        validation_split: Validation set Percentage in training set.
        shuffle: If True, shuffle the train and validation data.

    Returns: Train result

    """

    # Initial setting
    dataset = CustomDataset(x, y)
    val_num = int(validation_split * len(dataset))
    train_num = len(dataset) - val_num
    batch_train_loss = []
    batch_val_loss = []

    # Select model running device
    if device == 'cuda':
        assert torch.cuda.is_available() is True, "cuda isn't available"
        print('Using gpu backend.\n'
              'gpu type: {}'.format(torch.cuda.get_device_name(0)))
        model.to('cuda')
    else:
        print('Using cpu backend.')
        model.to('cpu')

    # Check lr_scheduler and validation_split legality.
    if validation_split < 0 or validation_split > 1:
        raise ValueError('validation_split must between 0 and 1.')

    # Training process
    start_train = time.time()

    for epoch_i in range(epoch):
        start_epoch = time.time()

        # Generate training & validation set
        if shuffle:
            train_set, val_set = random_split(dataset, [train_num, val_num])
        else:
            train_set = Subset(dataset, range(train_num))
            val_set = Subset(dataset, range(train_num, len(dataset)))

        # Generate training Dataloader
        train_dataloader = DataLoader(train_set, batch_size, shuffle=False)

        # Train & valid data.
        # Train_loss is calculated within a batch.
        train_loss = train_epoch(model, train_dataloader, optimizer, loss_function, device)
        train_loss = train_loss * batch_size
        batch_train_loss.append(train_loss)

        # Generate loss display mode.
        if train_loss >= 1e-4:
            disp_train = '{2:.4f}'
        else:
            disp_train = '{2:.3e}'

        if validation_split and 0. < validation_split < 1.:
            valid_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
            val_loss = eval_epoch(model, valid_dataloader, loss_function, device)[2]
            val_loss = val_loss * batch_size
            batch_val_loss.append(val_loss)

            if val_loss >= 1e-4:
                disp_val = '{3:.4f}'
            else:
                disp_val = '{3:.3e}'

            print(
                ('Epoch {0}/{1}  -train_loss: ' + disp_train + '  -val_loss: ' +
                 disp_val + '  -time: {4:.1f}s').format(
                     epoch_i + 1, epoch, train_loss, val_loss, time.time() -
                    start_epoch))

        else:
            print(('Epoch {0}/{1} -train_loss: ' + disp_train + ' -time: {3:.1f}s').format(
                  epoch_i + 1, epoch, train_loss, time.time() - start_epoch))

    # Train process end
    total_train_time = int(time.time() - start_train)
    print('Training process finish.\nTotal train time: {}h{}m{}s'.format(
          total_train_time // 3600, total_train_time % 3600 // 60,
          total_train_time % 3600 % 60))

    return batch_train_loss, batch_val_loss


def predict(model, x, y, loss_function, device,
            normalizer, batch_size, shuffle=False):
    """
    Implement predict process.

    Args:
        model: Model used for prediction.
        x: x_test. Shape: (batch, length, dim)
        y: y_test. Shape: (batch, length, dim)
        batch_size: Testing batch size.
        loss_function: Loss function for calculating loss.
        device: 'cpu' or 'cuda'.
        normalizer: Data normalizer.
        batch_size: Prediction process batch size.
        shuffle: True if shuffle the test dataset.

    Returns: De-normalized prediction and true value

    """

    # Initial setting
    predict_set = CustomDataset(x, y)

    # Prediction process
    # Generate test dataloader

    predict_dataloader = DataLoader(predict_set, batch_size=batch_size, shuffle=shuffle)

    # Make prediction result and transferred to numpy array
    y_predict, y_real, loss = eval_epoch(model, predict_dataloader, loss_function, device)

    y_predict = y_predict.to('cpu').numpy()
    y_real = y_real.to('cpu').numpy()

    # Rescale prediction result
    y_predict = (normalizer.inverse_transform(y_predict.reshape(-1, 1)).flatten())
    y_real = normalizer.inverse_transform(y_real.reshape(-1, 1)).flatten()

    return y_predict, y_real, loss
