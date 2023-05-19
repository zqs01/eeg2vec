import torch

def pearson_torch(y_true, y_pred, axis=1):
    """Pearson correlation function implemented in torch.

    Parameters
    ----------
    y_true: tf.Tensor
        Ground truth labels. Shape is (batch_size, time_steps, n_features)
    y_pred: tf.Tensor
        Predicted labels. Shape is (batch_size, time_steps, n_features)
    axis: int
        Axis along which to compute the pearson correlation. Default is 1.

    Returns
    -------
    tf.Tensor
        Pearson correlation.
        Shape is (batch_size, 1, n_features) if axis is 1.
    """
    # Compute the mean of the true and predicted values
    y_true_mean = torch.mean(y_true, dim=axis, keepdims=True)
    y_pred_mean = torch.mean(y_pred, dim=axis, keepdims=True)
    # Compute the numerator and denominator of the pearson correlation
    numerator = torch.sum(
        (y_true - y_true_mean) * (y_pred - y_pred_mean),
        dim=axis,
        keepdims=True,
    )
    # print(numerator.size())
    std_true = torch.sum(torch.square(y_true - y_true_mean), dim=axis, keepdims=True)
    std_pred = torch.sum(torch.square(y_pred - y_pred_mean), dim=axis, keepdims=True)
    denominator = torch.sqrt(std_true * std_pred)
    # print(denominator.size())
    # # Compute the pearson correlation
    # return tf.math.divide_no_nan(numerator, denominator)
    out = torch.div(numerator, denominator)
    # print(out)
    out = torch.mean(out, dim=0)
    # print(out)
    return out

# if __name__ == "__main__":
#     pred = torch.randn(700,1, 640)
#     label = torch.randn(700,1, 640)
#     x = pearson_torch(label, pred)