# ################################################################
# # Adapted from Naozumi Hiranuma'version                        #
# #                                                              #
# # Keras-compatible implmentation of Integrated Gradients       # 
# # proposed in "Axiomatic attribution for deep neuron networks" #
# # (https://arxiv.org/abs/1703.01365).                          #
# ################################################################

import tensorflow as tf

def interpolate_data(baseline, data, alpha):
    """
    Interpolates data for integrated gradients.

    Args:
    baseline (tensor): Baseline tensor (usually zeros, same shape as the data).
    data (tensor): Input data tensor.
    alpha (tensor): 1D tensor used for interpolation.

    Returns:
    tensor: Interpolated tensor.
    """
    # increase the dimension of the alpha to match the dimension of the data
    for _ in range(len(data.shape)):
        alpha = alpha[:, None]
    # add a dimension before the baseline
    baseline = tf.expand_dims(baseline, axis=0)
    # add a dimension before the data
    data = tf.expand_dims(data, axis=0)
    # delta is the difference between the data and the baseline
    delta = data - baseline
    # linear interpolation
    data = baseline + alpha * delta
    return data


def compute_gradients(model, data, regression, y_true):
    """
    Computes the gradient of the loss function with respect to the input.

    Allows for multi-input format.

    Args:
    model (Keras model): The Keras model.
    data (list of Tensor or Tensor): Input data to the model.
    regression (bool): If False, classification.
    y_true (int or float): Float for regression; int, target class index for classification.

    Returns:
    grad (list of Tensor or Tensor): Gradient calculated.
    """
    with tf.GradientTape() as tape:
        # watch means focusing on this data
        tape.watch(data)
        # calculate the predicted labels
        y_pred = model(data)
        # calculate the loss function
        if regression:
            # regression uses linear
            loss = y_true - y_pred
        else:
            # classification uses softmax
            loss = tf.nn.softmax(y_pred, axis=-1)[:, y_true]
    # calculate the gradient
    grad = tape.gradient(loss, data)
    return grad


def integral_approximation(gradient):
    """
    A function to compute the integral of the gradients.
    :param gradient: Tensor
    :return: integrated gradients: Tensor
    """
    # Riemann Trapezoidal method
    grad = (gradient[:-1] + gradient[1:]) / tf.constant(2.0)
    avg_gradient = tf.math.reduce_mean(grad, axis=0)
    return avg_gradient


@tf.function
def integrated_gradients(model, data, regression, y_true, index, m_steps=50, batch_size=32):
    """
    Computes integrated gradients for explaining model predictions.

    Args:
        model (tf.keras.Model): The Keras model.
        data (list of tf.Tensor or tf.Tensor): Input data to the model.
        regression (bool): If False, classification.
        y_true (int or float): Float for regression; int, target class index for classification.
        index (int): Index of the output tensor to compute gradients with respect to.
        m_steps (int): Number of steps for approximation.
        batch_size (int): Batch size for computation efficiency.

    Returns:
        tf.Tensor: Integrated gradients.
    """
    # create alpha array
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)
    # initialize TensorArray outside loop to collect gradients
    gradient_batches = tf.TensorArray(tf.float32, size=m_steps + 1)
    # iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps
    for alpha in tf.range(0, len(alphas), batch_size):
        # start from alpha
        start = alpha
        # end at start + batch_size, or when reaching the end, end at len(alphas)
        end = tf.minimum(start + batch_size, len(alphas))
        # a batch of alpha
        alpha_batch = alphas[start:end]
        # generate interpolated inputs between baseline and input
        interpolated_data = []
        for data_temp in data:
            data_temp = tf.cast(data_temp, dtype=tf.float32)
            baseline_temp = tf.zeros_like(data_temp, dtype=tf.float32)
            interpolated_temp = interpolate_data(baseline=baseline_temp, data=data_temp, alpha=alpha_batch)
            interpolated_data.append(interpolated_temp)
        # compute gradients between model outputs and interpolated inputs
        gradient_batch = compute_gradients(model=model, data=interpolated_data, regression=regression,
                                           y_true=y_true)[index]
        # write batch indices and gradients to extend TensorArray.
        gradient_batches = gradient_batches.scatter(tf.range(start, end), gradient_batch)
        # stack path gradients together row-wise into single tensor.
    total_gradients = gradient_batches.stack()
    # integral approximation through averaging gradients.
    avg_gradients = integral_approximation(gradient=total_gradients)
    # scale integrated gradients with respect to input.
    data = tf.cast(data[index], dtype=tf.float32)
    baseline = tf.zeros_like(data, dtype=tf.float32)
    integrated_gradient = (data - baseline) * avg_gradients
    return integrated_gradient
