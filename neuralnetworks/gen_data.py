import numpy as np
np.random.seed(1)

def gen_x(num_steps, num_vars, as_3d=False):
    """
    Returns ndarray of shape (num_steps, num_vars) or (1, num_steps, num_vars) if 3d
    E.g. to get multiple samples
    x1 = get_x(3, 2, True)
    x2 = get_x(3, 2, True)
    x3 = np.concatenate((x1, x2), axis=0)
    """
    x = np.random.normal(size=(num_steps, num_vars))
    return np.reshape(x, (1, x.shape[0], x.shape[1])) if as_3d else x


def gen_y(num_steps, as_3d=False):
    """
    Returns ndarray of ones just so we know the model tries to predict sales of 1 unit per day
    """
    return np.ones(shape=(1, num_steps, 1)) if as_3d else np.ones(shape=(num_steps, 1))


def gen_samples(size, num_steps, num_vars):
    x1 = gen_x(num_steps, num_vars, True)
    y1 = gen_y(num_steps, True)
    for i in range(1, size):
        x2 = gen_x(num_steps, num_vars, True)
        x1 = np.concatenate((x1, x2), axis=0)
        y2 = gen_y(num_steps, True)
        y1 = np.concatenate((y1, y2), axis=0)
    return x1, y1


