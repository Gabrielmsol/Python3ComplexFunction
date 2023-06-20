import numpy as np
import matplotlib.pyplot as plt


def make_grid(i, fi, n_x, n_y):
    x = np.linspace(i, fi, n_x)
    y = np.linspace(i, fi, n_y).reshape(n_y, 1)
    return x+y*1j


def make_circ(r_i, r_f, w_i, w_f, n_r, n_w):
    r = np.linspace(r_i, r_f, n_r)
    w = np.linspace(w_i, w_f, n_w).reshape(n_w, 1)
    return r*np.exp(w*1j)


def get_tensor(matrix):
    tensor = np.zeros((matrix.shape[0], matrix.shape[1], 3), dtype=complex)
    tensor[..., 0] = matrix
    tensor[..., 1] = np.abs(matrix)
    return tensor


def get_graph(tensor, dt):
    x = np.real(tensor[..., 0].flatten())
    y = np.imag(tensor[..., 0].flatten())
    ax.scatter(x, y, s=1, c=np.real(tensor[..., 1].flatten()), cmap='inferno_r')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    plt.draw()
    plt.pause(dt)


def f(matrix):
    return matrix - (matrix**5-1)/(5*matrix**4)


def loop(function, matrix, n, dt):
    m = np.abs(matrix)
    for i in range(n):
        tensor = get_tensor(matrix)
        tensor[..., 1] = m
        ax.cla()
        get_graph(tensor, dt)
        matrix = function(matrix)


Matrix1 = make_circ(0, 1, 0, 2*np.pi, 100, 500)
Matrix2 = make_grid(-1, 1, 223, 223)
plt.ion()
plt.style.use('dark_background')
fig, ax = plt.subplots()
loop(f, Matrix1, 2, 0.5)
plt.ioff()
plt.show()
