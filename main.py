import numpy as np


def f(x):
    return 3*x[0]**2 + x[1]**2 - x[0]*x[1] + x[0]


def gradient(f, x):
    grad = np.zeros(2)
    for i in range(2):
        eps = 1e-8
        delta = np.zeros(2)
        delta[i] = eps
        f_plus = f(x + delta)
        f_minus = f(x - delta)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def hessian(f, x):
    hess = np.zeros((2, 2))
    for i in range(2):
        eps1 = 1e-8
        delta1 = np.zeros(2)
        delta1[i] = eps1
        for j in range(i, 2):
            eps2 = 1e-8
            delta2 = np.zeros(2)
            delta2[j] = eps2
            f_pp = f(x + delta1 + delta2)
            f_pm = f(x + delta1 - delta2)
            f_mp = f(x - delta1 + delta2)
            f_mm = f(x - delta1 - delta2)
            hess[i][j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps1 * eps2)
            hess[j][i] = hess[i][j]
    return hess


def newton_raphson(eps1, eps2, x0, f, gradient, hessian, M=10):
    flag = False
    x_k_next = 0
    k = 0
    x_k = x0
    while not flag:
        grad = gradient(f, x_k)
        norma = np.linalg.norm(grad)
        if norma <= eps1:
            x_otv = x_k
            print('norma <= eps1')
            return x_otv,  k
        elif k < M:
            hess = hessian(f, x_k)
            eigvals = np.linalg.eigvals(hess)
            if np.all(eigvals > 0):
                d_k = -np.linalg.solve(hess, grad)
            else:
                d_k = -grad
            t_k = 1
            while f(x_k + t_k*d_k) > f(x_k) + eps2*t_k*np.dot(grad, d_k):
                t_k /= 2
            x_k_next = x_k + t_k*d_k
            if np.linalg.norm(x_k_next - x_k) < eps1 or np.linalg.norm(f(x_k_next) - f(x_k)) < eps1:
                flag = True
            else:
                x_k = x_k_next
                k += 1
        else:
            x_otv = x_k
            print('k >= M')
            return x_otv, k
    print('norm(x_k_next - x_k) < eps1 or norm(f(x_k_next) - f(x_k)) < eps1')
    return x_k_next, k


eps1 = 0.1
eps2 = 0.15
x0 = np.array([-1.5, -1.5])
result, k = newton_raphson(eps1, eps2, x0, f, gradient, hessian)
print(result, k)
