import numpy as np
import matplotlib.pyplot as plt
import time

INDEX_NUMBER = "198034"

def generate_matrix(N, a1, a2, a3):
    A = np.zeros(shape=(N, N), dtype=float)
    b = np.sin(np.arange(start=1, stop=N+1) * (int(INDEX_NUMBER[2]) + 1))

    np.fill_diagonal(A, a1)
    np.fill_diagonal(A[1:], a2)
    np.fill_diagonal(A[:, 1:], a2)
    np.fill_diagonal(A[2:], a3)
    np.fill_diagonal(A[:, 2:], a3)

    return A, b

def lu(A):
    n = A.shape[0]
    
    L = np.eye(n)
    U = np.zeros(shape=(n, n))
    
    for i in range(n):
        U[i, i:] = A[i, i:] - np.matmul(L[i, :i], U[:i, i:])
        L[(i+1):, i] = (A[(i+1):, i] - np.matmul(L[(i+1):, :i], U[:i, i])) / U[i, i]
    
    return L, U

def direct(A, b):
    start = time.time()

    L, U = lu(A)
    
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(U, y)

    end = time.time()
    
    rnorm = np.linalg.norm(np.matmul(A, x) - b)
    return x, rnorm, end - start
        
def jacobi(A, b):
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    D = np.diag(np.diag(A))
    x = np.ones(np.size(b))
    
    w = np.linalg.solve(D, b)
    M = np.linalg.solve(-D, L + U)

    rnorm = [np.linalg.norm(np.matmul(A, x) - b)]
    iterations = 0
    
    start = time.time()
    while iterations < 1000 and rnorm[-1] >= 10 ** -9:
        iterations += 1
        x = np.matmul(M, x) + w
        rnorm.append(np.linalg.norm(np.matmul(A, x) - b))
    end = time.time()    
    
    return x, rnorm, end - start, iterations

def gauss_seidl(A, b):
    U = np.triu(A, 1)
    T = A - U # D + L
    x = np.ones(np.size(b))
    
    M = np.linalg.solve(-T, U)
    w = np.linalg.solve(T, b)

    rnorm = [np.linalg.norm(np.matmul(A, x) - b)]
    iterations = 0
    
    start = time.time()
    while iterations < 1000 and rnorm[-1] >= 10 ** -9:
        iterations += 1
        x = np.matmul(M, x) + w
        rnorm.append(np.linalg.norm(np.matmul(A, x) - b))
    end = time.time()
        
    return x, rnorm, end - start, iterations

def plot_norm(jacobi_norm, gauss_seidl_norm):
    plt.figure(figsize=(10, 6))
    plt.plot(jacobi_norm, label="Jacobi")
    plt.plot(gauss_seidl_norm, label="Gauss-Seidl")
    
    plt.grid(visible=True)
    plt.yscale("log")
    plt.ylabel("Norma (log)")
    plt.xlabel("Iteracja")
    plt.title("Norma residuum w kolejnych iteracjach dla metod iteracyjnych")
    plt.legend()
    
    plt.show()
    
def plot_times(jacobi_times, gauss_seidl_times, direct_times, sizes, ylog=True):
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, jacobi_times, label="Jacobi", marker='x')
    plt.plot(sizes, gauss_seidl_times, label="Gauss-Seidl", marker='o')
    plt.plot(sizes, direct_times, label="LU", marker='^')
    
    plt.grid(visible=True)
    plt.yscale("log") if ylog else {}
    plt.ylabel("Czas [s] (log)" if ylog else "Czas [s]")
    plt.xlabel("Rozmiar macierzy")
    plt.title("Zależność czasu wykonania danej metody od rozmiaru macierzy")
    plt.legend()
    
    plt.show(block=False if ylog else True)