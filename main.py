from functions import *

N = 1200 + 10 * int(INDEX_NUMBER[-2]) + int(INDEX_NUMBER[-1])
a1 = 5 + int(INDEX_NUMBER[3])
a2 = a3 = -1

A, b = generate_matrix(N, a1, a2, a3)

print(f"a1 = {a1}:\n")

jacobi_x, jacobi_norm, jacobi_time, jacobi_iterations = jacobi(A, b)
gauss_seidl_x, gauss_seidl_norm, gauss_seidl_time, gauss_siedl_iterations = gauss_seidl(A, b)
plot_norm(jacobi_norm, gauss_seidl_norm)

direct_x, direct_norm, direct_time = direct(A, b)

print(f"Jacobi norm: {jacobi_norm[-1]}")
print(f"Jacobi solution: {jacobi_x}")
print(f"Jacobi iterations: {jacobi_iterations}")
print(f"Jacobi time: {jacobi_time} seconds\n")

print(f"Gauss-Seidl norm: {gauss_seidl_norm[-1]}")
print(f"Gauss-Seidl solution: {gauss_seidl_x}")
print(f"Gauss-Seidl iterations: {gauss_siedl_iterations}")
print(f"Gauss-Seidl time: {gauss_seidl_time} seconds\n")

print(f"Direct norm: {direct_norm}")
print(f"Direct solution: {direct_x}")
print(f"Direct time: {direct_time} seconds\n")

a1 = 3
A, b = generate_matrix(N, a1, a2, a3)

print("a1 = 3:\n")

jacobi_x, jacobi_norm, jacobi_time, jacobi_iterations = jacobi(A, b)
gauss_seidl_x, gauss_seidl_norm, gauss_seidl_time, gauss_siedl_iterations = gauss_seidl(A, b)
plot_norm(jacobi_norm, gauss_seidl_norm)

direct_x, direct_norm, direct_time = direct(A, b)

print(f"Jacobi norm: {jacobi_norm[-1]}")
print(f"Jacobi solution: {jacobi_x}")
print(f"Jacobi iterations: {jacobi_iterations}")
print(f"Jacobi time: {jacobi_time} seconds\n")

print(f"Gauss-Seidl norm: {gauss_seidl_norm[-1]}")
print(f"Gauss-Seidl solution: {gauss_seidl_x}")
print(f"Gauss-Seidl iterations: {gauss_siedl_iterations}")
print(f"Gauss-Seidl time: {gauss_seidl_time} seconds\n")

print(f"Direct norm: {direct_norm}")
print(f"Direct solution: {direct_x}")
print(f"Direct time: {direct_time} seconds\n")

a1 = 5 + int(INDEX_NUMBER[3])
sizes = [100, 250, 500, 750, 1000, 1500, 2000, 3000]
jacobi_times = []
gauss_seidl_times = []
direct_times = []

for N in sizes:
    A, b = generate_matrix(N, a1, a2, a3)
    _, _, jacobi_time, _ = jacobi(A, b)
    _, _, gauss_seidl_time, _ = gauss_seidl(A, b)
    _, _, direct_time = direct(A, b)
    jacobi_times.append(jacobi_time)
    gauss_seidl_times.append(gauss_seidl_time)
    direct_times.append(direct_time)
    
plot_times(jacobi_times, gauss_seidl_times, direct_times, sizes)
plot_times(jacobi_times, gauss_seidl_times, direct_times, sizes, ylog=False)