import taichi as ti
from cgsolver import CGPoissonSolver
from bicgsolver import BICGPoissonSolver
from scipy.sparse.linalg import bicgstab
import inspect

ti.init(arch=ti.x64, default_fp = ti.f64) # default_fp=ti.f64 to ensure precision
psize = 64
cgsolver = BICGPoissonSolver(psize, 1e-6, 0.0, quiet=False)

print('Solving in Taichi...')
cgsolver.solve()

print('Saving results...')
a = cgsolver.build_A() # a => np.ndarray(psize*psize, psize*psize)
b = cgsolver.build_b() # b => np.ndarray(psize*psize, 1)
cgsolver.save_history()# Save convergence history to convergence.txt

# Prepare callback function for Scipy to log residual history
ssl_conv_history = []
def report(xk):
    frame = inspect.currentframe().f_back
    ssl_conv_history.append((frame.f_locals['resid']))

print('Solving in Scipy...')
x = bicgstab(a, b, tol=1e-6, callback=report)     # Solve in Scipy
with open('scipyconv.txt', 'w') as f:
    for line in ssl_conv_history:
        f.write(str(line)+'\n')

# Plot convergence history
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
taichi_conv_data = np.log(np.loadtxt('convergence.txt')[1:])
scipy_conv_data = np.log(np.loadtxt('scipyconv.txt'))

plt.plot(taichi_conv_data, '.-', label='Bicgstab in Taichi')
plt.plot(scipy_conv_data, '.-', label='Bicgstab in Scipy')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('log(Residual)')
plt.grid(True)
plt.show()


# Procedure in Octave
'''
% Octave/Matlab script to produce a comparison plot:
% Assuming A.csv, b.csv, convergence.txt exist

clear all;
load('A.csv');
load('b.csv');
load('convergence.txt');
[x, flag, relres, iter, resvec] = bicgstab(A, b, 1e-06, 100000);

r = resvec(1:2:end); # Skip every other element; see Octave's official doc on Bicgstab
logc = log(convergence);
logr = log(r);

plot(logc, 'displayname', 'Bicgstab in Taichi', 'marker', '.');
hold on;
plot(logr, 'displayname', 'Bicgstab in Octave', 'marker', '.');
legend();
xlabel('Iteration');
ylabel('log(Residual)');
grid on;
'''
