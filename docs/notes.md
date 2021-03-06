# Notes:
# >>> for i in ti.ndrange(5):
# ...     print(i)
# ...
# (0,)
# (1,)
# (2,)
# (3,)
# (4,)

# Problems:

# fill_Au
# 1. Inlet and outlet setting.
# => For inlet, Au[k,k] = 1 and bu[k] = inlet velocity. Will not affect p correction.
# => For outlet, Au[k,k] = Au[k-ny,k-ny], and Au[k,k-ny] = -Au[k,k].

# 2. Upper and lower wall boundary setting.
# => For upper bound, an = 0 and ap += 2*mu*dx/dy

# 3. Investigate the property of Au, is it symmetry? Is it positive definite?
# => Au is not symmetry, but it is full rank and all eig values are positive.

# 4. Depends on dt, which is quicker? CG or Jacobian?

# 5. When inlet boundary is implemented as simple Au[i,i] = 1, what's the results on A's property?
#    Does it affect p correction equation?

# fill_Av
# 1. Accessing out of bound elements is causing unexpected write-in in Av.
# => Make sure all access (k-1, k-ny-1, k+1, k+ny+1) are within boundary.

# Problems almost solved; Confirmed that the solution is correct for 2D plane hagen-posiule flow.
# Next step will be implementing the BiCGSTAB solver to replace Jacobian.
# One remaining issue: In quick Jacobian, solver is accessing elements out of bounds.

# Linear solver
# 1. Normal CG will diverge on u momentum eqn. because it's not symmetric.
# 2. BiCG can converge on u momentum eqn. but the converge is slow and spiky.

# 3. On a 128 x 64 field (eps = 1e-5), BiCG (with M) takes 77sec, Jacobian takes 57 sec to converge.
#    So BiCG (with diag(A) as preconditioner) is no better than Jacobian.
#    BiCG (with no preconditioner) took 19 sec and is much better than Jacobian.

#    For BiCGSTAB with M=diag(A), it took about 7 sec (10x faster than BiCG!)
#    And with M=1.0, it took about 12 sec, showing different behavior than BiCG.
#    Generally, the convergence is much much smoother than BiCG.

# 4. For the v momentum, bicg will fail when all v are zero because the initial rho = 0.

# GUI Display
# 1. Can only write a ti.GUI in main routine, cannot write as a separate function. Don't know why yet.
# 2. u and v are interpolated into u_post and v_post, and then shown in gui.
# 3. Image size needs to adjusted so that show actual aspect ratio lx/ly.

# When lx and ly changed, the pressure drop also needs to be changed to reproduce the hagen flow cond.

# fill_Ap
# 1. Be careful with the nx and ny settings. Because you might lose cells if nx or ny is not divisible
#    by 8 when allocating pointer structures.
# 2. Accessing elements out of bound is causing error in Ap calculations. For example when A[-1,0] != 0.0,
#    it will be added to Ap[0, 0].

# Problems with p correction:
# 1. P correction seems to be proportional to grid size. The smaller the grid, the bigger the correction.
#    => My current explanation: the fix velocity inlet is creating 2 singular point on the corner, and
#       generating unlimited source of p correction.
#    => OpenFOAM's simpleFOAM showed similar trend, the pressure on the corner will grow proportional
#       to the grid size, the smaller the grid, the higher the pressure on corner.
#    => The pressure on the inlet and outlet are fixed by setting the pressure correction to zero.
#       See the fill_Ap() func, where Ap[i,j] = 1 and bp[] = 0.0
#       Accordingly, u velocity on inlet and outlet are set to be zero gradient.
#       Give the intended the pressure difference in init(), and velocity will develop itselt.
# 2. V momentum xv_back() was wrong => Now corrected.
# 3. The under-relaxation of velocity was wrong. Velocity correction does NOT need any relaxation.
#    Instead, under relax the velocity change inside the solving of momentum eqn.
#    This is implemented in the xu_back() and xv_back().
# 4. Implemented a convergence checker in puv_correction: return the max p correction.

# Convection scheme issue 2021.2.5
# 1. Confirmed on OpenFOAM that Upwind can cause strong numerical viscosity, which will eliminate the vortex
#    Must use some other convection scheme. Mid-point can see the vortex, but might be unstable.
# 2. Mesh actually doesn't have to be fine. Even on very coarse mesh, vortex can happen.
# 3. The following version is modified to mid-point scheme instead of Upwind.

# Modified sphere size; hope bigger sphere can generate vortex..? 2021.2.23