# karman_tachi

**karman_taichi** is an incompressible fluid solver written in the [Taichi programming language](https://taichi.graphics/). It solves a 2-D rectagular fluid field with a cylindrical obstacle. A [Karman vortext street](https://en.wikipedia.org/wiki/K%C3%A1rm%C3%A1n_vortex_street) can be observed when the [Reynolds number](https://en.wikipedia.org/wiki/Reynolds_number) is sufficiently large.

In the following example, the animation is showing the y-direction velocity, x-direction velocity and the pressure field, respectively from top to bottom. The Reynolds number in this example is around 150. (The characteristic length used in Reynolds number should be **the diameter of the cylinder**, not the width of the channel.)

![calculated result](./gif/karman-vortex.gif)

Incompressible fluid solver
--------------
I used the [finite volume method (FVM)](https://en.wikipedia.org/wiki/Finite_volume_method) to represent the Navier-Stokes equation. The [Semi-Implicit Method for Pressure Linked Equations (SIMPLE)](https://en.wikipedia.org/wiki/SIMPLE_algorithm) was used to solve the velocity and pressure field iteratively.

Navigating the code
--------------
The main loop of the solver is simply a implementation of [what's described](https://en.wikipedia.org/wiki/SIMPLE_algorithm) as the SIMPLE method. 

*Under construction...* 

About Karman vortex's simulation
--------------
- Grid of high resolution is **NOT** required to be able to see the vortex. I found that even on very coarse mesh, Karman vortex can occur and develop. In the example above, I have 64 cells in the y direction and 320 cells in the x direction (Even this is probably more than enough).

- You do **NOT** need very fine, adaptive boundary layer to generate the vortex. I have no adaptive mesh layer in my simulation. In fact, the cylinder itself is approximated by square cells.

- Be careful with your discretization scheme for the convection term in the momentum equation. Some numerical schemes, for example, the first order [upwind scheme](https://en.wikipedia.org/wiki/Upwind_differencing_scheme_for_convection), have strong numerical viscosity and can suppress the vortex street. Simple mid-point scheme was used in this solver.

Current issues 
--------------
- GPU didn't work on my old Thinkpad W530 laptop. Feel free to try my code if you have CUDA capable PC!

- BiCGSTAB was used in my solver to solve both the momentum equation and the pressure correction equation. While BiCGSTAB is stable and fast on itself, lack of pre-conditioner and multigrid led to very slow computation speed (Every frame will take 5-10 minutes to compute!)

- Abstraction level of the code is very limited. Especially the handling of boundary condition is ugly and clumsy. 

- Much more ...