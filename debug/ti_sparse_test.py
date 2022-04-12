import taichi as ti

real = ti.f32
#ti.init(default_fp=real, arch=ti.cpu, debug=True)
ti.init(default_fp=real, arch=ti.cpu)

# -- Grid parameters --
N = 2
N_tot = 2 * N
N_ext = N // 2

x = ti.field(dtype=real)
y = ti.field(dtype=real)
# ti.root.dense(ti.ij, (N_tot, N_tot)).place(x, y)
# ti.root.pointer(ti.ij, (N_tot//4, N_tot//4)).dense(ti.ij, (4,4)).place(x, y)
ti.root.pointer(ti.ij, (N_tot//4, N_tot//4)).pointer(ti.ij, (4,4)).place(x, y)


@ti.kernel
def init():
    for i, j in ti.ndrange((N_ext, N_tot-N_ext), (N_ext, N_tot-N_ext)):
        print('Now assigning x and y at', i, j)
        x[i, j] = 0.0
        y[i, j] = 0.0


@ti.kernel
def compute():
    for i, j in x:
        x[i, j] = y[i-1, j] + y[i, j-1] + y[i+1, j] + y[i, j+1]
        print('Now accessing x at', i, j, 'and x =', x[i,j])
        if x[i, j] != 0.0:
            print('Warning! Non-zero element discovered.')
        

init()
compute()
print('Try print a y outside the matrix at 20,20:', y[20,20])   # return 0.0
print('Try print a y outside the matrix at -1,-1:', y[-1,-1])   # return 0.0


