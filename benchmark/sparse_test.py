import taichi as ti

ti.init(arch=ti.cpu)

n = 16
a = ti.field(dtype=ti.f32)
# ti.root.pointer(ti.i, n // 4).dense(ti.i, 4).place(a)
ti.root.dense(ti.i, n).place(a)

@ti.kernel
def init():
    for i in ti.ndrange( (3,7) ):
        print(i)
        a[i] = i

@ti.kernel
def printa():
    for i in a:
        print(f'a[{i}] = {a[i]}')

init()
printa()
