import taichi as ti

ti.init(arch=ti.cpu)

t = ti.field(dtype=ti.f32, shape=129)

@ti.kernel
def serial_fill():
    for _ in range(1):
        for i in range(129):
            t[i] = i
            print(t[i])

@ti.kernel
def fill():
    for i in range(129):
        t[i] = i
        print(t[i])
        
serial_fill()            
