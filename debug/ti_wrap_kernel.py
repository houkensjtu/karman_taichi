import taichi as ti
from karman_taichi import bicgstab

ti.init(default_fp=ti.f32, arch = ti.gpu)

t = ti.field(float, shape=127)

# Wrap inner loop as a kernel
@ti.kernel
def internal(t:ti.template()):
    for i in t:
        t[i] = i

# Simulate the bicgstab stepping in a outer loop
def loop():
    for s in range(100):
        print(s)
        internal(t)
        if s > 50:
            break

# Call the loop        
loop()        
    


    
