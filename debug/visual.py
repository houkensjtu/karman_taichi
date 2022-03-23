import taichi as ti

ti.init(arch=ti.cpu, default_fp = ti.f32)

data = ti.field(dtype=ti.f32,shape=(500,500))

# Visualize mat as a np array
def visual_matrix(mat):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm    
    im = plt.imshow(mat, cmap = 'bone')
    plt.colorbar(im)
    plt.show()

@ti.kernel
def fill():
    for i, j in data:
        data[i,j] = ti.sin(ti.cast(i,ti.f32)/100) * ti.cos(ti.cast(j, ti.f32)/100)


if __name__ == '__main__':        
    fill()
    npdata = data.to_numpy()
    visual_matrix(npdata)
