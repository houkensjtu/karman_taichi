import taichi as ti


@ti.kernel # Ax = A dot x
def compute_dot(A:ti.template(),
                x:ti.template(),
                Ax:ti.template()):
    n = A.shape[0]
    for i in range(n):
        Ax[i] = 0.0
        # for j in range(n):
        for j in ti.static([i-3,i-1,i,i+1,i+3]):            
            Ax[i] += A[i, j] * x[j]

@ti.kernel # Return v1T * v2
def reduce(v1:ti.template(), v2:ti.template())->ti.f64:
    n = v1.shape[0]
    sum = 0.0
    for i in range(n):
        sum += v1[i] * v2[i]
    return sum

@ti.kernel # Compute r = b - Ax, r_tld=r
def compute_res(A:ti.template(),
                b:ti.template(),
                x:ti.template(),
                Ax:ti.template(),
                r:ti.template(),
                r_tld:ti.template()):
    n = A.shape[0]
    for i in range(n):
        Ax[i] = 0.0
        for j in range(n):
            Ax[i] += A[i,j] * x[j]
        r[i] = b[i] - Ax[i]
        r_tld[i] = r[i]
        
def newbicg(A,b,x,M,Ax,r,r_tld,p,p_hat,Ap,s,s_hat,t,nx,ny,n,eps,output):
    # Global -> newbicg(A) -> @ti.kernel printA(A)
    compute_res(A,b,x,Ax,r,r_tld)
    residual_init = 0.0
    residual_init = reduce(r,r)
    omega = 1.0
    alpha = 1.0
    beta = 1.0
    rho_1 = 1.0
    for steps in range(100*n):
        rho = 0.0
        rho = reduce(r,r_tld)
        if rho == 0.0:
            if output:
                print("        >> Bicgstab failed...")
            break

        if steps == 0:
            for i in range(n):
                p[i] = r[i]
        else:
            beta = (rho / rho_1) * (alpha/omega)
            for i in range(n):
                p[i]  = r[i] + beta*(p[i] - omega*Ap[i])
        for i in range(n):
            p_hat[i] = 1/M[i,i] * p[i]
            
        # dot(A,p)
        # Ap => v
        compute_dot(A,p_hat,Ap)
        alpha_lower = 0.0
        for i in range(n):
            alpha_lower += r_tld[i] * Ap[i]
        alpha = rho / alpha_lower
        
        for i in range(n):
            s[i] = r[i] - alpha * Ap[i]
        # Early convergnece check...
        for i in range(n):
            s_hat[i] = 1/M[i, i]*s[i]
            
        compute_dot(A,s_hat,t)
        omega_upper = reduce(t, s)
        omega_lower = reduce(t, t)
        omega = omega_upper / omega_lower
        
        for i in range(n):
            x[i] += alpha* p_hat[i] + omega*s_hat[i]
        for i in range(n):
            r[i] = s[i] - omega*t[i]

        residual = reduce(r,r)
        if output:            
            print("        >> Iteration ", steps, ", initial residual = ", residual_init, ", current residual = ", residual)
            
        if ti.sqrt(residual / residual_init) < eps:
            if output:
                print("        >> The solution has converged...")
            break
        
        if omega==0.0:
            if output:
                print("        >> Omega = 0.0 ...")
            break
        
        rho_1 = rho

    
@ti.kernel
def bicgstab(A:ti.template(),
             b:ti.template(),
             x:ti.template(),
             M:ti.template(),
             Ax:ti.template(),
             r:ti.template(),
             r_tld:ti.template(),
             p:ti.template(),
             p_hat:ti.template(),
             Ap:ti.template(),
             s:ti.template(),
             s_hat:ti.template(),
             t:ti.template(),
             nx:ti.i32,
             ny:ti.i32,
             n:ti.i32,
             eps: ti.f64,
             output:ti.i32):
    # dot(A,x)
    for i in range(n):
        Ax[i] = 0.0
        # Only traverse certain elements. Need to use ti.static() to convert python list.        
        # for j in ti.static([i-ny-1,i-ny,i-1,i,i+1,i+ny,i+ny+1]):
        # 2022.3.1 For test purpose, traverse all elements
        for j in range(n):
            Ax[i] += A[i, j] * x[j]

    # r = b - dot(A,x)
    for i in range(n):
        r[i] = b[i] - Ax[i]
        r_tld[i] = r[i]

    residual_init = 0.0
    for i in range(n):
        residual_init += r[i] * r[i]

    omega = 1.0
    alpha = 1.0
    beta = 1.0
    rho_1 = 1.0
    for _ in range(1):
        for steps in range(100*n):
            rho = 0.0
            for i in range(n):
                rho += r[i] * r_tld[i]
            if rho == 0.0:
                if output:
                    print("        >> Bicgstab failed...")
                pass
        
            if steps == 0:
                for i in range(n):
                    p[i] = r[i]
            else:
                beta = (rho / rho_1) * (alpha/omega)
                for i in range(n):
                    p[i]  = r[i] + beta*(p[i] - omega*Ap[i])
            for i in range(n):
                p_hat[i] = 1/M[i,i] * p[i]
            
            # dot(A,p)
            # Ap => v        
            for i in range(n):
                Ap[i] = 0.0
                # Only traverse certain elements. Need to use ti.static() to convert python list.
                for j in ti.static([i-3,i-1,i,i+1,i+3]):
                #for j in ti.static([i-ny-1,i-ny,i-1,i,i+1,i+ny,i+ny+1]):                    
                #for j in range(n):
                    Ap[i] += A[i, j] * p_hat[j]

            alpha_lower = 0.0
            for i in range(n):
                alpha_lower += r_tld[i] * Ap[i]
            alpha = rho / alpha_lower

            for i in range(n):
                s[i] = r[i] - alpha * Ap[i]

            # Early convergnece check...
            for i in range(n):
                s_hat[i] = 1/M[i, i]*s[i]

            for i in range(n):
                t[i] = 0.0
                # Only traverse certain elements. Need to use ti.static() to convert python list.            
                #for j in ti.static([i-ny-1,i-ny,i-1,i,i+1,i+ny,i+ny+1]):
                for j in ti.static([i-3,i-1,i,i+1,i+3]):                
                #for j in range(n):
                    t[i] += A[i, j] * s_hat[j]

            omega_upper = 0.0
            omega_lower = 0.0
            for i in range(n):
                omega_upper += t[i] * s[i]
                omega_lower += t[i] * t[i]
            omega = omega_upper / omega_lower

            for i in range(n):
                x[i] += alpha* p_hat[i] + omega*s_hat[i]

            for i in range(n):
                r[i] = s[i] - omega*t[i]

            residual = 0.0
            for i in range(n):
                residual += r[i] * r[i]
            if output:            
                print("        >> Iteration ", steps, ", initial residual = ", residual_init, ", current residual = ", residual)
            
            if ti.sqrt(residual / residual_init) < eps:
                if output:
                    print("        >> The solution has converged...")
                break
        
            if omega==0.0:
                if output:
                    print("        >> Omega = 0.0 ...")
                pass
        
            rho_1 = rho
