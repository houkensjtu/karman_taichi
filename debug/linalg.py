import taichi as ti

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
                # for j in ti.static([i-ny-1,i-ny,i-1,i,i+1,i+ny,i+ny+1]):
                for j in range(n):
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
                # for j in ti.static([i-ny-1,i-ny,i-1,i,i+1,i+ny,i+ny+1]):
                for j in range(n):
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
