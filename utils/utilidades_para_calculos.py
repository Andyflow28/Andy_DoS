import numpy as np

#! Calculo del jacobiano para el promedio en la superficie de femi

def jacobian(x, Hop_t, En, parametro_de_red):
    cos_kx = np.cos(x * parametro_de_red)
    sin_kx = np.sqrt(1 - cos_kx**2)

    #? Terminos intermedios
    T_1 = En + 2*Hop_t*cos_kx
    z = np.pi - np.arccos((1/2)*(T_1/Hop_t)) # termino de los limites en z

    cos_kz = np.cos(z)
    sin_kz = np.sqrt(1 - cos_kz**2)

    v_x = -2*Hop_t*sin_kx
    v_z = -2*Hop_t*sin_kz

    modulo_de_v_k = np.sqrt(v_x*2 + v_z*2)
    dS_F = np.sqrt(1 + (v_z/v_x)**2)

    jacobiano = dS_F / modulo_de_v_k

    return jacobiano

#! Brecha de energia k-esima

def brecha_k(brecha_0, parametro_de_red, x, Hop_t, En):
    cos_kx = np.cos(x * parametro_de_red)
    T_1 = En + 2*Hop_t*cos_kx
    z = np.pi - np.arccos((1/2)*(T_1/Hop_t)) # termino de los limites en z

    cos_kz = np.cos(z)

    brecha = brecha_0 * (cos_kx - cos_kz) 

    return brecha

#! Terminos a calcular

def termino_positivo(omega, alpha, brecha_k):
    a_k = omega*2 - alpha2 - brecha_k*2
    b_k = 2*omega*alpha
    rho_k = np.sqrt(a_k*2 + b_k*2)

    factor_sin_raiz = omega / np.sqrt(2*rho_k)
    factor_con_raiz = np.sqrt(1 + a_k/rho_k)

    argumento = factor_sin_raiz * factor_con_raiz

    return argumento

def termino_negativo(omega, alpha, brecha_k):
    a_k = omega*2 - alpha2 - brecha_k*2
    b_k = 2*omega*alpha
    rho_k = np.sqrt(a_k*2 + b_k*2)

    factor_sin_raiz = omega / np.sqrt(2*rho_k)
    factor_con_raiz = np.sqrt(1 - a_k/rho_k)

    argumento = factor_sin_raiz * factor_con_raiz

    return argumento