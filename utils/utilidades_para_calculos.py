import numpy as np

#! Calculo del jacobiano para el promedio en la superficie de femi

def jacobian(x, Hop_t, En):
    cos_kx = np.cos(x)
    sin_kx = np.sqrt(1 - cos_kx**2)

    #? Terminos intermedios
    T_1 = En + 2*Hop_t*cos_kx
    z = np.pi - np.arccos((1/2)*(T_1/Hop_t)) # termino de los limites en z

    cos_kz = np.cos(z)
    sin_kz = np.sqrt(1 - cos_kz**2)

    v_x = -2*Hop_t*sin_kx
    v_z = -2*Hop_t*sin_kz

    modulo_de_v_k = np.sqrt(v_x**2 + v_z**2)
    dS_F = np.sqrt(1 + (v_z/v_x)**2)

    jacobiano = dS_F / modulo_de_v_k

    return jacobiano