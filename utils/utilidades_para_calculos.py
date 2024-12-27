import numpy as np

# ⚠️ Evitar divisiones por cero
EPSILON = 1e-12

# =====================================
# 📌 Cálculo del Jacobiano para el promedio en la superficie de Fermi
# =====================================

def jacobian(x, Hop_t, En, parametro_de_red):
    """
    Calcula el jacobiano para el promedio en la superficie de Fermi.

    Args:
        x (float): Parámetro de integración.
        Hop_t (float): Parámetro de salto.
        En (float): Energía.
        parametro_de_red (float): Parámetro de red.

    Returns:
        float: Valor del jacobiano.
    """
    cos_kx = np.cos(x * parametro_de_red)
    sin_kx = np.sqrt(1 - cos_kx**2)

    # Términos intermedios
    T_1 = En + 2 * Hop_t * cos_kx
    try:
        z = np.pi - np.arccos((1 / 2) * (T_1 / Hop_t))
    except ValueError:
        z = 0.0  # Asegurar valor válido en caso de errores numéricos

    cos_kz = np.cos(z)
    sin_kz = np.sqrt(max(0, 1 - cos_kz**2))  # Asegurar que no haya raíz negativa

    v_x = -2 * Hop_t * sin_kx
    v_z = -2 * Hop_t * sin_kz

    modulo_de_v_k = np.sqrt(v_x**2 + v_z**2 + EPSILON)  # Evitar división por cero
    dS_F = np.sqrt(1 + (v_z / (v_x + EPSILON))**2)

    jacobiano = dS_F / modulo_de_v_k

    return jacobiano

# =====================================
# 📌 Brecha de energía k-ésima
# =====================================

def brecha_k(brecha_0, parametro_de_red, x, Hop_t, En):
    """
    Calcula la brecha de energía para un valor dado de x.

    Args:
        brecha_0 (float): Brecha inicial.
        parametro_de_red (float): Parámetro de red.
        x (float): Parámetro de integración.
        Hop_t (float): Parámetro de salto.
        En (float): Energía.

    Returns:
        float: Valor de la brecha.
    """
    cos_kx = np.cos(x * parametro_de_red)
    T_1 = En + 2 * Hop_t * cos_kx

    try:
        z = np.pi - np.arccos((1 / 2) * (T_1 / Hop_t))
    except ValueError:
        z = 0.0  # Asegurar valor válido en caso de errores numéricos

    cos_kz = np.cos(z)

    brecha = brecha_0 * (cos_kx - cos_kz)

    return brecha

# =====================================
# 📌 Términos a calcular
# =====================================

def termino_positivo(omega, alpha, brecha_k):
    """
    Calcula el término positivo.

    Args:
        omega (float): Frecuencia angular.
        alpha (float): Parámetro alfa.
        brecha_k (float): Valor de la brecha en k.

    Returns:
        float: Valor del término positivo.
    """
    a_k = omega**2 - alpha**2 - brecha_k**2
    b_k = 2 * omega * alpha
    rho_k = np.sqrt(max(0, a_k**2 + b_k**2))  # Evitar raíces negativas

    factor_sin_raiz = omega / np.sqrt(2 * rho_k + EPSILON)
    factor_con_raiz = np.sqrt(1 + a_k / (rho_k + EPSILON))

    argumento = factor_sin_raiz * factor_con_raiz

    return argumento

def termino_negativo(omega, alpha, brecha_k):
    """
    Calcula el término negativo.

    Args:
        omega (float): Frecuencia angular.
        alpha (float): Parámetro alfa.
        brecha_k (float): Valor de la brecha en k.

    Returns:
        float: Valor del término negativo.
    """
    a_k = omega**2 - alpha**2 - brecha_k**2
    b_k = 2 * omega * alpha
    rho_k = np.sqrt(max(0, a_k**2 + b_k**2))  # Evitar raíces negativas

    factor_sin_raiz = omega / np.sqrt(2 * rho_k + EPSILON)
    factor_con_raiz = np.sqrt(1 - a_k / (rho_k + EPSILON))

    argumento = factor_sin_raiz * factor_con_raiz

    return argumento
