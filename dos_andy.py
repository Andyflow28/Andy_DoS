import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quadrature
import sys
import os

# Ajustar el path para importar el archivo de funciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))
from utilidades_para_calculos import *

# Declaración de constantes
Delta_0 = 33.90
parametro_de_red = 1
Hop_t = -0.2
E_n = -0.04
Limite_inferior = np.pi - np.arccos(E_n / (4 * Hop_t))
Limite_superior = np.pi

# Calculo del nivel de Fermi usando Gauss-Kronrod
# cuadrature retorna dos valores: el resultado y el error estimado.
N_F, error = quadrature(jacobian, Limite_inferior, Limite_superior, args=(Hop_t, E_n, parametro_de_red))

print(f"Nivel de Fermi: {N_F} (Error estimado: {error})")
