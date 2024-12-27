import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Usar un backend interactivo para gráficos
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sys
import os

# Ajustar el path para importar el archivo de funciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))
from utilidades_para_calculos import *

# ⚙️ Declaración de constantes
Delta_0 = 33.90
parametro_de_red = 1
Hop_t = -0.2
E_n = -0.04
epsilon = 1e-12  # Para evitar divisiones por cero

# ⚙️ Cálculo de límites de integración
Limite_inferior = np.pi - np.arccos(E_n / (4 * Hop_t))
Limite_superior = np.pi

# ⚙️ Función jacobian modificada (en utilidades_para_calculos.py)
# Asegúrate de que esta función use epsilon para evitar divisiones por cero.

# ⚙️ Cálculo del nivel de Fermi usando scipy.integrate.quad
try:
    N_F, error = quad(
        jacobian,
        Limite_inferior,
        Limite_superior,
        args=(Hop_t, E_n, parametro_de_red),
        epsabs=1e-6,  # Tolerancia absoluta
        epsrel=1e-6,  # Tolerancia relativa
        limit=1000    # Máximo número de subdivisiones
    )
    print(f"\n✅ Nivel de Fermi: {N_F:.6f} (Error estimado: {error:.6e})")
except Exception as e:
    print(f"❌ Error en la integración: {e}")

# 📊 Visualización de la función integrando
try:
    x = np.linspace(Limite_inferior, Limite_superior, 1000)
    y = [jacobian(xi, Hop_t, E_n, parametro_de_red) for xi in x]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='jacobian(x)', color='b')
    plt.xlabel('x')
    plt.ylabel('jacobian(x)')
    plt.title('Comportamiento del integrando')
    plt.legend()
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"❌ Error al graficar: {e}")
    plt.savefig('grafica_integrando.png')
    print("✅ La gráfica ha sido guardada como 'grafica_integrando.png'")
