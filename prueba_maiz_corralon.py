import math
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Parámetros fijos
producto = "maiz"
densidad_min = 700  #kg/m3
densidad_max = 820
densidad_promedio = (densidad_min + densidad_max) / 2

# Dimensiones Almacen
ancho_base = 40
largo_almacen = 100

angulo_inclinacion = 30  # grados
hueco_transito = 20

# Dimensiones saco
base_saco = 1.0
largo_saco = 1.0
alto_saco = 1.5
vol_saco = base_saco * largo_saco * alto_saco

# Objetivo
target_tonelaje = 22000
altura_fija_primera = 10  # m

lower = 0
upper = 150
tolerance = 1.0
max_iter = 100
pause = 0.05

def largo_sup_from_largo_inf(largo_inf, altura, angulo_deg):
    d = altura * math.tan(math.radians(angulo_deg))
    return largo_inf - 2 * d

def volumen_trapecio_prism(largo_inf, ancho_inf, altura, angulo_deg):
    d_ancho = altura * math.tan(math.radians(angulo_deg))
    ancho_sup = ancho_inf - 2 * d_ancho
    largo_sup = largo_sup_from_largo_inf(largo_inf, altura, angulo_deg)
    if ancho_sup <= 0 or largo_sup <= 0:
        return 0.0, largo_sup, ancho_sup
    area_inf = largo_inf * ancho_inf
    area_sup = largo_sup * ancho_sup
    volumen = altura * (area_inf + area_sup) / 2.0
    return volumen, largo_sup, ancho_sup

def calcular_tonelaje_total(largo_inf, altura):
    vol_ruma, largo_sup, ancho_sup = volumen_trapecio_prism(largo_inf, ancho_base, altura, angulo_inclinacion)
    tonelaje_ruma = vol_ruma * densidad_promedio / 1000.0

    sacas_lado1 = math.ceil(largo_inf - hueco_transito)
    sacas_lado2 = math.ceil(ancho_base)
    sacas_lado3 = math.ceil(largo_inf)
    sacas_lado4 = math.ceil(ancho_base)

    total_sacos = sacas_lado1 + sacas_lado2 + sacas_lado3 + sacas_lado4
    tonelaje_corralon = total_sacos * vol_saco * densidad_promedio / 1000.0

    tonelaje_total = tonelaje_ruma + tonelaje_corralon
    return tonelaje_total, tonelaje_ruma, tonelaje_corralon, total_sacos, largo_sup, ancho_sup, vol_ruma

# --- Primera etapa: buscar largo decimal ideal (altura fija) ---
print("Buscando largo base inferior ideal con altura fija de 10 m...\n")

lower = 0
upper = 150
iteracion = 0
largo_decimal = None

while iteracion < max_iter:
    mid = (lower + upper) / 2.0
    tonelaje_total, *_ = calcular_tonelaje_total(mid, altura_fija_primera)

    sys.stdout.write(f"\rIter {iteracion:02d}: largo = {mid:7.3f} m | Tonelaje = {tonelaje_total:10.2f} t")
    sys.stdout.flush()

    if tonelaje_total >= target_tonelaje and (tonelaje_total - target_tonelaje) < tolerance:
        if mid <= largo_almacen:
            largo_decimal = mid
            break
        else:
            # Largo ideal mayor que largo almacen: cortar aquí para calcular con largo_almacen
            largo_decimal = None
            break

    if tonelaje_total > target_tonelaje:
        upper = mid
    else:
        lower = mid

    iteracion += 1
    time.sleep(pause)

if largo_decimal is None:
    # Largo ideal mayor que largo_almacen, evaluamos con largo_almacen directamente
    tonelaje_total_max, tonelaje_ruma_max, tonelaje_corralon_max, total_sacos_max, largo_sup_max, ancho_sup_max, vol_ruma_max = calcular_tonelaje_total(largo_almacen, altura_fija_primera)

    tonelaje_restante = target_tonelaje - tonelaje_total_max

    print(f"\nLargo ideal excede el largo máximo del almacén ({largo_almacen} m).")
    print(f"Evaluando con largo máximo disponible: {largo_almacen} m")
    print(f"Tonelaje máximo con largo {largo_almacen} m: {tonelaje_total_max:.2f} toneladas")
    print(f"Tonelaje restante por almacenar: {tonelaje_restante:.2f} toneladas")
    largo_entero = largo_almacen
else:
    largo_entero = math.ceil(largo_decimal)
    print(f"\n\nLargo base inferior ideal (decimal): {largo_decimal:.3f} m")
    print(f"Redondeado hacia arriba al saco entero más próximo: {largo_entero} m")

# --- Segunda etapa: con largo fijo, buscar altura óptima ---
print("\nBuscando altura óptima con largo fijo para el tonelaje objetivo...\n")

lower = 0
upper = 50
iteracion = 0
altura_final = None

while iteracion < max_iter:
    mid = (lower + upper) / 2.0
    tonelaje_total, tonelaje_ruma, tonelaje_corralon, total_sacos, largo_sup, ancho_sup, vol_ruma = calcular_tonelaje_total(largo_entero, mid)

    sys.stdout.write(f"\rIter {iteracion:02d}: altura = {mid:6.3f} m | Tonelaje = {tonelaje_total:10.2f} t")
    sys.stdout.flush()

    if tonelaje_total >= target_tonelaje and (tonelaje_total - target_tonelaje) < tolerance:
        altura_final = mid
        break

    if tonelaje_total > target_tonelaje:
        upper = mid
    else:
        lower = mid

    iteracion += 1
    time.sleep(pause)

if altura_final is None:
    step = 0.0005
    while True:
        tonelaje_total, tonelaje_ruma, tonelaje_corralon, total_sacos, largo_sup, ancho_sup, vol_ruma = calcular_tonelaje_total(largo_entero, mid)
        if tonelaje_total >= target_tonelaje:
            altura_final = mid
            break
        mid += step

# --- Resultados ---
print("\n\nResultado final:")
print(f"Altura óptima: {altura_final:.3f} m")
print(f"Sacos por lado del corralón:")
print(f"  Lado 1 (largo con hueco): {math.ceil(largo_entero - hueco_transito)}")
print(f"  Lado 2 (ancho): {math.ceil(ancho_base)}")
print(f"  Lado 3 (largo): {math.ceil(largo_entero)}")
print(f"  Lado 4 (ancho): {math.ceil(ancho_base)}")
print(f"Dimensiones del corralón (base inferior): {largo_entero:.2f} x {ancho_base:.2f} m")
print(f"Dimensiones de la ruma (base inferior - interior): {max(0,largo_entero - 1):.2f} x {max(0,ancho_base - 1):.2f} m")
print(f"Dimensiones de la ruma (base superior): {largo_sup:.2f} x {ancho_sup:.2f} m")
print(f"Tonelaje estimado del corralón: {tonelaje_corralon:.2f} toneladas")
print(f"Tonelaje estimado de la ruma: {tonelaje_ruma:.2f} toneladas")
print(f"Tonelaje total (ruma + corralón): {tonelaje_total:.2f} toneladas")



# --- Gráfico 3D con sacos ---

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range])
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    ax.set_xlim3d(x_middle - max_range/2, x_middle + max_range/2)
    ax.set_ylim3d(y_middle - max_range/2, y_middle + max_range/2)
    ax.set_zlim3d(z_middle - max_range/2, z_middle + max_range/2)

def plot_cubo(ax, x, y, z, dx, dy, dz, color='grey', alpha=0.8):
    vertices = np.array([[x, y, z],
                         [x + dx, y, z],
                         [x + dx, y + dy, z],
                         [x, y + dy, z],
                         [x, y, z + dz],
                         [x + dx, y, z + dz],
                         [x + dx, y + dy, z + dz],
                         [x, y + dy, z + dz]])
    caras = [[vertices[j] for j in [0,1,2,3]],  # base inferior
             [vertices[j] for j in [4,5,6,7]],  # base superior
             [vertices[j] for j in [0,1,5,4]],  # lado 1
             [vertices[j] for j in [1,2,6,5]],  # lado 2
             [vertices[j] for j in [2,3,7,6]],  # lado 3
             [vertices[j] for j in [3,0,4,7]]]  # lado 4
    ax.add_collection3d(Poly3DCollection(caras, facecolors=color, edgecolors='k', linewidths=0.3, alpha=alpha))

# Parámetros sacos (fijos)
base_saco = 1.0
largo_saco = 1.0
alto_saco = 1.5

# Sacos por lado (calculados)
sacos_lado1 = math.ceil(largo_entero - hueco_transito)
sacos_lado2 = math.ceil(ancho_base)
sacos_lado3 = math.ceil(largo_entero)
sacos_lado4 = math.ceil(ancho_base)

# Dimensiones ruma según cálculo final
base_inferior_largo = max(0, largo_entero - 1)  # 88
base_inferior_ancho = max(0, ancho_base - 1)    # 39
base_superior_largo = largo_sup
base_superior_ancho = ancho_sup
altura_ruma = altura_final
altura_sacos = alto_saco

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111, projection='3d')

# 1. Dibujar sacos borde corralón con hueco de tránsito en lado 1
# Lado 1: sacos en X de 0 a sacos_lado1, Y=0
for i in range(sacos_lado1):
    plot_cubo(ax, i, 0, 0, base_saco, largo_saco, alto_saco, color='sandybrown', alpha=0.9)

# Hueco de tránsito (del lado 1 al lado 2, no dibujar nada)

# Lado 2: sacos en X = sacos_lado1 + hueco_transito, Y de 0 a sacos_lado2
for j in range(sacos_lado2):
    plot_cubo(ax, largo_entero, j, 0, base_saco, largo_saco, alto_saco, color='sandybrown', alpha=0.9)

# Lado 3: sacos en X de 0 a sacos_lado3, Y = ancho_base (40)
for i in range(sacos_lado3):
    plot_cubo(ax, i, ancho_base, 0, base_saco, largo_saco, alto_saco, color='sandybrown', alpha=0.9)

# Lado 4: sacos en X = 0, Y de 0 a sacos_lado4 (40)
for j in range(sacos_lado4):
    plot_cubo(ax, 0, j, 0, base_saco, largo_saco, alto_saco, color='sandybrown', alpha=0.9)

# 2. Dibujar pirámide truncada (ruma) dentro del corralón, justo encima de los sacos

# Base inferior de la pirámide (ruma), justo encima de los sacos del corralón
base_inf_piramide = np.array([
    [1, 1, 0],
    [base_inferior_largo + 1, 1, 0],
    [base_inferior_largo + 1, base_inferior_ancho + 1, 0],
    [1, base_inferior_ancho + 1, 0]
])

# Base superior de la pirámide (tope)

base_sup_piramide = np.array([
    [1 + (base_inferior_largo - base_superior_largo) / 2,
     1 + (base_inferior_ancho - base_superior_ancho) / 2,
     altura_sacos + altura_ruma],
    [1 + (base_inferior_largo + base_superior_largo) / 2,
     1 + (base_inferior_ancho - base_superior_ancho) / 2,
     altura_sacos + altura_ruma],
    [1 + (base_inferior_largo + base_superior_largo) / 2,
     1 + (base_inferior_ancho + base_superior_ancho) / 2,
     altura_sacos + altura_ruma],
    [1 + (base_inferior_largo - base_superior_largo) / 2,
     1 + (base_inferior_ancho + base_superior_ancho) / 2,
     altura_sacos + altura_ruma]
])

caras_piramide = [
    [base_inf_piramide[0], base_inf_piramide[1], base_inf_piramide[2], base_inf_piramide[3]],  # base inferior
    [base_sup_piramide[0], base_sup_piramide[1], base_sup_piramide[2], base_sup_piramide[3]],  # base superior
    [base_inf_piramide[0], base_inf_piramide[1], base_sup_piramide[1], base_sup_piramide[0]],
    [base_inf_piramide[1], base_inf_piramide[2], base_sup_piramide[2], base_sup_piramide[1]],
    [base_inf_piramide[2], base_inf_piramide[3], base_sup_piramide[3], base_sup_piramide[2]],
    [base_inf_piramide[3], base_inf_piramide[0], base_sup_piramide[0], base_sup_piramide[3]],
]

ax.add_collection3d(Poly3DCollection(caras_piramide, facecolors='lightblue', edgecolors='b', alpha=0.5))


# Ajustes visuales y ejes
ax.set_xlabel('Largo (m)')
ax.set_ylabel('Ancho (m)')
ax.set_zlabel('Altura (m)')
ax.set_title(f'Corralón con sacos borde y Ruma interior\nAltura óptima: {altura_ruma:.2f} m')

ax.set_xlim(0, sacos_lado3 + hueco_transito + 5)
ax.set_ylim(0, ancho_base + 5)
ax.set_zlim(0, altura_ruma + altura_sacos + 5)

set_axes_equal(ax)

plt.tight_layout()
plt.show()



# ------------------------ GRAFICO PLOTLY ---------------------

import plotly.graph_objects as go
import numpy as np

def graficar_corralon_y_ruma(
    base_inferior_largo,
    altura_fija_primera,
    hueco_transito,
    ancho_base,
    largo_sup,
    ancho_sup,
    base_saco,
    largo_saco,
    alto_saco
):
    largo_entero = base_inferior_largo
    altura_ruma = altura_fija_primera

    # Sacos por lado
    sacos_lado1 = int(np.ceil(largo_entero - hueco_transito))
    sacos_lado2 = int(np.ceil(ancho_base))
    sacos_lado3 = int(np.ceil(largo_entero))
    sacos_lado4 = int(np.ceil(ancho_base))

    # Dimensiones pirámide truncada
    base_inferior_largo_calc = max(0, largo_entero - 1)
    base_inferior_ancho = max(0, ancho_base - 1)
    base_superior_largo = largo_sup
    base_superior_ancho = ancho_sup
    altura_final = altura_ruma
    altura_sacos = alto_saco

    def cubo_plotly(x, y, z, dx, dy, dz, color='sandybrown', opacity=0.9):
        vertices = np.array([
            [x, y, z],
            [x+dx, y, z],
            [x+dx, y+dy, z],
            [x, y+dy, z],
            [x, y, z+dz],
            [x+dx, y, z+dz],
            [x+dx, y+dy, z+dz],
            [x, y+dy, z+dz]
        ])
        I = [0,0,0,3,3,1,4,4,5,6,7,7]
        J = [1,3,4,2,7,2,5,6,6,7,4,0]
        K = [3,4,5,7,4,6,6,7,7,4,0,3]

        return go.Mesh3d(
            x=vertices[:,0], y=vertices[:,1], z=vertices[:,2],
            i=I, j=J, k=K,
            color=color,
            opacity=opacity,
            flatshading=True,
            name='Cubo'
        )

    def piramide_truncada(base_inf, base_sup, altura, color='lightblue', opacity=0.5):
        vertices = np.vstack([base_inf, base_sup])
        faces = [
            [0,1,2], [0,2,3],      # base inferior
            [4,5,6], [4,6,7],      # base superior
            [0,1,5], [0,5,4],
            [1,2,6], [1,6,5],
            [2,3,7], [2,7,6],
            [3,0,4], [3,4,7]
        ]
        I, J, K = zip(*faces)
        return go.Mesh3d(
            x=vertices[:,0], y=vertices[:,1], z=vertices[:,2],
            i=list(I), j=list(J), k=list(K),
            color=color,
            opacity=opacity,
            flatshading=True,
            name='Piramide truncada'
        )

    fig = go.Figure()

    # Dibujar sacos borde (cubos)
    for i in range(sacos_lado1):
        fig.add_trace(cubo_plotly(i, 0, 0, base_saco, largo_saco, alto_saco))

    for j in range(sacos_lado2):
        fig.add_trace(cubo_plotly(largo_entero, j, 0, base_saco, largo_saco, alto_saco))

    for i in range(sacos_lado3):
        fig.add_trace(cubo_plotly(i, ancho_base, 0, base_saco, largo_saco, alto_saco))

    for j in range(sacos_lado4):
        fig.add_trace(cubo_plotly(0, j, 0, base_saco, largo_saco, alto_saco))

    # Coordenadas pirámide truncada
    base_inf_piramide = np.array([
        [1, 1, 0],
        [base_inferior_largo_calc + 1, 1, 0],
        [base_inferior_largo_calc + 1, base_inferior_ancho + 1, 0],
        [1, base_inferior_ancho + 1, 0]
    ])

    base_sup_piramide = np.array([
        [1 + (base_inferior_largo_calc - base_superior_largo) / 2,
         1 + (base_inferior_ancho - base_superior_ancho) / 2,
         altura_sacos + altura_final],
        [1 + (base_inferior_largo_calc + base_superior_largo) / 2,
         1 + (base_inferior_ancho - base_superior_ancho) / 2,
         altura_sacos + altura_final],
        [1 + (base_inferior_largo_calc + base_superior_largo) / 2,
         1 + (base_inferior_ancho + base_superior_ancho) / 2,
         altura_sacos + altura_final],
        [1 + (base_inferior_largo_calc - base_superior_largo) / 2,
         1 + (base_inferior_ancho + base_superior_ancho) / 2,
         altura_sacos + altura_final]
    ])

    fig.add_trace(piramide_truncada(base_inf_piramide, base_sup_piramide, altura_final))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Largo (m)', range=[0, sacos_lado3 + hueco_transito + 5], backgroundcolor="white"),
            yaxis=dict(title='Ancho (m)', range=[0, ancho_base + 5], backgroundcolor="white"),
            zaxis=dict(title='Altura (m)', range=[0, altura_final + altura_sacos + 5], backgroundcolor="white"),
            aspectmode='manual',
            aspectratio=dict(
                x=1,
                y=(ancho_base + 5) / (sacos_lado3 + hueco_transito + 5),
                z=(altura_final + altura_sacos + 5) / (sacos_lado3 + hueco_transito + 5)
            )
        ),
        title=f'Corralón con sacos borde y Ruma interior\nAltura óptima: {altura_final:.2f} m',
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()


graficar_corralon_y_ruma(
    base_inferior_largo=largo_entero,
    altura_fija_primera=altura_final,
    hueco_transito=hueco_transito,
    ancho_base=ancho_base,
    largo_sup=largo_sup,
    ancho_sup=ancho_sup,
    base_saco=base_saco,
    largo_saco=largo_saco,
    alto_saco=alto_saco
)





