import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt

# Cargar datos GeoJSON
gdf = gpd.read_file('C:\\Users\\Usuario\\Desktop\\proyecto\\Estaciones_Troncales_de_TRANSMILENIO.geojson') # Asegúrate de usar la ruta correcta

# Crear el grafo
G = nx.Graph()

# Añadir nodos (estaciones) al grafo basado en los datos GeoJSON
for index, row in gdf.iterrows():
    G.add_node(row['nombre_estacion'], pos=(row['coordenada_x_estacion'], row['coordenada_y_estacion']))

# Suposición inicial de conexiones secuenciales dentro de las troncales
troncales = gdf.groupby('troncal_estacion')
for troncal, frame in troncales:
    estaciones_ordenadas = frame.sort_values('objectid')
    for i in range(len(estaciones_ordenadas) - 1):
        estacion_actual = estaciones_ordenadas.iloc[i]
        estacion_siguiente = estaciones_ordenadas.iloc[i + 1]
        G.add_edge(estacion_actual['nombre_estacion'], estacion_siguiente['nombre_estacion'])

# Añadir manualmente conexiones específicas entre estaciones, si se conocen
# G.add_edge('NombreEstacion1', 'NombreEstacion2')

# Función para encontrar la ruta más corta usando Dijkstra
def encontrar_ruta_mas_corta(origen, destino):
    try:
        ruta = nx.shortest_path(G, source=origen, target=destino)
        return ruta
    except nx.NetworkXNoPath:
        return "No hay ruta disponible entre las estaciones seleccionadas."

# Usar la función para encontrar una ruta
origen = 'Alcalá'  # Sustituir con el nombre exacto de la estación de origen
destino = 'Portal Norte'  # Sustituir con el nombre exacto de la estación de destino
ruta = encontrar_ruta_mas_corta(origen, destino)
print("Ruta más corta encontrada:", ruta)

# Opcional: Visualización básica del grafo
# Esto puede ayudar a visualizar cómo están conectadas las estaciones en tu grafo
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True, node_size=50, font_size=8)
plt.show()

import pandas as pd

# Crear un DataFrame con datos de ejemplo
data = {
    'Hora_del_día': [7, 9, 12, 15, 18, 20],  # Hora del día en formato 24h
    'Día_de_la_semana': ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado'],
    'Estación_origen': ['Portal del Norte', 'Museo del Oro', 'Marly', 'NQS Calle 75', 'Av. Jiménez', 'Portal de Usme'],
    'Estación_destino': ['Universidades', 'Calle 100', 'Calle 45', 'Suba', 'Portal del Sur', 'Portal de Suba'],
    'Número_pasajeros': [200, 180, 150, 300, 250, 100],  # Número estimado de pasajeros
    'Tiempo_viaje_min': [20, 25, 15, 30, 35, 40]  # Tiempo de viaje en minutos
}

df = pd.DataFrame(data)

print(df)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Convertir datos categóricos en numéricos
df_encoded = pd.get_dummies(df, columns=['Día_de_la_semana', 'Estación_origen', 'Estación_destino'])

# Definir las variables independientes (X) y la variable dependiente (y)
X = df_encoded.drop('Tiempo_viaje_min', axis=1)
y = df_encoded['Tiempo_viaje_min']

# Dividir los datos en conjunto de entrenamiento y de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)



#ultima actividad 

import pandas as pd

# Crear un DataFrame con datos de ejemplo
data = {
    'ID_Usuario': range(1, 11),
    'Viajes_Mañana': [5, 2, 3, 4, 1, 2, 5, 4, 3, 2],
    'Viajes_Tarde': [1, 3, 2, 1, 4, 5, 2, 3, 4, 5],
    'Viajes_Noche': [0, 1, 0, 0, 2, 1, 0, 1, 0, 2],
    'Estación_Frecuente': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B', 'C', 'A']
}

df = pd.DataFrame(data)

print(df)


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Preparación de datos para el clustering (excluyendo ID de usuario y datos categóricos)
X = df[['Viajes_Mañana', 'Viajes_Tarde', 'Viajes_Noche']].values

# Ejecutar K-Means
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)

# Añadir las etiquetas de cluster al DataFrame original
df['Cluster'] = kmeans.labels_

# Visualizar los clusters
plt.scatter(df['Viajes_Mañana'], df['Viajes_Tarde'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Viajes en la Mañana')
plt.ylabel('Viajes en la Tarde')
plt.title('Clusters de Usuarios por Viajes en la Mañana vs. Tarde')
plt.show()
