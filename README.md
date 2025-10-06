BOA-TSAM

Implementación del algoritmo bio-inspirado de Bobcat (BOA) para el problema de TSAM (Transporte y Asignación de Módulos).

Este repositorio incluye:

El código del algoritmo BOA (BOA.py).

Un código adicional (BOA-PSO.py) que ejecuta un algoritmo PSO para el mismo problema y permite comparar los resultados con BOA.

Un informe técnico (Informe_Técnico.pdf) donde se describen los detalles del problema, los algoritmos y los resultados obtenidos.

Comparativas entre BOA y PSO/solver clásico, evaluando la calidad de la solución y el óptimo alcanzado.

Descripción

El algoritmo BOA simula el comportamiento de los bobcats para explorar y explotar el espacio de soluciones del problema TSAM. El objetivo es encontrar una asignación óptima de clientes a hubs, considerando:

Distancias máximas toleradas.

Capacidad de cada hub.

Costos fijos de apertura de hubs.

El informe técnico adjunto explica con mayor detalle:

La estructura del problema TSAM.

El funcionamiento del algoritmo BOA y del PSO.

Resultados de experimentos y análisis de convergencia.

Comparación de BOA vs PSO/solver clásico para evaluar la calidad de las soluciones encontradas.

Uso del programa

El uso es sencillo: se deben pasar dos parámetros por consola:

Número de agentes (partículas) del enjambre.

Número de iteraciones que realizará el algoritmo.

python BOA.py <num_particulas> <num_iteraciones>

Ejemplo
python BOA.py 10 50


Esto ejecutará el algoritmo con 10 agentes y 50 iteraciones.

Durante la ejecución, se generarán gráficos de convergencia, boxplot y QMetric.

Los resultados se almacenan automáticamente en el directorio /Gráficos.

Para ejecutar el algoritmo PSO y comparar resultados, se utiliza:

python BOA-PSO.py <num_particulas> <num_iteraciones>

Archivos adjuntos

BOA.py: Código fuente del algoritmo BOA.

BOA-PSO.py: Código fuente del algoritmo PSO para comparación.

Informe_Técnico.pdf: Documento explicativo del problema y los experimentos.

Notas adicionales

El algoritmo permite analizar la influencia de diferentes cantidades de agentes y número de iteraciones sobre la calidad de la solución.

Los resultados de cada ejecución se almacenan en la carpeta /Gráficos.

En el informe se incluye la comparación entre BOA y PSO/solver clásico para evaluar el óptimo alcanzado por cada algoritmo.
