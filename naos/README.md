## Neural Network Algorithm and Optimisation Selector

### Esquema del proyecto
 - FileParser
    - main (provisional):
        1. Crea y llama a fp
        2. Llama a getInputs
        3. Escribe por pantalla
    - getInputs:
        1. Saca parámetros comunes del nombre del fichero
        2. Recorre Results entrando en las carpetas que tengan los mismos parámetros comunes
        3. Guarda el menor tiempo y algoritmo correspondiente (en un futuro optimizaciones)
        4. Devuelve parámetros comunes + algoritmo (+ optimizaciones)
 - NNAS
    - entrenar:
        1. Hace lo mismo que el main de FileParser
        2. Pasa los parámetros a la ANN para entrenarla en ese ejemplo (más complejo de lo que parece)
    - predecir:
        - Repurpose test/ANN
        1. Llamar a la red neuronal con los parámetros de entrada
        2. Devuelve algoritmo (+ optimizaciones)
    - main:
        1. Input: fichero .c + mutantes + tests? (sacar #lineas?)
        2. Opción -t: entrenar
        3. Opción -p: predecir

#### Inputs (6)
 - Nombre del programa (no cuenta)
 - #mutantes
 - #tests
 - #cores
 - #algoritmo
 - #optimizaciones
 - #lineas del .c (**falta**)
 - *tiempo total (tiempos totales?) (solo para entrenar)*
 - *tiempo de cada mutante? (solo para entrenar)*
 - *tamaño del fichero?*

Al final los inputs tienen que ser datos que se sepan antes de ejecutar porque el objetivo del proyecto es hacer un predictor, el tiempo de ejecución solo puede servir para determinar qué algoritmo (y optimizaciones) es el mejor, por lo que el tiempo de los mutantes es creo que irrelevante para la ANN

#### Outputs (5-11)
 - Algoritmo (1-5)
 - Optimizaciones (0-1 en cada una de las 6 optimizaciones)
