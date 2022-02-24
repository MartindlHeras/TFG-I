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
 - NAOS
    - entrenar:
        1. Hace lo mismo que el main de FileParser
        2. Pasa los parámetros a la ANN para entrenarla en ese ejemplo
    - predecir:
        - Repurpose test/ANN
        1. Llamar a la red neuronal con los parámetros de entrada
        2. Devuelve algoritmo (+ optimizaciones)
    - main:
        1. Input: fichero .c + mutantes + tests? (sacar #lineas?)
        2. Opción -t: entrenar
        3. Opción -p: predecir

#### Learn
 - Training epochs
 - Hidden layers
    - Number
    - Size
 - Activation function (RELU/SOFTMAX)
 - Loss function (negative log likelihood)
 - Weight initialization (Xavier)
 - Batch sizes

#### Inputs (6)
 - Nombre del programa (no cuenta)
 - #mutantes
 - #tests
 - #cores
 - #algoritmo
 - #optimizaciones
 - #lineas del .c (**falta**)
 - #especificaciones del ordenador? (*muy problemático a la hora de entrenar*)
 - max num de cores
 - *tiempo total (tiempos totales?) (solo para entrenar)*
 - *tiempo de cada mutante? (solo para entrenar)*
 - *tamaño del fichero?*

Al final los inputs tienen que ser datos que se sepan antes de ejecutar porque el objetivo del proyecto es hacer un predictor, el tiempo de ejecución solo puede servir para determinar qué algoritmo (y optimizaciones) es el mejor, por lo que el tiempo de los mutantes es creo que irrelevante para la ANN

#### Outputs (5-11)
 - Algoritmo (1-5)
 - Optimizaciones (0-1 en cada una de las 6 optimizaciones)


#### ANNs
 - [Github de ANN en Java](https://github.com/yacineMahdid/artificial-intelligence-and-machine-learning/tree/master/Neural%20Network%20from%20Scratch%20in%20Java/src)
 - [Vídeo del GitHub de encima](https://www.youtube.com/watch?v=1DIu7D98dGo)
 - [MNIST Database](http://yann.lecun.com/exdb/mnist/)
 - [DL4j Example](https://towardsdatascience.com/part-5-training-the-network-to-read-handwritten-digits-c2288f1a2de3)
 - [Digit recognizer](https://itnext.io/building-a-handwritten-digit-recognizer-in-java-4eca4014eb2f)
 - [Guía de DL4J de verdad que funciona](https://www.rcp-vision.com/build-your-first-neural-network-with-eclipse-deeplearning4j/)
 - [Serialization Java](https://www.tutorialspoint.com/java/java_serialization.htm)
 - [Visualization UI](https://deeplearning4j.konduit.ai/deeplearning4j/how-to-guides/tuning-and-training/visualization)
 - [Saving and loading](https://deeplearning4j.konduit.ai/deeplearning4j/reference/saving-and-loading-models)