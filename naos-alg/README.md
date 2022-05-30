## Neural Network Algorithm and Optimization Selector

### Stuff
 - >1 hidden layer, have same no. of hidden units in every layer (usually the more the better).
 - High bias (underfit)
   - Decreasing lambda (regularization parameter)
   - Adding polynomial features
   - Getting additional features
   - Small network prone to this
 - Logistic regression


### Esquema del proyecto
 - Trainer
    - getInputs:
      1. Saca parámetros comunes del nombre del fichero
      2. Recorre Results entrando en las carpetas que tengan los mismos parámetros comunes
      3. Guarda el menor tiempo y algoritmo correspondiente (en un futuro optimizaciones)
      4. Devuelve parámetros comunes + algoritmo (+ optimizaciones)
    - getFullInputs:
      1. Saca los parámetros de todos los ficheros
    - fill:
      1. Llama a las dos funciones de getInputs
      2. Escribe los datos en las bases de datos
    - train:
      1. Llama a fill
      2. Crea un objeto DB que utiliza para hacer fit otra vez con la red neuronal
 - Predict
    - mutate:
      1. Manda los ficheros a Mutomvo (.c y tests) creando la carpeta si hace falta
      2. Ejecuta mutomvo para generar los mutantes del programa
      3. Comprime los mutantes y los manda a la carpeta de la app
      4. Genera los autotests
    - predict:
      1. Obtiene los inputs mutants, tests, tsSize y lines
      2. Comprueba si hay valor de cores, si no saca el valor del ordenador
      3. Llama a autotest para preparar una ejecución de Malone de solo el programa sin mutar
      4. Realiza la ejecución en Malone y obtiene el valor del tiempo de ejecución
      5. Crea un objeto input con los datos que se pasan por la red
      6. El proceso se realiza dos veces para las optimizaciones 000000 y 100000 y el algoritmo 4
 - Autotest
    - generate:
      1. Genera autotests de todos los posibles modos de ejecución
      2. Genera un fichero bash para ejecutar los autotests
    - generateSingle:
      1. Igual que generate pero modificando para que solo ejecute el programa original
 - Naos
    - Interfaz gráfica del proyecto, llama a las funciones:
      - fill
      - train
      - mutate
      - predict

#### Inputs (6)
 - Nombre del programa
 - #Mutantes
 - #Tests
 - #Lineas del .c
 - #Cores
 - Tiempo total de ejecucion del programa sin mutar
 - Tamaño del Test Suite
 - [Complejidad ciclomática](https://github.com/ideadapt/metriculator)
 - *Tiempo total (solo para entrenar)*
 - *Tiempo de cada mutante? (solo para entrenar)*
 - *Mutation Score (solo para entrenar)*

#### Outputs (320)
 - Algoritmo (1-5)
 - Optimizaciones (0-1 en cada una de las 6 optimizaciones)
 - Salida: 64*(alg-1)+op


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