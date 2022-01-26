# TFG Ingeniería Informática

Trabajo de Fin de Grado de Ingeniería Informática en la UAM
- [Proyecto en Overleaf](https://www.overleaf.com/project/619df580e0cdd6ba1598798b)

#### Autor
| Martín de las Heras Moreno
| --------------------------
| martin.delasheras@estudiante.uam.es

#### Tutor
| Pablo Cerro Cañizares
| --------------------------
| pablo.cerro@uam.es

#### ToDo
 - [x] Ver cómo guardar los weights de la ANN (extender a reconocer dígitos) y convertirlo en un programa ejecutable.
 - [x] Incluir un par de aplicaciones más: una sencilla y alguna más compleja.
 - [x] stand.ini file encontrar la manera de averiguar cuántos mutantes hay de verdad y cambiar la variable TotalMutants
 - [x] Demo ANN: selecciona una aplicación externa y vemos su integración en java y como funciona.
 - [ ] Mutantes empiezan por el 1 y no por el 0
 - [ ] Programas más complejos para ejecutar?
 - [ ] Prototipo muy inicial en java de la herramienta del TFG: Sería una capa superior a Malone, que lo gestiona, permite ejecutarlo, obtener los datos y realizar la predicción con la red neuronal. Extender el script.
 - [ ] Intentar saber que cantidad de modelos con los que es necesario entrenar el modelo inicial. ??
 - [ ] Rehacer la función de lectura de datos de la ANN para los de ahora

#### Ideas
 - La salida de la ANN va directamente al mpirun
 - Acceder directamente los archivos en $MALONE_HOME/Results
 - No cambiar la carpeta para ejecutar tanto Malone como Mutomvo
 - Subcarpeta en outputs con el nombre del programa
 - Meter en el fichero de los datos de ejecución desde el principio todas las métricas que pueda (puedo meter tanto mutantes como tests como lineas del .c)

#### C Programs
 - [Programas en C en un archivo](https://github.com/nothings/single_file_libs)
 - [Single file C](https://www.programiz.com/c-programming/examples)
 - [Compresión MIT](https://people.csail.mit.edu/smcc/projects/single-file-programs/)

#### ANNs
 - [Github de ANN en Java](https://github.com/yacineMahdid/artificial-intelligence-and-machine-learning/tree/master/Neural%20Network%20from%20Scratch%20in%20Java/src)
 - [Vídeo del GitHub de encima](https://www.youtube.com/watch?v=1DIu7D98dGo)
 - [MNIST Database](http://yann.lecun.com/exdb/mnist/)
 - [DL4j Example](https://towardsdatascience.com/part-5-training-the-network-to-read-handwritten-digits-c2288f1a2de3)
 - [Digit recognizer](https://itnext.io/building-a-handwritten-digit-recognizer-in-java-4eca4014eb2f)
 - [Guía de DL4J de verdad que funciona](https://www.rcp-vision.com/build-your-first-neural-network-with-eclipse-deeplearning4j/)
 - [Serialization Java](https://www.tutorialspoint.com/java/java_serialization.htm)