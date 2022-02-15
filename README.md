# TFG Ingeniería Informática

Trabajo de Fin de Grado de Ingeniería Informática en la UAM
- [Proyecto en Overleaf](https://www.overleaf.com/project/619df580e0cdd6ba1598798b)

### Autor
| Martín de las Heras Moreno
| --------------------------
| martin.delasheras@estudiante.uam.es

### Tutor
| Pablo Cerro Cañizares
| --------------------------
| pablo.cerro@uam.es

---

### ToDo
 - [ ] Adaptar más aplicaciones
 - [ ] Insertar los parámetros a la ANN.
 - [ ] Generar automáticamente los entornos ini (Malone/Environments/autotest) 
 - [ ] Ver #mutante y test que falla

### Mid term
 - [ ] Intentar saber qué cantidad de modelos con los que es necesario entrenar el modelo inicial. 
 - [ ] Prototipo muy inicial en java de la herramienta del TFG: Sería una capa superior a Malone, que lo gestiona, permite ejecutarlo, obtener los datos y realizar la predicción con la red neuronal.

---

### Ideas
 - La salida de la ANN va directamente al mpirun
 - Acceder directamente los archivos en $MALONE_HOME/Results
 - No cambiar la carpeta para ejecutar tanto Malone como Mutomvo
 - Subcarpeta en outputs con el nombre del programa
 - Meter en el fichero de los datos de ejecución desde el principio todas las métricas que pueda (puedo meter tanto mutantes como tests como lineas del .c)

### Preguntas
 1. Tema Inputs/Outputs cerrarlo (params del ordenador?)
 2. De apps/ aplicacion.zip y testSuite
