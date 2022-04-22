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

### Cositas
 - El problema con KANN es básicamente que estás creando modelos que luego van a ser utilizados, necesitamos la base de datos (que en este caso es MNIST por tanto es fácil) y luego el mismo modelo creado para ejecutar las demás cosas por lo que veo chungo el hacer un testFile válido para Malone
 - El problema con cjson es que los ejemplos que pone los crea a mano dentro del .c por lo que no habría que cambiar básicamente la mitad del código para hacerlo funcionar como queremos
 - Bzip2 no produce nada como salida por lo que no sé qué puede pillar malone para evaluar

### ToDo
 - [x] Tabla latex de las cosas (completar la del readme)
 - [x] Modificar autotest para eliminar los cosos esos
 - [x] En vez de compilar con javac exportar a .jar
 - [x] Mirar repositorio KANN (ae, mlp, mnist-cnn, rnn-bit, textgen, vae)
 - [ ] mutate.sh -> mutate.java
 - [ ] Mirar si con otra función final se puede hacer lo de las 11 salidas (SOFTMAX)

### Mid term
 - [ ] Cambiar n 2 a n 3 en las ejecuciones del mpirun
 - [ ] Dividir database en training y testing
 - [ ] Añadir al sistema [complejidad ciclomatica](https://github.com/ideadapt/metriculator)

---

### Ideas
 - La salida de la ANN va directamente al mpirun
 - Acceder directamente los archivos en $MALONE_HOME/Results
 - No cambiar la carpeta para ejecutar tanto Malone como Mutomvo
 - UI para la ANN
 - Cortar salidas para que solo muestre algoritmo u optimizaciones