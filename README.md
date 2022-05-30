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
 - [ ] Tweak hyperparameters (learning rate and training epochs) 26/320* (26/256) me tienen que aparecer 294 excluidas
 - [x] Mirar CCAN + algoritmos de búsqueda

---

### Problemas
 - KANN: básicamente que estás creando modelos que luego van a ser utilizados, necesitamos la base de datos (que en este caso es MNIST por tanto es fácil) y luego el mismo modelo creado para ejecutar las demás cosas por lo que veo chungo el hacer un testFile válido para Malone
 - cjson, libpqueue y avl: los ejemplos que pone los crea a mano dentro del .c por lo que no habría que cambiar básicamente la mitad del código para hacerlo funcionar como queremos
 - bzip2: no produce nada como salida por lo que no sé qué puede pillar malone para evaluar

### Ideas
 - UI para la ANN
 - Cortar salidas para que solo muestre algoritmo u optimizaciones
 - Generar mutantes en mutomvo por línea de comandos


Las clases están 0 diferenciadas, se equivoca en cosas que me parece muy muy normal hacerlo mal
Tengo mis dudas sobre que sea la mejor manera de predecir ya que muchas veces lo que varía el tiempo es mínimo y se podría ejecutar de varias maneras diferentes obteniendo esencialmente el mismo resultado

### SVM
  - Trabaja mejor en dimensiones altas (dimensión 6 no es alta)
  - Cuando el número de dimensiones es mayor que el número de samples (de momento tenemos 6 vs 30/45 y va a duplicarse)
  - Cuando hay un margen claro de separación trabaja mejor (las clases se solapan muy fácilmente eg massive)
  - In cases where the number of features for each data point exceeds the number of training data samples, the SVM will underperform
  - One vs One con 320 clases es una animalada (hacer 320+319+...+1 hiperplanos)
  - One vs All con 320 clases se necesitan más datos y unbalanced (hay que pillar samples de dentro de las clases)