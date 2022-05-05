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
 - [x] Mutate.java: zip + no generar mutantes si existen
 - [x] Eliminar duplicados de Autotest y Naos
 - [ ] Tweak hyperparameters (learning rate and training epochs)
 - [ ] Mirar CCAN
 - [ ] GUI de mutate + naos + autotest
 - [ ] Generar mutantes en mutomvo por línea de comandos

---

### Problemas
 - KANN: básicamente que estás creando modelos que luego van a ser utilizados, necesitamos la base de datos (que en este caso es MNIST por tanto es fácil) y luego el mismo modelo creado para ejecutar las demás cosas por lo que veo chungo el hacer un testFile válido para Malone
 - cjson, libpqueue y avl: los ejemplos que pone los crea a mano dentro del .c por lo que no habría que cambiar básicamente la mitad del código para hacerlo funcionar como queremos
 - bzip2: no produce nada como salida por lo que no sé qué puede pillar malone para evaluar

### Ideas
 - UI para la ANN
 - Cortar salidas para que solo muestre algoritmo u optimizaciones