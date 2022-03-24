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

### Preguntas
 - factorial falla porque en el que era el caso 78 (-84) al ser negativo se imprime por pantalla error y tal y cierra el proceso entero porque "Original program NOT ok!!" pero esto es un testeo, tengo que probar también los casos extremos de fallo no?
 - dictionaryOrder funciona ejecutar TS perfectamente pero dentro de Malone revienta muy bastamente
 - gcd revienta en cuando le paso solo un argumento
 - fallan todos los de matrices por bucles infinitos

### Cositas
 1. apps/ - Adaptar más aplicaciones
 2. DB - Llenar base de datos
 3. Cambiar el puts a print
 4. Excel con todas las ejecuciones tipo csv con los datos de ejecución de todo (modificando naos)
 5. Modificación de la función predict de naos
 6. Cambiar n 2 a n 3 en las ejecuciones del mpirun (después)

### ToDo
 - [ ] Un par de aplicaciones más adaptadas.
 - [ ] Llenar la base de datos
 - [ ] Mirar repositorio KANN
 - [ ] Mirar si con otra función final se puede hacer lo de las 11 salidas (SOFTMAX)

### Mid term
 - [ ] Dividir database en training y testing
 - [ ] Añadir al sistema [complejidad ciclomatica](https://github.com/ideadapt/metriculator)

---

### Ideas
 - La salida de la ANN va directamente al mpirun
 - Acceder directamente los archivos en $MALONE_HOME/Results
 - No cambiar la carpeta para ejecutar tanto Malone como Mutomvo
 - UI para la ANN
 - Cortar salidas para que solo muestre algoritmo u optimizaciones