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
 1. Autotest - Generar 64*5 por cada iteración de cores 2-4-8-16-32 -> .txt (mpirun -n 2 ./malone -e autotest/test_autotest_$1_stand.ini -a 4 estos comandos en un txt)
 2. NAOS - Carpeta de apps/ para sacar #lineas y tamaño TS
 3. apps/ - Terminar de adaptar las aplicaciones
 4. apps/ - Adaptar más aplicaciones
 5. DB - Llenar base de datos

### ToDo
 - [x] Adaptar el script para soportar que, dada una aplicacion, se ejecute todo lo que necesitamos para las ANN.
 - [x] Editar Autotest.java -> (tiene que generar 64 por cada algoritmo y luego ir aumentando número de cores de 2^1 a 2^5 = 320*5).
 - [x] Pasar la carpeta del repositorio de aplicacion, para sacar datos de la app.
 - [ ] Un par de aplicaciones más adaptadas.
 - [ ] Llenar la base de datos
 - [ ] Mirar transformer
 - [ ] Mirar repositorio KANN

### Mid term
 - [ ] Dividir database en training y testing
 - [ ] Añadir al sistema [complejidad ciclomatica](https://github.com/ideadapt/metriculator)

---

### Ideas
 - La salida de la ANN va directamente al mpirun
 - Acceder directamente los archivos en $MALONE_HOME/Results
 - No cambiar la carpeta para ejecutar tanto Malone como Mutomvo
 - UI para la ANN
