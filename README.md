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
 1. NAOS - Argumentos de entrada
 2. NAOS - Carpeta de entrada (Results)
 3. ANN - Regenerar con nuevas capas
 4. ANN - Metadatos
 5. DB - Llenar base de datos
 6. Script - Automatizar que dándole al script se haga todo
 7. README - apps/
 8. Complejidad ciclomática?

 ### Preguntas
 1. Tema Inputs: si tengo la carpeta de Results de dónde saco las líneas del .c y el tamaño del TS?

### ToDo
 - [x] Argumentos de entrada de NAOS
 - [ ] Dividir database en training y testing
 - [x] Ver #procesadores
 - [ ] Ver si se pueden incluir metadatos en la fase de entrenamiento, que despues no haga falta incluir en la fase de predicción. (Metadatos como en [lo de DeepMind](https://www.youtube.com/watch?v=AO6ID_xoqq4))
 - [ ] Un par de aplicaciones más adaptadas.
 - [ ] Llenar la base de datos -> probar qué tal predice.
 - [ ] Adaptar el script para soportar que, dada una aplicacion, se ejecute todo lo que necesitamos para las ANN.
 - [ ] Tabla en latex, que incluya: nombre app, LoC, grado de complejidad o tamaño, descripción.
 - [ ] Meterle mas parametros a la ANN.


### Mid term
 - [ ] Intentar saber qué cantidad de modelos con los que es necesario entrenar el modelo inicial. 
 - [ ] Anyadir al sistema complejidad ciclomatica: [https://github.com/ideadapt/metriculator](https://github.com/ideadapt/metriculator)

---

### Ideas
 - La salida de la ANN va directamente al mpirun
 - Acceder directamente los archivos en $MALONE_HOME/Results
 - No cambiar la carpeta para ejecutar tanto Malone como Mutomvo
 - UI para la ANN
