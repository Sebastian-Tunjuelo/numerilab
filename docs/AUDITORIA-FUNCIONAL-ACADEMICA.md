# Auditoría funcional y académica de NumériLab

**18 de noviembre de 2026**

## Dictamen

NumériLab cubre los cuatro temas indicados en la [foto de clase](../Captura%20de%20pantalla%202026-11-17%20213107.png). El proyecto es defendible para una buena nota porque permite reproducir el procedimiento, explicar cuándo funciona cada método y examinar sus límites. Ante la ausencia de rúbrica oficial, la evaluación se basó en cobertura, exactitud, explicación paso a paso y manejo de errores. Los [requisitos internos](SPEC.md) sirvieron para orientar las verificaciones de funcionamiento.

| Tema | Evidencia para la sustentación |
| --- | --- |
| Factorización L.U. | Doolittle, Crout y pivoteo parcial. Se verificaron la identidad PA = LU, la solución y el flujo visible de sustitución. El caso de pivote nulo permite explicar el intercambio de filas. |
| Jacobi | Se comprobó el primer paso contra la forma matricial, además del radio espectral y el residual. La tabla permite examinar la convergencia y el criterio de parada. |
| Gauss-Seidel y SOR | Se verificó el uso de componentes recién calculadas y el barrido de 38 valores de ω. En el caso tridiagonal, SOR con ω = 1,05 tardó 8 iteraciones, Gauss-Seidel 11 y Jacobi 19 bajo los parámetros predeterminados. |
| Newton-Raphson | Se ejecutaron una raíz escalar y sistemas 2×2 y 3×3 en el navegador. El inspector muestra el Jacobiano evaluado y permite explicar cómo la factorización L.U. obtiene la corrección en cada iteración. |

## Correcciones

- El veredicto confundía la posibilidad de usar pivoteo con la existencia de una solución única. Ahora refleja el resultado de la factorización y distingue una matriz singular o casi singular de un pivote que requiere cambiar de estrategia.
- Cuando un paso de tamaño positivo llegaba al vector cero, el error relativo aparecía como 100 %. Ahora se devuelve como indefinido y el criterio relativo espera otra iteración. Un error relativo fuera de rango tampoco invalida un residual que sí es calculable.
- Gauss-Seidel y SOR con ω = 1 ejecutaban operaciones de relajación que el contador omitía. Ahora usan directamente la actualización de Gauss-Seidel. El comparador indica las iteraciones del mismo método al que atribuye el menor costo.
- El ejemplo de ventaja SOR cargaba ω = 1,25 y necesitaba 14 iteraciones frente a 11 de Gauss-Seidel. Ahora carga ω = 1,05, que fue el mejor valor del barrido en el caso predeterminado.

## Verificación local

- **Backend.** 76 pruebas aprobadas con pytest, incluidas regresiones para el caso singular y el error relativo fuera de rango.
- **Referencia numérica.** 300 factorizaciones y 300 primeros pasos iterativos contrastados con NumPy en dimensiones de 2 a 6. El contraste incluyó la solución, las formas matriciales, el radio espectral y los residuales.
- **HTTP.** 7 peticiones y 20 aserciones aprobadas en Postman contra el backend local.
- **Navegador.** Chromium recorrió los cuatro temas, el cambio a pivoteo, el barrido, el inspector del Jacobiano, las exportaciones CSV y LaTeX, la guía y el cambio de tema. La vista de 390 px comprobada no presentó desbordamiento horizontal de página. Se revisaron errores de consola durante el recorrido.
- **Compilación.** TypeScript y Vite terminaron correctamente. Vite mantiene una advertencia por el tamaño de dos fragmentos. Pytest muestra una advertencia de deprecación del cliente de pruebas.

La evidencia corresponde a la ejecución local en Chromium y a los casos probados. La evaluación del docente, otros motores de navegador y un eventual despliegue quedan sin comprobar.
