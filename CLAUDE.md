# Proyecto: Atribución multicanal (Olist Marketing Funnel)

## Decisiones ya tomadas (no cambiar sin avisar)
- Periodo de análisis: leads de enero a mayo de 2018 (los cierres de 2017 están incompletos).
- Variable objetivo: `conv_90` (cierre dentro de 90 días).
- Variables del modelo: origin, lp_grupo, dia_semana. No usar columnas de closed_deals (fuga de información).
- Split temporal: entrenamiento ene-mar 2018, prueba abr-may 2018.
- Los nulos de `origin` se agrupan con `unknown`.
- Cada lead tiene un solo canal y una sola landing page: la comparación de métodos de atribución se hace en recorridos simulados y debe declararse como tal.

## Estructura
data/, notebooks/ (01 a 04), src/markov.py, app/app.py (Streamlit), docs/.

## Reglas
- Pedir confirmación antes de borrar archivos, instalar paquetes, hacer git push o modificar los CSV originales de data/.
- No modificar celdas existentes de los notebooks: añadir celdas nuevas al final.
- Si cambia algún resultado numérico, avisar: el documento IEEE cita esas cifras.
- Responder en español.