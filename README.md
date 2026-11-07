# Atribución multicanal: Olist Marketing Funnel

Proyecto final. Modelo de atribución con cadenas de Markov, XGBoost y simulación de recorridos multicanal, con dashboard en Streamlit. Metodología CRISP-DM.

## Estructura

- `data/`: datos originales y procesados
- `notebooks/`: 01 EDA, 02 journeys y Markov, 03 modelos, 04 simulación
- `src/`: lógica reutilizable (`markov.py`)
- `app/`: dashboard Streamlit y modelo entrenado
- `docs/`: documento técnico y figuras

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En Mac, XGBoost requiere `brew install libomp`.

## Datos

Olist Marketing Funnel (Kaggle): https://www.kaggle.com/datasets/olistbr/marketing-funnel-olist

Coloca los CSV originales en `data/`:

- `olist_marketing_qualified_leads_dataset.csv`
- `olist_closed_deals_dataset.csv`

Ejecuta los notebooks en orden (01 a 04). Generan `data/leads_procesados.csv`, `data/atribucion_real.csv`, `data/atribucion_lp.csv`, `data/atribucion_simulada.csv`, `data/resultados_mae.csv`, `app/modelo_xgb.json`, `app/columnas_modelo.json` y `docs/shap_summary.png`, que usa el dashboard.

## Dashboard

```bash
streamlit run app/app.py
```

Secciones: canales, impacto, simulador de escenarios y metodología.

## Resultados principales

- La conversión depende más de la landing page que del canal.
- Con un solo toque por lead, las reglas tradicionales de atribución coinciden entre sí; la comparación con Markov se hizo en recorridos simulados.
- XGBoost mejora la línea base (AUC-PR 0,196 frente a 0,106) pero no supera de forma significativa a una regresión logística.

## Limitaciones

Cada lead tiene un solo canal y una sola landing page. Los recorridos multicanal de la comparación son sintéticos, calibrados con tasas reales, y su "verdad" es una definición propia. El simulador de escenarios estima asociaciones, no efectos causales. Detalle completo en `docs/`.

## Autor

Danilo Stiven Collo Medina y Emerson Andres Medina Pencue, Fundación Universitaria de Popayán.

Repositorio: [COMPLETAR: enlace de GitHub]
