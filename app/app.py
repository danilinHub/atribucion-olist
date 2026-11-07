import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
st.set_page_config(page_title="Atribución multicanal | Olist", layout="wide")


# ---------------------------------------------------------------- datos
@st.cache_data
def cargar_leads():
    df = pd.read_csv(ROOT / "data" / "leads_procesados.csv",
                     parse_dates=["first_contact_date", "won_date"])
    df["dia_semana"] = df["first_contact_date"].dt.day_of_week
    df["mes"] = df["first_contact_date"].dt.to_period("M").astype(str)
    return df


def leer(nombre):
    ruta = ROOT / "data" / nombre
    return pd.read_csv(ruta, index_col=0) if ruta.exists() else None


def largo(wide):
    return (wide.rename_axis("canal").reset_index()
                .melt(id_vars="canal", var_name="metodo", value_name="credito"))


@st.cache_resource
def preparar_modelo(df):
    ruta_m = ROOT / "app" / "modelo_xgb.json"
    ruta_c = ROOT / "app" / "columnas_modelo.json"
    if not (ruta_m.exists() and ruta_c.exists()):
        return None
    modelo = xgb.XGBClassifier()
    modelo.load_model(ruta_m)
    cols = json.load(open(ruta_c))
    p_raw = modelo.predict_proba(features(df, cols))[:, 1]
    # El modelo se entrenó con scale_pos_weight: sus probabilidades están
    # infladas. K las reescala para que el promedio iguale la tasa observada.
    k = df["conv_90"].mean() / p_raw.mean()
    return modelo, cols, k, np.clip(p_raw * k, 0, 1)


def features(d, cols):
    X = pd.get_dummies(d[["origin", "lp_grupo", "dia_semana"]],
                       columns=["origin", "lp_grupo"])
    return X.reindex(columns=cols, fill_value=0).astype(float)


def predecir(d, modelo, cols, k):
    return np.clip(modelo.predict_proba(features(d, cols))[:, 1] * k, 0, 1)


def armar_escenario(df, p_base, modo, modelo, cols, k, **kw):
    d = df[["origin", "lp_grupo", "dia_semana"]].copy()
    d["p"] = p_base
    d["w"] = 1.0
    if modo == "mover":
        m = d["origin"] == kw["origen"]
        mov = d[m].copy()
        mov["origin"] = kw["destino"]
        mov["p"] = predecir(mov, modelo, cols, k)
        mov["w"] = kw["pct"]
        d.loc[m, "w"] = 1 - kw["pct"]
        d = pd.concat([d, mov])
    else:
        d["w"] = d["origin"].map(kw["cambios"]).fillna(0) + 1
    return d


def resumen(d):
    d = d.assign(conv=d["w"] * d["p"])
    return d.groupby("origin").agg(leads=("w", "sum"), conv=("conv", "sum"))


# ---------------------------------------------------------------- app
st.title("Atribución multicanal: Olist Marketing Funnel")
st.caption("Leads de enero a mayo de 2018. Conversión = cierre dentro de 90 días.")

if not (ROOT / "data" / "leads_procesados.csv").exists():
    st.error("No se encontró data/leads_procesados.csv. Corre antes el notebook 02.")
    st.stop()

df = cargar_leads()
tab1, tab2, tab3, tab4 = st.tabs(["Canales", "Impacto", "Simulador", "Metodología"])

# ---------------------------------------------------------------- canales
with tab1:
    st.subheader("Conversión por canal de origen")
    por_canal = (df.groupby("origin")
                   .agg(leads=("mql_id", "count"),
                        cierres_90d=("conv_90", "sum"),
                        tasa_90d=("conv_90", "mean"))
                   .sort_values("tasa_90d", ascending=False))
    c1, c2 = st.columns([1, 2])
    c1.dataframe(por_canal.round(4))
    fig = px.bar(por_canal.reset_index(), x="origin", y="tasa_90d",
                 hover_data=["leads", "cierres_90d"],
                 labels={"origin": "Canal", "tasa_90d": "Tasa de conversión"})
    c2.plotly_chart(fig)

    st.subheader("Conversión por landing page")
    por_lp = (df.groupby("lp_grupo")
                .agg(leads=("mql_id", "count"), tasa_90d=("conv_90", "mean"))
                .sort_values("leads", ascending=False))
    c3, c4 = st.columns([1, 2])
    c3.dataframe(por_lp.round(4))
    fig = px.bar(por_lp.reset_index(), x="lp_grupo", y="tasa_90d",
                 hover_data=["leads"],
                 labels={"lp_grupo": "Landing page", "tasa_90d": "Tasa de conversión"})
    c4.plotly_chart(fig)

    st.subheader("Leads por canal y landing page")
    cruce = pd.crosstab(df["origin"], df["lp_grupo"])
    st.plotly_chart(px.imshow(cruce, aspect="auto", text_auto=True,
                              labels={"x": "Landing page", "y": "Canal", "color": "Leads"}))

    st.subheader("Evolución mensual")
    mensual = df.groupby("mes").agg(leads=("mql_id", "count"), tasa_90d=("conv_90", "mean"))
    c5, c6 = st.columns(2)
    c5.plotly_chart(px.bar(mensual.reset_index(), x="mes", y="leads"))
    c6.plotly_chart(px.line(mensual.reset_index(), x="mes", y="tasa_90d", markers=True))

# ---------------------------------------------------------------- impacto
with tab2:
    st.subheader("Atribución con datos reales: Markov frente a First-touch")
    real = leer("atribucion_real.csv")
    if real is None:
        st.warning("Falta data/atribucion_real.csv (guárdalo al final del notebook 02).")
    else:
        cols_real = [c for c in ["Markov", "First-touch"] if c in real.columns]
        fig = px.bar(largo(real[cols_real]), x="canal", y="credito",
                     color="metodo", barmode="group",
                     labels={"credito": "Parte del crédito"})
        st.plotly_chart(fig)
        st.caption("Con datos reales cada lead tiene un solo canal y una sola landing page. "
                   "Markov aquí aporta una atribución ajustada por landing page.")
        lp_att = leer("atribucion_lp.csv")
    if lp_att is not None:
        st.subheader("Crédito de Markov por landing page")
        lp_att = lp_att.rename_axis("lp").reset_index().sort_values("Markov", ascending=False)
        st.plotly_chart(px.bar(lp_att, x="lp", y="Markov",
                               labels={"lp": "Landing page", "Markov": "Parte del crédito"}))
    st.subheader("Simulación multicanal: métodos frente a la verdad simulada")
    sim = leer("atribucion_simulada.csv")
    mae = leer("resultados_mae.csv")
    if sim is None:
        st.warning("Falta data/atribucion_simulada.csv (guárdalo al final del notebook 04).")
    else:
        fig = px.bar(largo(sim), x="canal", y="credito", color="metodo", barmode="group",
                     labels={"credito": "Parte del crédito"})
        st.plotly_chart(fig)
    if mae is not None:
        st.write("Error medio absoluto frente a la verdad (puntos porcentuales, 10 semillas):")
        st.dataframe(mae.sort_values("mean"))
        st.caption("Los journeys son sintéticos. La 'verdad' es una definición propia "
                   "(riesgo de cada canal repartido proporcionalmente), no un dato observado.")

    st.subheader("Qué explica la conversión (XGBoost + SHAP)")
    img = ROOT / "docs" / "shap_summary.png"
    if img.exists():
        st.image(str(img))
    else:
        st.warning("Falta docs/shap_summary.png (lo genera el notebook 03).")

# ---------------------------------------------------------------- simulador
with tab3:
    st.subheader("Simulador de escenarios")
    prep = preparar_modelo(df)
    if prep is None:
        st.warning("Faltan app/modelo_xgb.json y app/columnas_modelo.json "
                   "(última celda del notebook 03).")
    else:
        modelo, cols, k, p_base = prep
        canales = sorted(df["origin"].unique())
        modo_txt = st.radio("Tipo de escenario",
                            ["Mover leads de un canal a otro", "Cambiar el volumen de leads por canal"],
                            horizontal=True)

        if modo_txt.startswith("Mover"):
            c1, c2, c3 = st.columns(3)
            origen = c1.selectbox("Canal de origen", canales,
                                  index=canales.index("social") if "social" in canales else 0)
            destino = c2.selectbox("Canal de destino", [c for c in canales if c != origen],
                                   index=0)
            pct = c3.slider("% de leads que se mueven", 0, 100, 20) / 100
            esc = armar_escenario(df, p_base, "mover", modelo, cols, k,
                                  origen=origen, destino=destino, pct=pct)
        else:
            st.write("Variación del volumen de leads de cada canal (%):")
            cambios = {}
            col_a, col_b = st.columns(2)
            for i, c in enumerate(canales):
                cambios[c] = (col_a if i % 2 == 0 else col_b).slider(c, -50, 100, 0) / 100
            esc = armar_escenario(df, p_base, "volumen", modelo, cols, k, cambios=cambios)

        base = resumen(armar_escenario(df, p_base, "volumen", modelo, cols, k, cambios={}))
        nuevo = resumen(esc)

        conv_b, conv_n = base["conv"].sum(), nuevo["conv"].sum()
        leads_b, leads_n = base["leads"].sum(), nuevo["leads"].sum()
        m1, m2, m3 = st.columns(3)
        m1.metric("Conversiones esperadas", f"{conv_n:,.0f}",
                  f"{conv_n - conv_b:+,.1f} ({(conv_n / conv_b - 1) * 100:+.1f}%)")
        m2.metric("Leads totales", f"{leads_n:,.0f}", f"{leads_n - leads_b:+,.0f}")
        m3.metric("Tasa de conversión", f"{conv_n / leads_n * 100:.2f}%",
                  f"{(conv_n / leads_n - conv_b / leads_b) * 100:+.2f} pp")

        comp = pd.concat([base.assign(escenario="Base"), nuevo.assign(escenario="Escenario")])
        st.plotly_chart(px.bar(comp.reset_index(), x="origin", y="conv", color="escenario",
                               barmode="group",
                               labels={"origin": "Canal", "conv": "Conversiones esperadas"}))
        tabla = pd.DataFrame({
            "leads base": base["leads"], "leads escenario": nuevo["leads"],
            "conv. base": base["conv"], "conv. escenario": nuevo["conv"]}).round(1)
        st.dataframe(tabla)
        st.caption("Estimación basada en un modelo de asociación, no de causalidad. "
                   "Los leads movidos conservan su landing page y su día de contacto. "
                   "Las probabilidades se reescalan para que el escenario base coincida "
                   "con la tasa observada.")

# ---------------------------------------------------------------- metodología
with tab4:
    st.markdown("""
**Fuente de datos:** Olist Marketing Funnel (leads calificados y negocios cerrados).

**Periodo:** leads de enero a mayo de 2018. Antes de esa fecha los cierres no están registrados de forma completa.

**Limitaciones**
- Cada lead tiene un solo canal y una sola landing page, así que no existen recorridos multicanal reales.
- La comparación entre métodos de atribución usa recorridos simulados, calibrados con tasas reales. Los supuestos de la simulación (número de toques, repetición de canal, tiempos entre toques) no son datos.
- El simulador de escenarios usa un modelo XGBoost entrenado con canal, landing page y día de la semana. Estima asociaciones, no efectos causales.
- En conversión, XGBoost no supera de forma significativa a una regresión logística.

**Metodología:** CRISP-DM (comprensión del negocio y de los datos, preparación, modelado, evaluación y despliegue).
""")