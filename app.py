"""
Dashboard Híbrido del TP Integrador · Ciencia de Datos · IFTS 18
¿La lluvia nos deja en casa? Viajes SUBE en el AMBA y clima.

Este dashboard se alimenta de los archivos exportados por el pipeline oficial del equipo:
  - sube_clima_eventos_maestro.csv (Dataset consolidado 2020-2026)
  - metricas_modelos.csv (Evaluación Test 2026)
  - predicciones_test.csv (Predicciones Test 2026)
  - EDA_outliers_real.png (Imagen del Pre-EDA)
"""
import datetime
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# =============================================================================
# TEXTOS COMPLETADOS CON LOS RESULTADOS DEL MODELADO HÍBRIDO
# =============================================================================
INTEGRANTES = "Cayoja, Dalma - Speranza, Pablo - Vázquez, Rocío - Vivas, Rafael"

INSIGHTS = [
    "Insight 1 (Clima vs Calendario): Eliminando el efecto pandemia, las temperaturas extremas y la lluvia dominan el 62% del comportamiento del volumen diario (Random Forest).",
    "Insight 2 (Impacto Operativo): Un feriado nacional resta en promedio 1.6 millones de viajes, mientras que un paro extrae ~600,000 pasajeros del sistema (Regresión Lineal).",
    "Insight 3 (El efecto Lluvia): Se pierden exactamente 10,027 viajes en el AMBA por cada milímetro de lluvia acumulada en un día hábil."
]
CONCLUSIONES = [
    "Conclusión 1: La lluvia SÍ nos deja en casa. Una tormenta fuerte (>60mm) tiene la capacidad operativa de paralizar la misma cantidad de pasajes que un paro oficial de transporte.",
    "Conclusión 2: Es posible predecir la demanda. El Random Forest entrenado con la historia completa (ASPO incluido) logra el menor margen de error diario (MAE: ~845k viajes en 2026)."
]
DECISION = "Herramienta Gubernamental: Usar las alertas predictivas para prevención de colapsos. Herramienta Operativa: Ajustar flotas y presupuestos basándose en la caída tarifaria por pronóstico de lluvia."
LIMITACIONES = [
    "El clima se mide en un solo punto, no en todo el AMBA.",
    "La lluvia es el total del día: no sabemos si cayó en hora pico.",
    "SUBE cuenta viajes pagados: no incluye caminata, bici, auto ni evasión."
]
ETICA = (
    "Los datos son agregados. Las predicciones tienen margen de error y no deben usarse para "
    "reducir servicios sin considerar el impacto en los ciudadanos más vulnerables."
)

# =============================================================================
# Configuración Visual
# =============================================================================
st.set_page_config(page_title="¿La lluvia nos deja en casa?", page_icon="🌧️", layout="wide")

CARPETA = Path(__file__).parent
DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
TIPOS_DIA = ["Hábil", "Sábado", "Domingo o feriado"]
NIVELES_LLUVIA = ["Sin lluvia", "Leve", "Fuerte"]
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
COLOR_TIPO = {"Hábil": "#2E6FBA", "Sábado": "#E08A2E", "Domingo o feriado": "#3A9E6E"}
COLOR_LLUVIA = {"Sin lluvia": "#9AA5B1", "Leve": "#5B9BD5", "Fuerte": "#1F4E8C"}
AZUL, GRIS, ROJO = "#2E6FBA", "#9AA5B1", "#C8463D"

# =============================================================================
# Funciones Auxiliares
# =============================================================================
def es(texto):
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")

def millones(x):
    return es(f"{x / 1e6:,.2f} M")

def pct(x, signo=True):
    return es(f"{x:+.1f} %" if signo else f"{x:.1f} %")

def estilo(fig, alto=380):
    fig.update_layout(height=alto, margin=dict(l=10, r=10, t=50, b=10), separators=",.",
                      legend=dict(orientation="h", y=-0.2, title=None),
                      hoverlabel=dict(bgcolor="white"), font=dict(size=13))
    return fig

# =============================================================================
# Carga de Datos Adaptada a Nuestro Maestro
# =============================================================================
def leer(nombre):
    ruta = CARPETA / nombre
    if not ruta.exists():
        st.error(f"No encuentro **{nombre}**. Asegúrate de que está en la misma carpeta que `app.py`.")
        st.stop()
    return pd.read_csv(ruta)

@st.cache_data
def cargar():
    # Leemos NUESTRO archivo maestro
    df = leer("sube_clima_eventos_maestro.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df.rename(columns={'total_amba': 'viajes'}, inplace=True)
    
    # Fabricamos las columnas que el Dashboard necesita para pintar los gráficos
    df['anio'] = df['fecha'].dt.year
    df['mes'] = df['fecha'].dt.month
    df['dia_semana'] = df['fecha'].dt.dayofweek.map({
    0: 'Lunes',
    1: 'Martes',
    2: 'Miércoles',
    3: 'Jueves',
    4: 'Viernes',
    5: 'Sábado',
    6: 'Domingo'
})
    
    # Tipo de día
    es_domingo = df['fecha'].dt.dayofweek == 6
    es_sabado = df['fecha'].dt.dayofweek == 5
    es_feriado = (df['es_feriado_nacional'] == 1) | (df['es_feriado_puente'] == 1)
    df['tipo_dia'] = np.where(es_domingo | es_feriado, 'Domingo o feriado',
                              np.where(es_sabado, 'Sábado', 'Hábil'))
    df["tipo_dia"] = pd.Categorical(df["tipo_dia"], TIPOS_DIA, ordered=True)
    
    # Nivel de lluvia
    df['llovio'] = np.where(df['precipitation_sum'] >= 1, 1, 0)
    df['nivel_lluvia'] = np.where(df['precipitation_sum'] >= 10, 'Fuerte',
                                  np.where(df['precipitation_sum'] >= 1, 'Leve', 'Sin lluvia'))
    df["nivel_lluvia"] = pd.Categorical(df["nivel_lluvia"], NIVELES_LLUVIA, ordered=True)

    metricas = leer("metricas_modelos.csv")
    pred = leer("predicciones_test.csv")
    pred["fecha"] = pd.to_datetime(pred["fecha"])
    
    return df, metricas, pred

df, metricas, pred = cargar()

# =============================================================================
# Funciones para el Simulador
# =============================================================================
MEJOR = metricas.sort_values("MAE_test")["modelo"].iloc[0]

# Features que usamos en nuestro modelo ganador
FEATURES_MODELO = ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 
                   'es_feriado_nacional', 'es_feriado_puente', 'es_paro', 
                   'es_inhabil_acordada', 'es_dia_no_laborable', 'es_aspo']

@st.cache_resource
def entrenar(nombre):
    # Re-entrenamos el modelo rápidamente para el simulador usando nuestro maestro
    modelos = {
        "Regresión Lineal": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }
    datos = df.dropna(subset=FEATURES_MODELO + ["viajes"])
    X = datos[FEATURES_MODELO]
    y = datos["viajes"]
    modelo = modelos.get(nombre, modelos["Random Forest"]).fit(X, y)
    return modelo, list(X.columns)

# =============================================================================
# Encabezado y Filtros
# =============================================================================
st.title("🌧️ ¿La lluvia nos deja en casa?")
st.markdown(
    f"**Modelado Híbrido: Viajes en transporte público del AMBA (SUBE) y Clima.** "
    f"TP Integrador de Ciencia de Datos · IFTS 18 · {INTEGRANTES}"
)

with st.sidebar:
    st.header("Filtros")
    años = sorted(df["anio"].unique())
    f_años = st.multiselect("Año", años, default=años)
    f_meses = st.select_slider("Meses", options=list(range(1, 13)), value=(1, 12),
                               format_func=lambda m: MESES[m - 1])
    f_tipos = st.multiselect("Tipo de día", TIPOS_DIA, default=TIPOS_DIA)
    st.caption("Los filtros afectan el Resumen, Clima y Calendario.")

d = df[df["anio"].isin(f_años) & df["mes"].between(*f_meses) & df["tipo_dia"].isin(f_tipos)]
if d.empty:
    st.warning("No hay días con esa combinación de filtros.")
    st.stop()
hab = d[d["tipo_dia"] == "Hábil"]

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Resumen", "🔍 Pre-EDA (Anomalías)", "🌧️ Clima y Viajes", "📅 Calendario", "🤖 Modelo Predictivo", "💡 Conclusiones"])

# =============================================================================
# 1. RESUMEN
# =============================================================================
with tab1:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Viajes típicos (Mediana)", millones(d["viajes"].median()))
    record = d.loc[d["viajes"].idxmax()]
    k2.metric("Día récord", millones(record["viajes"]), help=f"{record['dia_semana']} {record['fecha']:%d/%m/%Y}")
    k3.metric("Días con lluvia", f"{d['llovio'].mean() * 100:.0f} %")
    
    if hab["llovio"].nunique() == 2:
        caida = hab[hab["llovio"] == 1]["viajes"].median() / hab[hab["llovio"] == 0]["viajes"].median() - 1
        k4.metric("Caída Demanda Lluvia Fuerte", pct(caida * 100))
    else:
        k4.metric("Caída Demanda Lluvia", "—")

    serie = d.sort_values("fecha")
    fig = go.Figure()
    for tipo in [t for t in TIPOS_DIA if t in f_tipos]:
        s = serie[serie["tipo_dia"] == tipo]
        fig.add_scatter(x=s["fecha"], y=s["viajes"] / 1e6, mode="markers", name=tipo,
                        marker=dict(size=5, color=COLOR_TIPO[tipo], opacity=0.55))
    fig.update_layout(title="Serie Temporal de Viajes: Identificación de niveles según Tipo de Día", yaxis_title="Millones de viajes")
    st.plotly_chart(estilo(fig, 420), use_container_width=True)

# =============================================================================
# 2. PRE-EDA Y ANOMALÍAS
# =============================================================================
with tab2:
    st.header("Análisis Exploratorio y Detección de Anomalías (Outliers)")
    st.markdown("Previo al entrenamiento de los algoritmos de Machine Learning, fue indispensable comprender la estructura y calidad de la base histórica de datos del AMBA (2020-2026).")
    
    ruta_img = CARPETA / "EDA_outliers_real.png"
    if ruta_img.exists():
        imagen = Image.open(ruta_img)
        st.image(imagen, caption="Identificación de anomalías de calendario y el impacto estructural del ASPO.", use_container_width=True)
    else:
        st.warning("Sube el archivo 'EDA_outliers_real.png' a la misma carpeta del script para visualizar el gráfico del Pre-EDA.")
        
    st.markdown("### Tratamiento Metodológico")
    st.markdown("""
    * **Faltantes e Imputación:** No se aplicó imputación automática a los vacíos, ya que coincidían con eventos extraordinarios como paros de transporte. 
    * **Impacto del ASPO:** La pandemia representó un cambio estructural masivo, no un simple valor atípico. Para aislar matemáticamente este fenómeno, se estructuró un **entrenamiento en paralelo** (Con ASPO vs Sin ASPO).
    """)

# =============================================================================
# 3. CLIMA Y VIAJES
# =============================================================================
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        if not hab.empty:
            fig = px.box(hab, x="nivel_lluvia", y=hab["viajes"] / 1e6, color="nivel_lluvia",
                         category_orders={"nivel_lluvia": NIVELES_LLUVIA}, color_discrete_map=COLOR_LLUVIA,
                         points="outliers", labels={"y": "Millones de viajes", "nivel_lluvia": ""})
            fig.update_layout(title="Impacto del volumen de lluvia en días hábiles")
            st.plotly_chart(estilo(fig), use_container_width=True)
    with c2:
        # Gráfico adaptado para mostrar la relación de viajes totales por lluvia
        if hab["llovio"].nunique() == 2:
            resumen_lluvia = hab.groupby('nivel_lluvia', observed=True)['viajes'].median().reset_index()
            fig = px.bar(resumen_lluvia, x="nivel_lluvia", y=resumen_lluvia["viajes"] / 1e6,
                         labels={"y": "Millones de viajes", "nivel_lluvia": ""},
                         text=(resumen_lluvia["viajes"] / 1e6).map(lambda v: es(f"{v:.2f} M")))
            fig.update_traces(marker_color=[GRIS, AZUL, ROJO], textposition="outside")
            fig.update_layout(title="Mediana de viajes totales según intensidad de lluvia")
            st.plotly_chart(estilo(fig), use_container_width=True)

# =============================================================================
# 4. CALENDARIO
# =============================================================================
with tab4:
    sin_fer = d[d["es_feriado_nacional"] == 0]
    por_dia = sin_fer.groupby("dia_semana")["viajes"].mean().reindex(DIAS).dropna()
    c1, c2 = st.columns(2)
    with c1:
        if len(por_dia) > 0:
            pdia = por_dia.rename("viajes").reset_index()
            fig = px.bar(pdia, x="dia_semana", y=pdia["viajes"] / 1e6, labels={"y": "Millones de viajes", "dia_semana": ""},
                         text=(pdia["viajes"] / 1e6).map(lambda v: es(f"{v:.2f}")))
            fig.update_traces(marker_color=[AZUL if d_ in DIAS[:5] else GRIS for d_ in pdia["dia_semana"]], textposition="outside")
            fig.update_layout(title="Promedio estructural de viajes por día")
            st.plotly_chart(estilo(fig), use_container_width=True)
    with c2:
        mapa = hab.pivot_table(index="anio", columns="mes", values="viajes", aggfunc="mean") / 1e6
        if not mapa.empty:
            fig = px.imshow(mapa, color_continuous_scale="Blues", aspect="auto", text_auto=".2f")
            fig.update_layout(title="Evolución mensual (Días hábiles)")
            st.plotly_chart(estilo(fig), use_container_width=True)

# =============================================================================
# 5. MODELO PREDICTIVO
# =============================================================================
with tab5:
    st.markdown(f"**Estrategia Híbrida:** Se entrenaron modelos paramétricos (Regresión Lineal) y de ensamble (Random Forest). El Campeón Predictivo para despliegue gubernamental es **{MEJOR}**.")
    
    tabla = metricas.copy()
    tabla["MAE (viajes)"] = tabla["MAE_test"].map(lambda v: es(f"{v:,.0f}"))
    tabla["RMSE (viajes)"] = tabla["RMSE_test"].map(lambda v: es(f"{v:,.0f}"))
    tabla["Elegido"] = np.where(tabla["modelo"] == MEJOR, "🏆", "")
    st.dataframe(tabla[["Elegido", "modelo", "MAE (viajes)", "RMSE (viajes)"]], hide_index=True, use_container_width=True)

    col_pred = f"pred_Random Forest"
    if col_pred in pred.columns:
        pt = pred.assign(real=pred["viajes"] / 1e6, predicho=pred[col_pred] / 1e6)
        lim = [pt[["real", "predicho"]].min().min() * 0.95, pt[["real", "predicho"]].max().max() * 1.03]
        fig = px.scatter(pt, x="real", y="predicho", color="tipo_dia", color_discrete_map=COLOR_TIPO)
        fig.add_shape(type="line", x0=lim[0], y0=lim[0], x1=lim[1], y1=lim[1], line=dict(color="#555", dash="dash"))
        fig.update_layout(title="Evaluación Test 2026: Predicción vs Realidad", xaxis_title="Viajes Reales (M)", yaxis_title="Predicción (M)")
        st.plotly_chart(estilo(fig, 400), use_container_width=True)

    st.subheader("🔮 Simulador de Negocio (Gobierno y Operaciones)")
    modelo, columnas = entrenar(MEJOR)
    mae = metricas.set_index("modelo").loc[MEJOR, "MAE_test"]

    s1, s2, s3 = st.columns(3)
    fecha = s1.date_input("Día a simular", value=datetime.date.today() + datetime.timedelta(days=1), format="DD/MM/YYYY")
    lluvia = s2.slider("Lluvia del día (mm)", 0.0, 80.0, 0.0, 0.5)
    tmax = s3.slider("Temperatura máxima (°C)", -2, 40, 24)
    s4, s5, _ = st.columns(3)
    tmin = s4.slider("Temperatura mínima (°C)", -5, 30, min(15, tmax))

    fer = holidays.country_holidays("AR", years=fecha.year)
    es_feriado_n = 1 if fecha in fer else 0
    
    nuevo = pd.DataFrame([{
        "temperature_2m_max": tmax,
        "temperature_2m_min": tmin,
        "precipitation_sum": lluvia,
        "es_feriado_nacional": es_feriado_n,
        "es_feriado_puente": 0,
        "es_paro": 0,
        "es_inhabil_acordada": 0,
        "es_dia_no_laborable": 0,
        "es_aspo": 0
    }])
    estimado = float(modelo.predict(nuevo[columnas])[0])
    
    nuevo_seco = nuevo.copy()
    nuevo_seco["precipitation_sum"] = 0.0
    seco = float(modelo.predict(nuevo_seco[columnas])[0])

    r1, r2, r3 = st.columns(3)
    r1.metric("Volumen Estimado (SUBE)", millones(estimado))
    r2.metric("Margen de error (MAE)", f"± {millones(mae)}")
    r3.metric("Impacto Neto de la Lluvia", pct((estimado / seco - 1) * 100))

# =============================================================================
# 6. CONCLUSIONES
# =============================================================================
with tab6:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("💡 Insights del Modelado Híbrido")
        for i in INSIGHTS:
            st.markdown(f"- {i}")
        st.subheader("✅ Conclusiones")
        for c in CONCLUSIONES:
            st.markdown(f"- {c}")
        st.subheader("🎯 Aplicación de Negocio")
        st.markdown(DECISION)
    with c2:
        st.subheader("⚠️ Limitaciones Estructurales")
        for lim in LIMITACIONES:
            st.markdown(f"- {lim}")
        st.subheader("🛡️ Ética Analítica")
        st.markdown(ETICA)
