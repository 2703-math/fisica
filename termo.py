"""
🔥 Termofísica & Trocas de Calor Visual
========================================
Executar com: streamlit run termofisica_app.py

Seis módulos didáticos e interativos:
  1. Mecanismos de Troca de Calor (Condução, Convecção e Radiação)
  2. Fluxo de Calor – Lei de Fourier e Gradientes Térmicos
  3. Perda de Calor – Lei de Resfriamento de Newton
  4. Dilatação Térmica – Linear, Superficial e Volumétrica
  5. Calorimetria – Calor Sensível, Latente e Curva de Aquecimento
  6. Equilíbrio Térmico – Troca de Calor em Sistemas Isolados
"""

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA E ESTILOS CSS
# ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Termofísica & Trocas de Calor Visual",
    page_icon="🔥",
    layout="wide"
)

st.markdown("""
<style>
.main-title { font-size: 2.2rem; font-weight: 800; color: #0f172a; text-align: center; margin-bottom: .3rem; }
.subtitle { font-size: 1.05rem; color: #64748b; text-align: center; margin-bottom: 1.4rem; }
.card { background: #f8fafc; border-radius: 12px; padding: 1.1rem 1.3rem; border-left: 4px solid #3b82f6; margin-bottom: .9rem; color: #334155; line-height: 1.6; }
.form { background: #f0fdf4; border-radius: 10px; padding: .7rem 1.2rem; border-left: 4px solid #10b981; margin: .5rem 0; color: #065f46; font-size: 1.05rem; }
.warn { background: #fffbeb; border-radius: 10px; padding: .7rem 1.2rem; border-left: 4px solid #f59e0b; margin: .5rem 0; color: #78350f; }
.highlight { background: #eff6ff; border-radius: 8px; padding: .8rem; border: 1px solid #bfdbfe; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# UTILITÁRIOS
# ────────────────────────────────────────────────────────────
def mostrar(fig, h=460):
    fig.update_layout(
        height=h,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family="sans-serif", size=13)
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def formula(latex_str):
    st.latex(latex_str)

def quiz(chave, enunciado, opcoes, correta, explicacao):
    with st.expander("🎯 Teste sua intuição"):
        r = st.radio(enunciado, opcoes, index=None, key=f"q_{chave}")
        if r is not None:
            if opcoes.index(r) == correta:
                st.success(f"✅ {explicacao}")
            else:
                st.warning("🤔 Ainda não — observe o gráfico e os parâmetros e tente novamente.")

# ────────────────────────────────────────────────────────────
# BANCO DE DADOS DE MATERIAIS
# ────────────────────────────────────────────────────────────
MATERIAIS_COND = {
    "Cobre": {"k": 385.0, "cor": "#b87333"},
    "Alumínio": {"k": 205.0, "cor": "#94a3b8"},
    "Ferro / Aço": {"k": 50.0, "cor": "#475569"},
    "Vidro Comum": {"k": 0.8, "cor": "#38bdf8"},
    "Tijolo Cerâmico": {"k": 0.7, "cor": "#ea580c"},
    "Madeira": {"k": 0.13, "cor": "#a16207"},
    "Isopor (EPS)": {"k": 0.03, "cor": "#cbd5e1"},
}

MATERIAIS_DILAT = {
    "Alumínio": {"alpha": 2.3e-5, "cor": "#94a3b8"},
    "Latão": {"alpha": 1.9e-5, "cor": "#eab308"},
    "Cobre": {"alpha": 1.7e-5, "cor": "#b87333"},
    "Aço / Ferro": {"alpha": 1.2e-5, "cor": "#475569"},
    "Vidro Comum": {"alpha": 0.9e-5, "cor": "#38bdf8"},
    "Vidro Pyrex": {"alpha": 0.3e-5, "cor": "#0284c7"},
}

# ────────────────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configurações Gerais")
    mostrar_formulas = st.checkbox("Mostrar equações e deduções", value=True)
    st.markdown("---")
    st.subheader("🧑‍🏫 Roteiro de Estudo")
    st.markdown(
        "**1. Mecanismos:** Entenda como o calor se propaga nos 3 meios.\n\n"
        "**2. Fluxo (Fourier):** Simule a condução através de materiais térmicos.\n\n"
        "**3. Resfriamento:** Observe o decaimento exponencial de Newton.\n\n"
        "**4. Dilatação:** Veja os corpos se expandirem com a temperatura.\n\n"
        "**5. Calorimetria:** Acompanhe os patamares de mudança de fase.\n\n"
        "**6. Equilíbrio:** Calcule a temperatura final após a troca de energia."
    )
    st.markdown("---")
    st.caption("Desenvolvido para Ensino Médio e Superior")

# ────────────────────────────────────────────────────────────
# TÍTULO PRINCIPAL
# ────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔥 Termofísica & Trocas de Calor Visual</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Propagação térmica, lei de Fourier, resfriamento de Newton, dilatação e calorimetria</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "1. Tipos de Troca",
    "2. Fluxo (Fourier)",
    "3. Resfriamento (Newton)",
    "4. Dilatação Térmica",
    "5. Calorimetria & Fases",
    "6. Equilíbrio Térmico"
])

# ══════════════════════════════════════════════════════════════
# ABA 1 — TIPOS DE TROCA DE CALOR
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("""<div class="card">
    O <b>calor</b> é energia térmica em trânsito motivada por uma diferença de temperatura.
    Existem três mecanismos fundamentais para a sua propagação: <b>Condução</b>, <b>Convecção</b> e <b>Radiação</b>.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2.2])
    with c1:
        st.markdown("### 📌 Comparativo dos Mecanismos")
        mecanismo = st.radio("Selecione para explorar:", [
            "Condução Térmica",
            "Convecção Térmica",
            "Radiação Térmica (Irradiação)"
        ], key="mec_radio")

        if mecanismo == "Condução Térmica":
            st.info("""
            **Condução:** Ocorre principalmente nos **sólidos**. A energia passa molécula a molécula por vibração atômica sem transporte de matéria.
            *   **Exemplo:** Colher de metal em sopa quente.
            *   **Meio necessário:** Requer meio material.
            """)
        elif mecanismo == "Convecção Térmica":
            st.info("""
            **Convecção:** Ocorre em **fluidos** (líquidos e gases). A matéria aquecida se dilata, fica menos densa e sobe, criando **correntes de convecção**.
            *   **Exemplo:** Ar-condicionado no alto, congelador da geladeira.
            *   **Meio necessário:** Requer meio fluido com gravidade.
            """)
        else:
            st.info("""
            **Radiação Térmica:** Propagação por **ondas eletromagnéticas** (principalmente infravermelho).
            *   **Exemplo:** Calor do Sol chegando à Terra, estufa de plantas.
            *   **Meio necessário:** Ocorre no **vácuo** e em meios transparentes.
            """)

        if mostrar_formulas:
            st.markdown('<div class="form">', unsafe_allow_html=True)
            if mecanismo == "Condução Térmica":
                formula(r"\Phi = \frac{Q}{\Delta t} = \frac{k \cdot A \cdot \Delta T}{L}")
            elif mecanismo == "Convecção Térmica":
                formula(r"\Phi = h \cdot A \cdot (T_{\text{superfície}} - T_{\text{fluido}})")
            else:
                formula(r"P = \varepsilon \cdot \sigma \cdot A \cdot T^4 \quad (\text{Lei de Stefan-Boltzmann})")
            st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        # Gráfico comparativo de condutividade térmica de materiais cotidianos
        mat_names = list(MATERIAIS_COND.keys())
        mat_ks = [MATERIAIS_COND[m]["k"] for m in mat_names]
        mat_cols = [MATERIAIS_COND[m]["cor"] for m in mat_names]

        fig_mec = go.Figure(go.Bar(
            x=mat_ks,
            y=mat_names,
            orientation="h",
            marker_color=mat_cols,
            text=[f"{k} W/(m·K)" for k in mat_ks],
            textposition="outside"
        ))
        fig_mec.update_xaxes(type="log", title="Condutividade Térmica k em escala logarítmica (W/m·K)")
        fig_mec.update_yaxes(autorange="reversed")
        fig_mec.update_layout(title="Comparativo de Condutividade Térmica (Isolantes vs Condutores)")
        mostrar(fig_mec, 420)

    quiz("mec1", "Por que as garrafas térmicas possuem paredes duplas de vidro espelhado com vácuo entre elas?",
         [
             "Para evitar a condução e a convecção pelo vácuo, e a radiação pelas paredes espelhadas.",
             "Apenas para proteger o vidro de quebrar.",
             "Para evitar somente a convecção do líquido.",
             "O vácuo impede a radiação térmica de sair."
         ], 0,
         "O vácuo elimina a condução e a convecção (não há matéria), e o espelhamento reflete as ondas térmicas, bloqueando a radiação!")

# ══════════════════════════════════════════════════════════════
# ABA 2 — FLUXO DE CALOR (LEI DE FOURIER)
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("""<div class="card">
    A <b>Lei de Fourier</b> quantifica a taxa de transferência de calor por condução em regime estacionário através de uma parede ou condutor.<br>
    O fluxo de calor <b>Φ = Q/Δt</b> é diretamente proporcional à condutividade térmica <b>k</b>, à área <b>A</b> e à diferença de temperatura <b>ΔT</b>, e inversamente proporcional à espessura <b>L</b>.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2.3])
    with c1:
        with st.container(border=True):
            mat_sel = st.selectbox("Material do condutor/parede", list(MATERIAIS_COND.keys()), index=1, key="fourier_mat")
            k_val = MATERIAIS_COND[mat_sel]["k"]

            t_quente = st.slider("Temperatura da face quente T₁ (°C)", 20.0, 300.0, 100.0, 5.0, key="fourier_t1")
            t_fria = st.slider("Temperatura da face fria T₂ (°C)", -20.0, 100.0, 20.0, 5.0, key="fourier_t2")

            if t_fria >= t_quente:
                st.warning("⚠️ T₁ deve ser maior que T₂ para haver fluxo do quente para o frio.")

            area_m2 = st.slider("Área transversal A (m²)", 0.1, 10.0, 2.0, 0.1, key="fourier_a")
            espessura_cm = st.slider("Espessura L (cm)", 0.5, 50.0, 10.0, 0.5, key="fourier_l")

        espessura_m = espessura_cm / 100.0
        delta_t = t_quente - t_fria
        fluxo_watts = (k_val * area_m2 * delta_t) / espessura_m if espessura_m > 0 else 0

        st.metric("Fluxo de Calor (Φ = Q/Δt)", f"{fluxo_watts:,.2f} W".replace(",", "X").replace(".", ",").replace("X", "."))
        st.caption(f"Em 1 hora, essa parede transfere **{(fluxo_watts * 3600 / 1e6):.2f} MJ** de energia.")

        if mostrar_formulas:
            st.markdown('<div class="form">', unsafe_allow_html=True)
            formula(r"\Phi = \frac{Q}{\Delta t} = \frac{k \cdot A \cdot (T_1 - T_2)}{L}")
            formula(rf"\Phi = \frac{{{k_val:.2f} \cdot {area_m2:.1f} \cdot ({t_quente:.1f} - {t_fria:.1f})}}{{{espessura_m:.3f}}} = {fluxo_watts:.2f}\text{{ W}}")
            st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        # Gráfico do perfil linear de temperatura no interior do condutor
        x_pts = np.linspace(0, espessura_cm, 200)
        t_pts = t_quente - (t_quente - t_fria) * (x_pts / espessura_cm)

        fig_four = go.Figure()

        # Preenchimento visual da parede
        fig_four.add_trace(go.Scatter(
            x=x_pts, y=t_pts, mode="lines", name="Perfil de Temperatura",
            line=dict(color="#ef4444", width=3.5),
            hovertemplate="Posição: %{x:.2f} cm<br>Temp: %{y:.1f} °C<extra></extra>"
        ))

        fig_four.add_hline(y=t_quente, line=dict(color="#f97316", dash="dot"), annotation_text=f"T₁ = {t_quente}°C")
        fig_four.add_hline(y=t_fria, line=dict(color="#06b6d4", dash="dot"), annotation_text=f"T₂ = {t_fria}°C")

        fig_four.update_xaxes(title="Posição ao longo da espessura da parede (cm)")
        fig_four.update_yaxes(title="Temperatura (°C)")
        fig_four.update_layout(title=f"Gradiente Térmico Linear no {mat_sel} (L = {espessura_cm} cm)")
        mostrar(fig_four, 440)

    quiz("four1", "Se dobrarmos a espessura de uma parede isolante (L) e dobrarmos a sua área (A), o fluxo de calor Φ irá:",
         ["Quadruplicar", "Dobrar", "Ficar inalterado", "Reduzir pela metade"], 2,
         "Como Φ é proporcional a A/L, se multiplicamos o numerador por 2 e o denominador por 2, o resultado permanece exatamente o mesmo!")

# ══════════════════════════════════════════════════════════════
# ABA 3 — PERDA DE CALOR (LEI DE RESFRIAMENTO DE NEWTON)
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("""<div class="card">
    A <b>Lei de Resfriamento de Newton</b> estabelece que a taxa de variação da temperatura de um corpo é proporcional à diferença de temperatura entre o corpo e o meio ambiente.<br>
    A temperatura evolui exponencialmente segundo a função: <b>T(t) = T_amb + (T₀ - T_amb) · e⁻ᵏᵗ</b>.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2.3])
    with c1:
        with st.container(border=True):
            t_ini = st.slider("Temperatura inicial do corpo T₀ (°C)", 30.0, 100.0, 90.0, 1.0, key="newton_t0")
            t_amb = st.slider("Temperatura ambiente T_amb (°C)", -10.0, 40.0, 25.0, 1.0, key="newton_tamb")
            k_resf = st.slider("Constante de resfriamento k (min⁻¹)", 0.01, 0.20, 0.05, 0.01, key="newton_k")
            tempo_max = st.slider("Tempo total de simulação (min)", 10, 180, 60, 5, key="newton_tmax")

        # Tempo para atingir a metade do resfriamento
        t_meio = (math.log(2) / k_resf) if k_resf > 0 else 0
        st.metric("Meia-vida de resfriamento (t½)", f"{t_meio:.1f} min")

        if mostrar_formulas:
            st.markdown('<div class="form">', unsafe_allow_html=True)
            formula(r"\frac{dT}{dt} = -k \cdot (T - T_{\text{amb}})")
            formula(r"T(t) = T_{\text{amb}} + (T_0 - T_{\text{amb}}) \cdot e^{-k \cdot t}")
            st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        t_arr = np.linspace(0, tempo_max, 300)
        temp_arr = t_amb + (t_ini - t_amb) * np.exp(-k_resf * t_arr)

        # Taxa de resfriamento em tempo real dT/dt
        taxa_arr = -k_resf * (temp_arr - t_amb)

        fig_newt = go.Figure()

        # Curva de Temperatura
        fig_newt.add_trace(go.Scatter(
            x=t_arr, y=temp_arr, mode="lines", name="Temperatura T(t)",
            line=dict(color="#ef4444", width=3),
            hovertemplate="Tempo: %{x:.1f} min<br>Temp: %{y:.2f} °C<extra></extra>"
        ))

        # Asíntota da Temperatura Ambiente
        fig_newt.add_hline(y=t_amb, line=dict(color="#64748b", dash="dash"),
                           annotation_text=f"T_amb = {t_amb}°C", annotation_position="bottom right")

        fig_newt.update_xaxes(title="Tempo t (minutos)")
        fig_newt.update_yaxes(title="Temperatura (°C)")
        fig_newt.update_layout(title="Curva Exponencial de Resfriamento de Newton")
        mostrar(fig_newt, 440)

        # Selecionar um instante t para analisar a taxa instantânea
        t_sel = st.slider("Inspecionar instante t (min)", 0, tempo_max, 10, key="newton_tsel")
        temp_sel = t_amb + (t_ini - t_amb) * math.exp(-k_resf * t_sel)
        taxa_sel = -k_resf * (temp_sel - t_amb)

        st.info(f"⏱️ No instante **t = {t_sel} min**: Temperatura = **{temp_sel:.2f} °C** | Taxa de resfriamento instantânea = **{taxa_sel:.3f} °C/min**")

    quiz("newt1", "Por que um xícara de café bem quente esfria muito mais rápido no primeiro minuto do que quando já está morna?",
         [
             "Porque a diferença de temperatura (T - T_amb) é maior no início, gerando maior taxa de variação.",
             "Porque o café perde massa por evaporação no início.",
             "Porque a constante k muda com o tempo.",
             "Porque o ar ambiente esquenta ao redor do café."
         ], 0,
         "Pela Lei de Newton, a taxa dT/dt é proporcional à diferença de temperatura (T - T_amb). Quanto mais quente o corpo está em relação ao meio, mais rápida é a perda de calor!")

# ══════════════════════════════════════════════════════════════
# ABA 4 — DILATAÇÃO TÉRMICA
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("""<div class="card">
    A <b>Dilatação Térmica</b> ocorre porque o aumento de temperatura intensifica a vibração dos átomos, aumentando a distância média entre eles.<br>
    • <b>Linear:</b> ΔL = L₀ · α · ΔT<br>
    • <b>Superficial:</b> ΔA = A₀ · β · ΔT \t (com β ≈ 2α)<br>
    • <b>Volumétrica:</b> ΔV = V₀ · γ · ΔT \t (com γ ≈ 3α)
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2.3])
    with c1:
        with st.container(border=True):
            mat_dil = st.selectbox("Selecione o Material", list(MATERIAIS_DILAT.keys()), index=0, key="dil_mat")
            alpha_val = MATERIAIS_DILAT[mat_dil]["alpha"]

            l0_m = st.slider("Comprimento inicial L₀ (m)", 1.0, 100.0, 10.0, 1.0, key="dil_l0")
            t_inicial = st.slider("Temperatura inicial T₀ (°C)", -20.0, 50.0, 20.0, 1.0, key="dil_t0")
            t_final = st.slider("Temperatura final T_f (°C)", -20.0, 500.0, 120.0, 5.0, key="dil_tf")

        delta_temp = t_final - t_inicial
        delta_l_mm = l0_m * alpha_val * delta_temp * 1000.0  # em mm
        l_final_m = l0_m + (delta_l_mm / 1000.0)

        st.metric("Variação de Comprimento (ΔL)", f"{delta_l_mm:.2f} mm")
        st.metric("Comprimento Final (L)", f"{l_final_m:.5f} m")

        if mostrar_formulas:
            st.markdown('<div class="form">', unsafe_allow_html=True)
            formula(r"\Delta L = L_0 \cdot \alpha \cdot \Delta T")
            formula(rf"\Delta L = {l0_m:.1f} \cdot ({alpha_val:.2e}) \cdot ({delta_temp:.1f}) = {delta_l_mm/1000.0:.6f}\text{{ m}}")
            st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        # Gráfico comparativo de expansão dos materiais para a mesma condição
        temps_grau = np.linspace(t_inicial, t_final, 200)

        fig_dil = go.Figure()

        for m_nome, m_dados in MATERIAIS_DILAT.items():
            a_v = m_dados["alpha"]
            dl_mm = l0_m * a_v * (temps_grau - t_inicial) * 1000.0
            esp = 4 if m_nome == mat_dil else 1.8

            fig_dil.add_trace(go.Scatter(
                x=temps_grau, y=dl_mm, mode="lines", name=m_nome,
                line=dict(color=m_dados["cor"], width=esp),
                hovertemplate=f"<b>{m_nome}</b><br>Temp: %{{x:.1f}} °C<br>ΔL: %{{y:.2f}} mm<extra></extra>"
            ))

        fig_dil.update_xaxes(title="Temperatura (°C)")
        fig_dil.update_yaxes(title="Dilatação Linear ΔL (mm)")
        fig_dil.update_layout(title=f"Comparativo de Dilatação Térmica para Barra de L₀ = {l0_m} m")
        mostrar(fig_dil, 440)

    quiz("dil1", "Lâminas bimetálicas (feitas com dois metais de coeficientes α diferentes colados) são usadas em termostatos porque:",
         [
             "Um metal esquenta mais que o outro.",
             "Como os metais se dilatam em taxas diferentes, a lâmina encurva ao ser aquecida, abrindo ou fechando um circuito.",
             "Um dos metais não se dilata.",
             "A lâmina bimetálica impede o fluxo de eletricidade."
         ], 1,
         "Como os coeficientes α são diferentes, o metal que dilata mais força a lâmina a encurvar para o lado do metal que dilata menos, servindo como interruptor térmico!")

# ══════════════════════════════════════════════════════════════
# ABA 5 — CALORIMETRIA & MUDANÇAS DE FASE
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("""<div class="card">
    A <b>Calorimetria</b> estuda a troca de energia entre corpos:<br>
    • <b>Calor Sensível (Q = m · c · ΔT):</b> Provoca variação de temperatura sem mudar o estado físico.<br>
    • <b>Calor Latente (Q = m · L):</b> Ocorre durante as mudanças de fase, mantendo a temperatura constante durante a transição.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2.3])
    with c1:
        st.markdown("### 🧊 Curva de Aquecimento da Água")
        with st.container(border=True):
            massa_g = st.slider("Massa de água/gelo m (g)", 10.0, 1000.0, 100.0, 10.0, key="cal_m")
            t_ini_agua = st.slider("Temperatura inicial (°C)", -40.0, -1.0, -20.0, 1.0, key="cal_t0")
            t_fim_agua = st.slider("Temperatura final (°C)", 101.0, 150.0, 120.0, 1.0, key="cal_tf")

        # Constantes da água (em cal/g e cal/g·°C)
        c_gelo, L_fusao, c_liq, L_vap, c_vap = 0.50, 80.0, 1.00, 540.0, 0.48

        q1 = massa_g * c_gelo * (0.0 - t_ini_agua)  # Aquecimento do gelo
        q2 = massa_g * L_fusao                      # Fusão do gelo
        q3 = massa_g * c_liq * (100.0 - 0.0)         # Aquecimento da água
        q4 = massa_g * L_vap                        # Vaporização da água
        q5 = massa_g * c_vap * (t_fim_agua - 100.0) # Aquecimento do vapor

        q_total = q1 + q2 + q3 + q4 + q5

        st.metric("Calor Total Necessário", f"{q_total:,.1f} cal".replace(",", "X").replace(".", ",").replace("X", "."))
        st.caption(f"Equivalente a **{(q_total * 4.184 / 1000):.2f} kJ** de energia.")

        if mostrar_formulas:
            st.markdown('<div class="form">', unsafe_allow_html=True)
            formula(r"Q_{\text{sensível}} = m \cdot c \cdot \Delta T")
            formula(r"Q_{\text{latente}} = m \cdot L")
            st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        # Construção dos patamares da curva de aquecimento
        q_eixo = [0, q1, q1 + q2, q1 + q2 + q3, q1 + q2 + q3 + q4, q_total]
        t_eixo = [t_ini_agua, 0.0, 0.0, 100.0, 100.0, t_fim_agua]
        fases = ["1. Gelo (Sensível)", "2. Fusão (Latente)", "3. Líquido (Sensível)", "4. Vaporização (Latente)", "5. Vapor (Sensível)"]

        fig_cal = go.Figure()

        fig_cal.add_trace(go.Scatter(
            x=q_eixo, y=t_eixo, mode="lines+markers",
            line=dict(color="#3b82f6", width=3.5),
            marker=dict(size=8, color="#1e40af"),
            hovertemplate="Calor acumulado: %{x:,.0f} cal<br>Temp: %{y:.1f} °C<extra></extra>"
        ))

        # Linhas guia dos patamares
        fig_cal.add_hline(y=0, line=dict(color="#38bdf8", dash="dot"), annotation_text="Fusão (0°C)")
        fig_cal.add_hline(y=100, line=dict(color="#ef4444", dash="dot"), annotation_text="Ebullição (100°C)")

        fig_cal.update_xaxes(title="Calor Fornecido Q (calorias)")
        fig_cal.update_yaxes(title="Temperatura (°C)")
        fig_cal.update_layout(title=f"Curva de Aquecimento da Água para m = {massa_g} g")
        mostrar(fig_cal, 440)

    quiz("cal1", "Por que a temperatura da água em ebulição permanece exatamente em 100 °C (ao nível do mar) mesmo com a boca do fogão ligada no máximo?",
         [
             "Porque todo o calor fornecido é usado exclusivamente para romper as ligações intermoleculares (Calor Latente).",
             "Porque o fogão para de transferir calor quando atinge 100 °C.",
             "Porque o vapor absorve a temperatura da água.",
             "Porque o calor específico da água fica infinito."
         ], 0,
         "Durante uma mudança de fase substancialmente pura, a energia recebida (calor latente) é gasta na alteração do arranjo molecular, sem aumentar a energia cinética média (temperatura)!")

# ══════════════════════════════════════════════════════════════
# ABA 6 — EQUILÍBRIO TÉRMICO
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("""<div class="card">
    Em um sistema termicamente isolado (como um calorímetro ideal), a soma de todas as quantidades de calor trocadas entre os corpos é nula:<br>
    <b>∑ Q = 0  ⟹  Q_cedido + Q_recebido = 0</b>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2.3])
    with c1:
        st.markdown("### 🧪 Mistura de Duas Substâncias")
        with st.container(border=True):
            st.markdown("**Corpo A (Mais Quente)**")
            mA = st.slider("Massa m_A (g)", 10.0, 500.0, 200.0, 10.0, key="eq_ma")
            cA = st.number_input("Calor específico c_A (cal/g·°C)", value=1.00, step=0.05, key="eq_ca")
            tA = st.slider("Temperatura T_A (°C)", 20.0, 100.0, 80.0, 1.0, key="eq_ta")

            st.markdown("**Corpo B (Mais Frio)**")
            mB = st.slider("Massa m_B (g)", 10.0, 500.0, 100.0, 10.0, key="eq_mb")
            cB = st.number_input("Calor específico c_B (cal/g·°C)", value=0.50, step=0.05, key="eq_cb")
            tB = st.slider("Temperatura T_B (°C)", 0.0, 50.0, 20.0, 1.0, key="eq_tb")

        # Cálculo da Temperatura de Equilíbrio Térmico
        # mA * cA * (Teq - TA) + mB * cB * (Teq - TB) = 0
        cap_A = mA * cA
        cap_B = mB * cB
        teq = (cap_A * tA + cap_B * tB) / (cap_A + cap_B)

        q_trocado = cap_A * (tA - teq)  # Energia cedida por A para B

        st.metric("Temperatura de Equilíbrio (T_eq)", f"{teq:.2f} °C")
        st.metric("Calor Trocado (Q)", f"{q_trocado:,.1f} cal".replace(",", "X").replace(".", ",").replace("X", "."))

        if mostrar_formulas:
            st.markdown('<div class="form">', unsafe_allow_html=True)
            formula(r"m_A \cdot c_A \cdot (T_{\text{eq}} - T_A) + m_B \cdot c_B \cdot (T_{\text{eq}} - T_B) = 0")
            formula(rf"T{{\text{{eq}}}} = \frac{{m_A c_A T_A + m_B c_B T_B}}{{m_A c_A + m_B c_B}} = {teq:.2f}^\circ\text{{C}}")
            st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        # Gráfico visual das Temperaturas Inicial vs Final
        fig_eq = go.Figure()

        fig_eq.add_trace(go.Bar(
            x=["Corpo A (Inicial)", "Corpo B (Inicial)", "Equilíbrio Térmico"],
            y=[tA, tB, teq],
            marker_color=["#ef4444", "#3b82f6", "#10b981"],
            text=[f"{tA:.1f} °C", f"{tB:.1f} °C", f"{teq:.2f} °C"],
            textposition="outside"
        ))

        fig_eq.add_hline(y=teq, line=dict(color="#10b981", dash="dash"),
                         annotation_text=f"T_eq = {teq:.2f}°C", annotation_position="top left")

        fig_eq.update_yaxes(title="Temperatura (°C)")
        fig_eq.update_layout(title="Evolução para o Equilíbrio Térmico")
        mostrar(fig_eq, 440)

        st.info(f"💡 **Capacidade Térmica:** Corpo A = **{cap_A:.1f} cal/°C** | Corpo B = **{cap_B:.1f} cal/°C**. "
                f"O corpo com maior capacidade térmica puxa a temperatura final para mais perto de si!")

    quiz("eq1", "Se misturarmos massas iguais de água a 80 °C e água a 20 °C em um recipiente isolado, a temperatura final será:",
         [
             "Exatamente 50 °C, pois os calores específicos e massas são iguais.",
             "Maior que 50 °C.",
             "Menor que 50 °C.",
             "Depende do volume do recipiente."
         ], 0,
         "Como m_A = m_B e c_A = c_B, a temperatura final é a média aritmética simples: (80 + 20) / 2 = 50 °C!")

# RODAPÉ
st.markdown("---")
st.markdown('<div style="text-align:center;color:#94a3b8;font-size:.85rem;padding:1rem;">'
    '🔥 <b>Termofísica & Trocas de Calor Visual</b> — Fourier · Newton · Dilatação · Calorimetria · Equilíbrio Térmico</div>',
    unsafe_allow_html=True)
