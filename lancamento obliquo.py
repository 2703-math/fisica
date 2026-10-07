import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# ============================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================
st.set_page_config(
    page_title="Física Visual: Lançamento Oblíquo",
    page_icon="🚀",
    layout="wide"
)

# ============================================
# CSS PERSONALIZADO (PADRÃO SAAS)
# ============================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .concept-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
    }
    .param-box {
        background: #fff;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Cores
COR_VACUO = "#3b82f6"
COR_AR = "#ef4444"
COR_PROJETIL = "#10b981"

# ============================================
# NÚCLEO DE SIMULAÇÃO FÍSICA (RK4)
# ============================================
def simular_lancamento(v0, ang_graus, h0, m, b, g=9.8, dt=0.01, max_t=20.0):
    theta = math.radians(ang_graus)
    vx0 = v0 * math.cos(theta)
    vy0 = v0 * math.sin(theta)
    
    # Sem resistência (Vácuo)
    t_voo_vacuo = (vy0 + math.sqrt(max(0.0, vy0**2 + 2 * g * h0))) / g
    ts_vacuo = np.linspace(0, t_voo_vacuo, 100)
    xs_vacuo = vx0 * ts_vacuo
    ys_vacuo = h0 + vy0 * ts_vacuo - 0.5 * g * (ts_vacuo**2)
    
    # Com resistência do ar (RK4)
    def deriv(state):
        x, y, vx, vy = state
        v = math.sqrt(vx**2 + vy**2)
        ax = -(b / m) * v * vx
        ay = -g - (b / m) * v * vy
        return np.array([vx, vy, ax, ay])
    
    state = np.array([0.0, h0, vx0, vy0])
    xs_ar, ys_ar, ts_ar = [0.0], [h0], [0.0]
    
    for _ in range(int(max_t / dt)):
        if state[1] < 0:
            break
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        xs_ar.append(state[0])
        ys_ar.append(max(0.0, state[1]))
        ts_ar.append(ts_ar[-1] + dt)
        
        if state[1] <= 0:
            break
            
    return ts_vacuo, xs_vacuo, ys_vacuo, np.array(ts_ar), np.array(xs_ar), np.array(ys_ar)

# ============================================
# TÍTULO E ABAS
# ============================================
st.markdown('<div class="main-title">🚀 Cinemática Vetorial: Lançamento Oblíquo</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Análise da Independência dos Movimentos e Efeito da Resistência do Ar</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs([
    "1. Vácuo & Independência dos Movimentos", 
    "2. Com Resistência do Ar (Arrasto)"
])

# ============================================
# ABA 1: VÁCUO E INDEPENDÊNCIA DOS MOVIMENTOS
# ============================================
with tab1:
    st.markdown("""
    <div class="concept-card">
        <b>Princípio da Independência dos Movimentos (Galileu):</b> O movimento oblíquo pode ser decomposto em dois movimentos simultâneos e independentes:<br>
        • <b>Eixo X (Horizontal):</b> Movimento Retilíneo Uniforme (MRU), pois não há forças na horizontal ($v_x = \text{constante}$).<br>
        • <b>Eixo Y (Vertical):</b> Movimento Retilíneo Uniformemente Variado (MRUV), sob ação exclusiva da gravidade ($g$).
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(r"""
    $$ x(t) = v_0 \cos(\theta) t \quad \text{e} \quad y(t) = h_0 + v_0 \sin(\theta) t - \frac{1}{2}gt^2 $$
    """)
    
    col1, col2 = st.columns([1, 2.5])
    with col1:
        st.markdown("<div class='param-box'>", unsafe_allow_html=True)
        v0_1 = st.slider("Velocidade Inicial ($v_0$ em m/s)", 5.0, 30.0, 20.0, step=1.0, key='v0_1')
        ang_1 = st.slider("Ângulo de Lançamento ($\theta$)", 10, 85, 45, step=1, key='ang_1')
        h0_1 = st.slider("Altura Inicial ($h_0$ em m)", 0.0, 10.0, 0.0, step=0.5, key='h0_1')
        
        duracao_ms = st.select_slider("Velocidade da Animação", options=[10, 20, 30, 50, 100], value=30, key='dur_1')
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        ts_v, xs_v, ys_v, _, _, _ = simular_lancamento(v0_1, ang_1, h0_1, m=1.0, b=0.0)
        
        # Ajuste de espaçamento vertical e alturas de linha para evitar sobreposição do título inferior
        fig1 = make_subplots(
            rows=2, cols=2, 
            column_widths=[0.72, 0.28], row_heights=[0.62, 0.30],
            horizontal_spacing=0.15, vertical_spacing=0.26,
            subplot_titles=("Trajetória Principal (X vs Y)", "Projeção Y(t)", "Projeção X(t)", "")
        )
        
        fig1.add_trace(go.Scatter(x=xs_v, y=ys_v, mode='lines', line=dict(color=COR_VACUO, width=3, dash='dash'), name='Trajetória'), row=1, col=1)
        fig1.add_trace(go.Scatter(x=[xs_v[0]], y=[ys_v[0]], mode='markers', marker=dict(color=COR_PROJETIL, size=16), name='Projétil'), row=1, col=1)
        
        fig1.add_trace(go.Scatter(x=ts_v, y=ys_v, mode='lines', line=dict(color='#9b59b6', width=2), name='Eixo Y (MRUV)'), row=1, col=2)
        fig1.add_trace(go.Scatter(x=xs_v, y=ts_v, mode='lines', line=dict(color='#e67e22', width=2), name='Eixo X (MRU)'), row=2, col=1)
        
        n_frames = len(xs_v)
        frames = []
        for i in range(n_frames):
            frames.append(go.Frame(
                data=[
                    go.Scatter(x=xs_v[:i+1], y=ys_v[:i+1]),
                    go.Scatter(x=[xs_v[i]], y=[ys_v[i]]),
                    go.Scatter(x=ts_v[:i+1], y=ys_v[:i+1]),
                    go.Scatter(x=xs_v[:i+1], y=ts_v[:i+1])
                ],
                traces=[0, 1, 2, 3],
                name=f"f{i}"
            ))
            
        fig1.frames = frames
        fig1.update_layout(
            height=530, showlegend=False, paper_bgcolor='white', plot_bgcolor='white',
            margin=dict(l=20, r=20, t=80, b=20),
            updatemenus=[{
                "type": "buttons", "showactive": False, "x": 0.0, "y": 1.25,
                "buttons": [
                    {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": duracao_ms, "redraw": True}, "fromcurrent": True}]},
                    {"label": "❚❚ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}}]}
                ]
            }]
        )
        
        fig1.update_xaxes(title_text="Distância X (m)", range=[0, max(xs_v)*1.1], row=1, col=1)
        fig1.update_yaxes(title_text="Altura Y (m)", range=[0, max(ys_v)*1.2], row=1, col=1)
        fig1.update_xaxes(title_text="Tempo (s)", range=[0, ts_v[-1]*1.1], row=1, col=2)
        fig1.update_yaxes(title_text="Altura Y (m)", range=[0, max(ys_v)*1.2], row=1, col=2)
        fig1.update_xaxes(title_text="Distância X (m)", range=[0, max(xs_v)*1.1], row=2, col=1)
        fig1.update_yaxes(title_text="Tempo (s)", range=[0, ts_v[-1]*1.1], row=2, col=1)
        
        st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})

# ============================================
# ABA 2: COM RESISTÊNCIA DO AR (ARRASTO)
# ============================================
with tab2:
    st.markdown("""
    <div class="concept-card" style="border-left-color: #ef4444;">
        <b>Lançamento Real com Arrasto Atmosférico:</b> No mundo real, o ar exerce uma força oposta à velocidade ($F_d = -b v \vec{v}$). Isso reduz drasticamente o alcance máximo e a altura máxima, além de tornar a trajetória assimétrica.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(r"""
    $$ m \vec{a} = m\vec{g} - b v \vec{v} \implies \begin{cases} a_x = -\frac{b}{m} v v_x \\ a_y = -g - \frac{b}{m} v v_y \end{cases} $$
    """)
    
    col_a1, col_a2 = st.columns([1, 2.5])
    with col_a1:
        st.markdown("<div class='param-box'>", unsafe_allow_html=True)
        v0_2 = st.slider("Velocidade Inicial ($v_0$ em m/s)", 5.0, 30.0, 20.0, step=1.0, key='v0_2')
        ang_2 = st.slider("Ângulo ($\theta$)", 10, 85, 45, step=1, key='ang_2')
        massa_2 = st.slider("Massa do Projétil ($m$ em kg)", 0.1, 5.0, 1.0, step=0.1, key='m_2')
        coef_b = st.slider("Coef. de Arrasto ($b$)", 0.0, 0.2, 0.05, step=0.01, key='b_2')
        
        duracao_ms_2 = st.select_slider("Velocidade da Animação", options=[10, 20, 30, 50, 100], value=30, key='dur_2')
        st.markdown("</div>", unsafe_allow_html=True)

    with col_a2:
        ts_v, xs_v, ys_v, ts_ar, xs_ar, ys_ar = simular_lancamento(v0_2, ang_2, h0=0.0, m=massa_2, b=coef_b)
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(x=xs_v, y=ys_v, mode='lines', line=dict(color=COR_VACUO, width=2, dash='dash'), name='Ideal (Sem Ar)'))
        fig2.add_trace(go.Scatter(x=xs_ar, y=ys_ar, mode='lines', line=dict(color=COR_AR, width=3), name='Real (Com Ar)'))
        fig2.add_trace(go.Scatter(x=[xs_ar[0]], y=[ys_ar[0]], mode='markers', marker=dict(color=COR_PROJETIL, size=16), name='Projétil'))
        
        n_frames_ar = len(xs_ar)
        idxs_v = np.linspace(0, len(xs_v)-1, n_frames_ar, dtype=int)
        
        frames = []
        for i in range(n_frames_ar):
            iv = idxs_v[i]
            frames.append(go.Frame(
                data=[
                    go.Scatter(x=xs_v[:iv+1], y=ys_v[:iv+1]),
                    go.Scatter(x=xs_ar[:i+1], y=ys_ar[:i+1]),
                    go.Scatter(x=[xs_ar[i]], y=[ys_ar[i]])
                ],
                traces=[0, 1, 2],
                name=f"f{i}"
            ))
            
        fig2.frames = frames
        fig2.update_layout(
            title="Comparativo Dinâmico: Vácuo vs. Resistência do Ar",
            height=430, paper_bgcolor='white', plot_bgcolor='white',
            margin=dict(l=20, r=20, t=80, b=20),
            xaxis=dict(title="Distância Horizontal (m)", range=[0, max(xs_v[-1], xs_ar[-1])*1.1]),
            yaxis=dict(title="Altura Vertical (m)", range=[0, max(max(ys_v), max(ys_ar))*1.2]),
            updatemenus=[{
                "type": "buttons", "showactive": False, "x": 0.0, "y": 1.25,
                "buttons": [
                    {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": duracao_ms_2, "redraw": True}, "fromcurrent": True}]},
                    {"label": "❚❚ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}}]}
                ]
            }]
        )
        
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
        
        alcance_vacuo = xs_v[-1]
        alcance_ar = xs_ar[-1]
        perda_alcance = ((alcance_vacuo - alcance_ar) / alcance_vacuo) * 100
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Alcance no Vácuo", f"{alcance_vacuo:.2f} m")
        col_m2.metric("Alcance com Resistência do Ar", f"{alcance_ar:.2f} m", delta=f"-{perda_alcance:.1f}%", delta_color="inverse")

# Rodapé
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.85rem; padding: 1rem;">
    🚀 <b>Física Visual: Lançamento Oblíquo</b> — Simulações modeladas por Runge-Kutta de 4ª ordem (RK4).
</div>
""", unsafe_allow_html=True)
