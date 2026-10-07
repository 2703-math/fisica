import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# ============================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================
st.set_page_config(
    page_title="Física Visual: Dinâmica & Estática",
    page_icon="⚡",
    layout="wide"
)

# ============================================
# CSS PROFISSIONAL - ESTILO SAAS / DASHBOARD
# ============================================
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 1.8rem;
        font-weight: 400;
    }

    .dashboard-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.3rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
        margin-bottom: 1.2rem;
    }

    .card-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 0.5rem;
    }

    .formula-box {
        background: #f8fafc;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
        color: #334155;
        margin: 0.8rem 0;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# MOTOR NUMÉRICO: INTEGRADOR RUNGE-KUTTA 4ª ORDEM (RK4)
# ============================================
def rk4_simular_movimento(massa, forca_ext, mu_k, t_max=5.0, dt=0.01, g=10.0):
    """
    Integração numérica via RK4 para resolver:
    dx/dt = v
    dv/dt = a(t, x, v) = (F_ext - sgn(v)*F_at) / m
    """
    normal = massa * g
    f_at = mu_k * normal
    
    # f_atrito impede o movimento se forca_ext <= f_at
    if forca_ext <= f_at:
        t_pts = np.arange(0, t_max + dt, dt)
        return t_pts, np.zeros_like(t_pts), np.zeros_like(t_pts), np.zeros_like(t_pts)

    def acel(v):
        f_resultante = forca_ext - f_at
        return f_resultante / massa

    N_steps = int(t_max / dt)
    t_pts = np.linspace(0, t_max, N_steps + 1)
    x_pts = np.zeros(N_steps + 1)
    v_pts = np.zeros(N_steps + 1)
    a_pts = np.full(N_steps + 1, (forca_ext - f_at) / massa)

    for i in range(N_steps):
        x = x_pts[i]
        v = v_pts[i]

        # RK4 para x e v
        k1_v = acel(v)
        k1_x = v

        k2_v = acel(v + 0.5 * dt * k1_v)
        k2_x = v + 0.5 * dt * k1_v

        k3_v = acel(v + 0.5 * dt * k2_v)
        k3_x = v + 0.5 * dt * k2_v

        k4_v = acel(v + dt * k3_v)
        k4_x = v + dt * k3_v

        x_pts[i+1] = x + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        v_pts[i+1] = v + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    return t_pts, x_pts, v_pts, a_pts

# ============================================
# FUNÇÕES DE PLOTAGEM DE DIAGRAMAS
# ============================================
def plot_plano_horizontal(massa, forca_aplicada, mu):
    fig = go.Figure()
    g = 10
    normal = massa * g
    atrito = normal * mu

    f_scale = 0.04
    v_scale = 0.015

    len_f = (forca_aplicada * f_scale) if forca_aplicada > 0 else 0
    len_at = (atrito * f_scale) if atrito > 0 else 0
    len_vert = normal * v_scale

    max_x_chao = max(8.0, 2.0 + len_f, 2.0 + len_at)
    fig.add_shape(type="rect", x0=-max_x_chao, y0=-1, x1=max_x_chao, y1=0,
                  fillcolor="#cbd5e1", line=dict(width=0))
    
    fig.add_shape(type="rect", x0=-1.5, y0=0, x1=1.5, y1=2,
                  fillcolor="#3b82f6", line=dict(color="#1d4ed8", width=2))
    
    if forca_aplicada > 0:
        x_end_f = 1.5 + max(1.5, len_f)
        fig.add_annotation(
            x=x_end_f, y=1, ax=1.5, ay=1,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=3, arrowcolor="#10b981"
        )
        fig.add_annotation(x=1.5 + (x_end_f - 1.5)/2, y=1.5, text=f"F = {forca_aplicada:.1f} N", showarrow=False, font=dict(color="#059669", size=13))
    
    if atrito > 0:
        x_end_at = -1.5 - max(1.5, len_at)
        fig.add_annotation(
            x=x_end_at, y=0.5, ax=-1.5, ay=0.5,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=3, arrowcolor="#ef4444"
        )
        fig.add_annotation(x=-1.5 + (x_end_at - (-1.5))/2, y=1.0, text=f"Fat = {atrito:.1f} N", showarrow=False, font=dict(color="#dc2626", size=13))

    y_down = 1 - max(2.0, len_vert)
    fig.add_annotation(
        x=0, y=y_down, ax=0, ay=1, xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowwidth=3, arrowcolor="#8b5cf6"
    )
    fig.add_annotation(x=1.2, y=1 + (y_down - 1)/2, text=f"P = {normal:.1f} N", showarrow=False, font=dict(color="#7c3aed", size=13))

    y_up = 1 + max(2.0, len_vert)
    fig.add_annotation(
        x=0, y=y_up, ax=0, ay=1, xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowwidth=3, arrowcolor="#f59e0b"
    )
    fig.add_annotation(x=1.2, y=1 + (y_up - 1)/2, text=f"N = {normal:.1f} N", showarrow=False, font=dict(color="#d97706", size=13))

    lim_x = max_x_chao + 2.0
    lim_y = max(4.0, abs(y_down), abs(y_up)) + 1.5

    fig.update_layout(
        xaxis=dict(range=[-lim_x, lim_x], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-lim_y, lim_y], showgrid=False, zeroline=False, visible=False),
        plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=0, r=0, t=10, b=10), height=320
    )
    return fig

def plot_plano_inclinado(massa, angulo_deg):
    fig = go.Figure()
    g = 10
    peso = massa * g
    ang_rad = math.radians(angulo_deg)
    px = peso * math.sin(ang_rad)
    py = peso * math.cos(ang_rad)
    
    R = 10 
    L = R * math.cos(ang_rad)
    H = R * math.sin(ang_rad)
    
    fig.add_trace(go.Scatter(
        x=[0, L, 0, 0], y=[0, 0, H, 0],
        fill="toself", fillcolor="#f1f5f9", line=dict(color="#cbd5e1", width=2),
        showlegend=False, hoverinfo="skip"
    ))
    
    cx, cy = L / 2, H / 2
    s = 1.0
    bx = cx + s * math.sin(ang_rad)
    by = cy + s * math.cos(ang_rad)
    
    def rot(px_val, py_val):
        rx = px_val * math.cos(-ang_rad) - py_val * math.sin(-ang_rad)
        ry = px_val * math.sin(-ang_rad) + py_val * math.cos(-ang_rad)
        return bx + rx, by + ry

    p1, p2, p3, p4 = rot(-s, -s), rot(s, -s), rot(s, s), rot(-s, s)
    
    fig.add_trace(go.Scatter(
        x=[p1[0], p2[0], p3[0], p4[0], p1[0]],
        y=[p1[1], p2[1], p3[1], p4[1], p1[1]],
        fill="toself", fillcolor="#3b82f6", line=dict(color="#1d4ed8", width=2),
        showlegend=False, hoverinfo="skip"
    ))
    
    force_scale = 0.015
    len_p = max(1.8, peso * force_scale)
    len_n = max(1.8, py * force_scale)
    len_px = max(1.8, px * force_scale)
    
    fig.add_annotation(
        x=bx, y=by - len_p, ax=bx, ay=by, xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowwidth=3, arrowcolor="#8b5cf6"
    )
    fig.add_annotation(x=bx + 0.8, y=by - len_p / 2, text=f"P={peso:.1f}N", showarrow=False, font=dict(color="#7c3aed", size=12))
    
    nx = bx + len_n * math.sin(ang_rad)
    ny = by + len_n * math.cos(ang_rad)
    fig.add_annotation(
        x=nx, y=ny, ax=bx, ay=by, xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor="#f59e0b"
    )
    fig.add_annotation(x=nx + 0.6*math.sin(ang_rad), y=ny + 0.6*math.cos(ang_rad), text=f"N={py:.1f}N", showarrow=False, font=dict(color="#d97706", size=12))
    
    pxx = bx + len_px * math.cos(ang_rad)
    pxy = by - len_px * math.sin(ang_rad)
    fig.add_annotation(
        x=pxx, y=pxy, ax=bx, ay=by, xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor="#10b981"
    )
    fig.add_annotation(x=pxx + 0.6*math.cos(ang_rad), y=pxy - 0.6*math.sin(ang_rad), text=f"Px={px:.1f}N", showarrow=False, font=dict(color="#059669", size=13))

    max_dim = max(L, H) + 4.0
    fig.update_layout(
        xaxis=dict(range=[-3, max_dim], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-4, max_dim], scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False, visible=False),
        plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=0, r=0, t=0, b=0), height=350
    )
    return fig

def plot_corpo_extenso_torque(L, x_pivo, massa_viga, f1, x1, ang1_deg, f2, x2, ang2_deg):
    fig = go.Figure()
    g = 10
    p_viga = massa_viga * g
    x_cg = L / 2.0

    # 1. Viga rígida
    fig.add_shape(type="rect", x0=0, y0=-0.15, x1=L, y1=0.15,
                  fillcolor="#cbd5e1", line=dict(color="#475569", width=2))

    # 2. Apoio/Pivô (Triângulo)
    fig.add_trace(go.Scatter(
        x=[x_pivo - 0.3, x_pivo, x_pivo + 0.3, x_pivo - 0.3],
        y=[-0.6, -0.15, -0.6, -0.6],
        fill="toself", fillcolor="#f59e0b", line=dict(color="#d97706", width=2),
        name="Pivô / Apoio", showlegend=False, hoverinfo="skip"
    ))

    scale = 0.02

    # Força 1
    ang1_rad = math.radians(ang1_deg)
    dx1 = f1 * math.cos(ang1_rad) * scale
    dy1 = f1 * math.sin(ang1_rad) * scale
    fig.add_annotation(
        x=x1 + dx1, y=0.15 + dy1, ax=x1, ay=0.15,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowwidth=3, arrowcolor="#10b981"
    )
    fig.add_annotation(x=x1 + dx1*0.5, y=0.4 + dy1, text=f"F₁ = {f1:.1f} N ({ang1_deg}°)", showarrow=False, font=dict(color="#059669", size=12))

    # Força 2
    ang2_rad = math.radians(ang2_deg)
    dx2 = f2 * math.cos(ang2_rad) * scale
    dy2 = f2 * math.sin(ang2_rad) * scale
    fig.add_annotation(
        x=x2 + dx2, y=0.15 + dy2, ax=x2, ay=0.15,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowwidth=3, arrowcolor="#3b82f6"
    )
    fig.add_annotation(x=x2 + dx2*0.5, y=0.4 + dy2, text=f"F₂ = {f2:.1f} N ({ang2_deg}°)", showarrow=False, font=dict(color="#1d4ed8", size=12))

    # Peso da Viga (no centro de gravidade)
    len_p = max(0.8, p_viga * scale)
    fig.add_annotation(
        x=x_cg, y=-0.15 - len_p, ax=x_cg, ay=-0.15,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowwidth=3, arrowcolor="#ef4444"
    )
    fig.add_annotation(x=x_cg, y=-0.3 - len_p, text=f"P_viga = {p_viga:.1f} N", showarrow=False, font=dict(color="#dc2626", size=12))

    fig.update_layout(
        xaxis=dict(range=[-1, L + 1], title="Posição ao longo da viga (m)", showgrid=True, zeroline=True),
        yaxis=dict(range=[-3.0, 3.0], showgrid=False, zeroline=False, visible=False),
        plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=10, r=10, t=10, b=10), height=380
    )
    return fig

# ============================================
# TÍTULO E ABAS SUPERIORES
# ============================================
st.markdown('<div class="main-title">⚡ Física Visual: Dinâmica & Estática</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Mecânica Clássica Interativa — Leis de Newton, Plano Inclinado, Solvedor RK4 e Torque</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "  1. 2ª Lei de Newton (Horizontal & RK4)  ", 
    "  2. Plano Inclinado (Decomposição)  ",
    "  3. Torque & Equilíbrio de Corpos Extensos  "
])

g = 10.0

# ============================================
# ABA 1: 2ª LEI DE NEWTON + RK4
# ============================================
with tab1:
    col_left, col_right = st.columns([1, 1.4], gap="medium")
    
    with col_left:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">⚙️ Controles do Sistema</div>', unsafe_allow_html=True)
        
        massa = st.slider("Massa do Bloco (kg)", 1.0, 50.0, 10.0, step=1.0, key="m1")
        forca = st.slider("Força Aplicada F (N)", 0.0, 200.0, 80.0, step=5.0, key="f1")
        mu = st.slider("Coeficiente de Atrito (μ)", 0.0, 1.0, 0.3, step=0.05, key="mu1")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        normal = massa * g
        atrito = mu * normal
        forca_resultante = max(0.0, forca - atrito)
        aceleracao = forca_resultante / massa
        
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">📊 Indicadores Chave (KPIs)</div>', unsafe_allow_html=True)
        
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Força Resultante", f"{forca_resultante:.1f} N")
        m_col2.metric("Aceleração (a)", f"{aceleracao:.2f} m/s²", delta=f"{aceleracao:.1f}" if aceleracao > 0 else "Repouso")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">🖥️ Diagrama de Corpo Livre</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_plano_horizontal(massa, forca, mu), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Simulação Dinâmica RK4
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">📈 Integração Numérica do Movimento (Algoritmo Runge-Kutta 4ª Ordem)</div>', unsafe_allow_html=True)
    
    t_pts, x_pts, v_pts, a_pts = rk4_simular_movimento(massa, forca, mu, t_max=5.0)

    fig_rk4 = make_subplots(rows=1, cols=2, subplot_titles=("Posição x(t) [m]", "Velocidade v(t) [m/s]"))
    fig_rk4.add_trace(go.Scatter(x=t_pts, y=x_pts, mode='lines', name='x(t)', line=dict(color='#3b82f6', width=2.5)), row=1, col=1)
    fig_rk4.add_trace(go.Scatter(x=t_pts, y=v_pts, mode='lines', name='v(t)', line=dict(color='#10b981', width=2.5)), row=1, col=2)
    fig_rk4.update_layout(height=300, plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
    st.plotly_chart(fig_rk4, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# ABA 2: PLANO INCLINADO
# ============================================
with tab2:
    col_left, col_right = st.columns([1, 1.4], gap="medium")
    
    with col_left:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">⚙️ Controles do Plano</div>', unsafe_allow_html=True)
        
        massa_plano = st.slider("Massa do Bloco (kg)", 1.0, 50.0, 10.0, step=1.0, key="m2")
        angulo = st.slider("Ângulo de Inclinação (°)", 0, 90, 30, step=1, key="ang2")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        p = massa_plano * g
        ang_rad = math.radians(angulo)
        px = p * math.sin(ang_rad)
        py = p * math.cos(ang_rad)
        
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">📊 Componentes Vetoriais</div>', unsafe_allow_html=True)
        
        p_col1, p_col2 = st.columns(2)
        p_col1.metric("Tangencial (Px)", f"{px:.1f} N")
        p_col2.metric("Normal (Py)", f"{py:.1f} N")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">🖥️ Decomposição Gráfica do Peso</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_plano_inclinado(massa_plano, angulo), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">🧮 Fórmulas Utilizadas</div>', unsafe_allow_html=True)
        st.markdown(r"$$ P_x = P \cdot \sin(\theta) \quad \text{e} \quad P_y = P \cdot \cos(\theta) $$")
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# ABA 3: TORQUE E EQUILÍBRIO DE CORPOS EXTENSOS
# ============================================
with tab3:
    col_left, col_right = st.columns([1, 1.4], gap="medium")

    with col_left:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">⚙️ Geometria da Viga e Pivô</div>', unsafe_allow_html=True)
        
        L_viga = 10.0  # Comprimento padrão da viga (m)
        massa_viga = st.slider("Massa da Viga (kg)", 0.0, 100.0, 20.0, step=5.0, key="m_viga")
        x_pivo = st.slider("Posição do Apoio / Pivô (m)", 0.0, L_viga, 5.0, step=0.5, key="xpivo")
        
        st.markdown('---')
        st.markdown('**Força 1 ($F_1$)**')
        f1 = st.slider("Intensidade F₁ (N)", 0.0, 300.0, 100.0, step=10.0, key="f1_t")
        x1 = st.slider("Posição x₁ (m)", 0.0, L_viga, 1.0, step=0.5, key="x1_t")
        ang1 = st.slider("Ângulo θ₁ (°)", 0, 180, 90, step=5, key="ang1_t")

        st.markdown('---')
        st.markdown('**Força 2 ($F_2$)**')
        f2 = st.slider("Intensidade F₂ (N)", 0.0, 300.0, 100.0, step=10.0, key="f2_t")
        x2 = st.slider("Posição x₂ (m)", 0.0, L_viga, 9.0, step=0.5, key="x2_t")
        ang2 = st.slider("Ângulo θ₂ (°)", 0, 180, 90, step=5, key="ang2_t")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Cálculos Físicos de Torque e Equilíbrio
    p_viga = massa_viga * g
    x_cg = L_viga / 2.0

    # Torques em relação ao Pivô selecionado: τ = r * F * sin(θ)
    # Convenção: Anti-horário = Positivo (+), Horário = Negativo (-)
    r1 = x1 - x_pivo
    tau1 = r1 * f1 * math.sin(math.radians(ang1))

    r2 = x2 - x_pivo
    tau2 = r2 * f2 * math.sin(math.radians(ang2))

    r_cg = x_cg - x_pivo
    tau_peso = r_cg * (-p_viga)  # Peso aponta para baixo (-y)

    tau_resultante = tau1 + tau2 + tau_peso

    # Forças verticais e horizontais resultantes
    f1_y = f1 * math.sin(math.radians(ang1))
    f2_y = f2 * math.sin(math.radians(ang2))
    f_apoio_y = p_viga - f1_y - f2_y  # Reação no apoio para Fy = 0

    f1_x = f1 * math.cos(math.radians(ang1))
    f2_x = f2 * math.cos(math.radians(ang2))
    f_res_x = f1_x + f2_x

    em_equilibrio_torque = abs(tau_resultante) < 1e-2
    em_equilibrio_forca = abs(f_res_x) < 1e-2 and f_apoio_y >= 0

    with col_right:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">🖥️ Diagrama de Corpo Extenso e Forças</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_corpo_extenso_torque(L_viga, x_pivo, massa_viga, f1, x1, ang1, f2, x2, ang2), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Dashboard de Balanço de Torques
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">📊 Indicadores de Torque e Equilíbrio</div>', unsafe_allow_html=True)
        
        t_col1, t_col2 = st.columns(2)
        t_col1.metric("Torque Resultante (τ_res)", f"{tau_resultante:.1f} N·m", 
                       delta="Equilíbrio Rotacional" if em_equilibrio_torque else f"{'Rotação Anti-Horária' if tau_resultante > 0 else 'Rotação Horária'}")
        t_col2.metric("Força de Reação no Apoio", f"{f_apoio_y:.1f} N")

        if em_equilibrio_torque and em_equilibrio_forca:
            st.success("✅ **Corpo Extenso em Equilíbrio Estático Completo!** (∑F = 0 e ∑τ = 0)")
        else:
            st.warning("⚠️ **Corpo Fora de Equilíbrio!** Ajuste as forças, posições ou ângulos para zerar o torque resultante.")

        st.markdown('</div>', unsafe_allow_html=True)

        # Passo a Passo da Segunda Condição de Equilíbrio
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">🧮 Equações de Torque (2ª Condição de Equilíbrio)</div>', unsafe_allow_html=True)
        st.markdown(r"$$ \tau = r \cdot F \cdot \sin(\theta) \quad \implies \quad \sum \vec{\tau} = \tau_1 + \tau_2 + \tau_{\text{peso}} = 0 $$")
        st.markdown(f"""
        * **Torque $F_1$:** $({r1:.1f}) \\times {f1:.1f} \\times \\sin({ang1}^\\circ) = {tau1:.1f}\\text{{ N}}\\cdot\\text{{m}}$
        * **Torque $F_2$:** $({r2:.1f}) \\times {f2:.1f} \\times \\sin({ang2}^\\circ) = {tau2:.1f}\\text{{ N}}\\cdot\\text{{m}}$
        * **Torque do Peso da Viga:** $({r_cg:.1f}) \\times (-{p_viga:.1f}) = {tau_peso:.1f}\\text{{ N}}\\cdot\\text{{m}}$
        * **Torque Resultante ($\sum \\tau$):** **${tau_resultante:.1f}\\text{{ N}}\\cdot\\text{{m}}$**
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# Rodapé
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 1rem;">
    ⚡ <b>Física Visual SaaS</b> — Plataforma Educacional de Alta Performance (Simulação Numérica RK4)
</div>
""", unsafe_allow_html=True)
