import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math

# ============================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================
st.set_page_config(
    page_title="Física Visual: Dinâmica",
    page_icon="⚡",
    layout="wide"
)

# ============================================
# CSS PROFISSIONAL - ESTILO SAAS / DASHBOARD
# ============================================
st.markdown("""
<style>
    /* Ocultar elementos padrão do Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Fundo geral da aplicação mais limpo (Off-white moderno) */
    .stApp {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Títulos e Cabeçalhos */
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
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Cartões do Dashboard (Cards) */
    .dashboard-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
        margin-bottom: 1.2rem;
    }

    .card-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 0.6rem;
    }

    /* Caixa de Fórmula Matemática */
    .formula-box {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
        font-family: monospace;
        color: #334155;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# FUNÇÕES DE PLOTAGEM COM AUTOSCALE DINÂMICO
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
        fig.add_annotation(x=1.5 + (x_end_f - 1.5)/2, y=1.5, text=f"F = {forca_aplicada:.1f} N", showarrow=False, font=dict(color="#059669", size=13, family="sans-serif"))
    
    if atrito > 0:
        x_end_at = -1.5 - max(1.5, len_at)
        fig.add_annotation(
            x=x_end_at, y=0.5, ax=-1.5, ay=0.5,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=3, arrowcolor="#ef4444"
        )
        fig.add_annotation(x=-1.5 + (x_end_at - (-1.5))/2, y=1.0, text=f"Fat = {atrito:.1f} N", showarrow=False, font=dict(color="#dc2626", size=13, family="sans-serif"))

    y_down = 1 - max(2.0, len_vert)
    fig.add_annotation(
        x=0, y=y_down, ax=0, ay=1, xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowwidth=3, arrowcolor="#8b5cf6"
    )
    fig.add_annotation(x=1.2, y=1 + (y_down - 1)/2, text=f"P = {normal:.1f} N", showarrow=False, font=dict(color="#7c3aed", size=13, family="sans-serif"))

    y_up = 1 + max(2.0, len_vert)
    fig.add_annotation(
        x=0, y=y_up, ax=0, ay=1, xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowwidth=3, arrowcolor="#f59e0b"
    )
    fig.add_annotation(x=1.2, y=1 + (y_up - 1)/2, text=f"N = {normal:.1f} N", showarrow=False, font=dict(color="#d97706", size=13, family="sans-serif"))

    lim_x = max_x_chao + 2.0
    lim_y = max(4.0, abs(y_down), abs(y_up)) + 1.5

    fig.update_layout(
        xaxis=dict(range=[-lim_x, lim_x], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-lim_y, lim_y], showgrid=False, zeroline=False, visible=False),
        plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=0, r=0, t=10, b=10), height=350
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
    
    cx = L / 2
    cy = H / 2
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
    
    pyx = bx - len_n * math.sin(ang_rad)
    pyy = by - len_n * math.cos(ang_rad)
    fig.add_annotation(
        x=pyx, y=pyy, ax=bx, ay=by, xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor="#ef4444"
    )
    fig.add_annotation(x=pyx - 0.6*math.sin(ang_rad), y=pyy - 0.6*math.cos(ang_rad), text=f"Py={py:.1f}N", showarrow=False, font=dict(color="#dc2626", size=12))
    
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
        plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=0, r=0, t=0, b=0), height=380
    )
    return fig

# ============================================
# TÍTULO E ABAS SUPERIORES
# ============================================
st.markdown('<div class="main-title">🍎 Física Visual: Dinâmica</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Plataforma Interativa de Ensino de Mecânica Clássica</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs([
    "  2ª Lei de Newton (Horizontal)  ", 
    "  Plano Inclinado (Decomposição)  "
])

g = 10

# ============================================
# ABA 1: 2ª LEI DE NEWTON
# ============================================
with tab1:
    col_left, col_right = st.columns([1, 1.4], gap="medium")
    
    with col_left:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">⚙️ Controles do Sistema</div>', unsafe_allow_html=True)
        
        massa = st.slider("Massa do Bloco (kg)", 1.0, 50.0, 10.0, step=1.0, key="m1")
        forca = st.slider("Força Aplicada (N)", 0.0, 200.0, 80.0, step=5.0, key="f1")
        mu = st.slider("Coeficiente de Atrito (μ)", 0.0, 1.0, 0.3, step=0.05, key="mu1")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Bloco de Métricas Principais estilo SaaS
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">📊 Indicadores Chave (KPIs)</div>', unsafe_allow_html=True)
        
        normal = massa * g
        atrito = mu * normal
        forca_resultante = max(0.0, forca - atrito)
        aceleracao = forca_resultante / massa
        
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Força Resultante", f"{forca_resultante:.1f} N")
        m_col2.metric("Aceleração", f"{aceleracao:.2f} m/s²", delta=f"{aceleracao:.1f}" if aceleracao > 0 else "Repouso")
        
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">🖥️ Simulação Gráfica em Tempo Real</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_plano_horizontal(massa, forca, mu), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">🧮 Raciocínio Passo a Passo</div>', unsafe_allow_html=True)
        st.markdown(r"$$ F_R = F - F_{at} = m \cdot a $$")
        st.markdown(f"""
        * **Normal ($N$):** $m \\cdot g = {massa} \\times {g} = {normal:.1f}\\text{{ N}}$
        * **Atrito ($F_{{at}}$):** $\\mu \\cdot N = {mu} \\times {normal:.1f} = {atrito:.1f}\\text{{ N}}$
        * **Resultante ($F_R$):** ${forca} - {atrito:.1f} = {forca_resultante:.1f}\\text{{ N}}$
        """)
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
        st.markdown('<div class="card-header">📊 Indicadores do Plano</div>', unsafe_allow_html=True)
        
        p_col1, p_col2 = st.columns(2)
        p_col1.metric("Componente Paralela (Px)", f"{px:.1f} N")
        p_col2.metric("Componente Perpendicular (Py)", f"{py:.1f} N")
        
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">🖥️ Decomposição Gráfica de Vetores</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_plano_inclinado(massa_plano, angulo), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">🧮 Equações Trigonométricas</div>', unsafe_allow_html=True)
        st.markdown(r"$$ P_x = P \cdot \sin(\theta) \quad | \quad P_y = P \cdot \cos(\theta) $$")
        st.markdown(f"""
        * **Peso Total ($P$):** ${p:.1f}\\text{{ N}}$
        * **$P_x$ (Deslizamento):** ${p:.1f} \\cdot \\sin({angulo}^\\circ) = {px:.1f}\\text{{ N}}$
        * **$P_y$ (Pressão):** ${p:.1f} \\cdot \\cos({angulo}^\\circ) = {py:.1f}\\text{{ N}}$
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# Rodapé
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 1rem;">
    ⚡ <b>Física Visual SaaS</b> — Plataforma Educacional de Alta Performance
</div>
""", unsafe_allow_html=True)
