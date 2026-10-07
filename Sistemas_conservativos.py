import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# ============================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================
st.set_page_config(
    page_title="Física Visual: Energia e Dinâmica",
    page_icon="⚡",
    layout="wide"
)

# ============================================
# CSS PERSONALIZADO
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
        border-left: 4px solid;
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

# Cores padrão para as Energias
COR_EC = "#3498db"   # Cinética (Azul)
COR_EPG = "#9b59b6"  # Potencial Gravitacional (Roxo)
COR_EPE = "#2ecc71"  # Potencial Elástica (Verde)
COR_EM = "#34495e"   # Mecânica (Cinza Escuro)

# ============================================
# FUNÇÕES AUXILIARES DE DESENHO
# ============================================
def criar_mola(x0, x1, y0, n_voltas=12, largura=0.35):
    if x0 >= x1:
        return [x0, x1], [y0, y0]
    x_vals = np.linspace(x0, x1, n_voltas * 2)
    y_vals = np.zeros_like(x_vals)
    for i in range(len(x_vals)):
        if i == 0 or i == len(x_vals) - 1:
            y_vals[i] = y0
        elif i % 2 == 0:
            y_vals[i] = y0 + largura
        else:
            y_vals[i] = y0 - largura
    return x_vals, y_vals

def criar_bloco(x_centro, y_base, largura=0.8, altura=0.8):
    hx = largura / 2
    x = [x_centro - hx, x_centro + hx, x_centro + hx, x_centro - hx, x_centro - hx]
    y = [y_base, y_base, y_base + altura, y_base + altura, y_base]
    return x, y

# ============================================
# NÚCLEO DE SIMULAÇÃO FÍSICA (RK4) E GERAÇÃO DE FIGURAS
# ============================================

# 1. Rampa em U
def gerar_figura_rampa_u(massa, altura_max, duracao_ms, gravidade=10):
    x_max = 5.0
    k = altura_max / (x_max**2)
    em = massa * gravidade * altura_max

    dt_sim = 0.002
    t_total = 2.0 * math.pi * math.sqrt(1 / (2 * gravidade * k))
    n_steps = int(t_total / dt_sim)

    x = x_max
    v = 0.0
    xs = [x]
    ys = [k * x**2]
    
    def deriv(state):
        xi, vi = state
        denom = 1.0 + 4.0 * (k**2) * (xi**2)
        acc = (-2.0 * gravidade * k * xi - 4.0 * (k**2) * xi * (vi**2)) / denom
        return np.array([vi, acc])

    for _ in range(n_steps):
        state = np.array([x, v])
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt_sim * k1)
        k3 = deriv(state + 0.5 * dt_sim * k2)
        k4 = deriv(state + dt_sim * k3)
        state += (dt_sim / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        x, v = state[0], state[1]
        if x > x_max:
            x = x_max
            v = -v
        elif x < -x_max:
            x = -x_max
            v = -v
        xs.append(x)
        ys.append(k * x**2)

    n_frames = 80
    idxs = np.linspace(0, len(xs)-1, n_frames, dtype=int)
    xs_f = [xs[i] for i in idxs]
    ys_f = [ys[i] for i in idxs]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.68, 0.32], horizontal_spacing=0.08)

    x_pista = np.linspace(-x_max, x_max, 100)
    y_pista = k * (x_pista**2)
    fig.add_trace(go.Scatter(x=x_pista, y=y_pista, mode='lines', line=dict(color='#7f8c8d', width=4), hoverinfo='skip'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=[xs_f[0]], y=[ys_f[0] + 0.3], mode='markers', marker=dict(color='#e74c3c', size=22, line=dict(color='#c0392b', width=2)), hoverinfo='skip'), row=1, col=1)

    epg_ini = massa * gravidade * ys_f[0]
    ec_ini = max(0.0, em - epg_ini)
    fig.add_trace(go.Bar(x=['Ec', 'Epg', 'Em'], y=[ec_ini, epg_ini, em], marker_color=[COR_EC, COR_EPG, COR_EM], text=[f"{ec_ini:.1f}J", f"{epg_ini:.1f}J", f"{em:.1f}J"], textposition='auto'), row=1, col=2)

    frames = []
    for i, (xa, ya) in enumerate(zip(xs_f, ys_f)):
        epg = massa * gravidade * ya
        ec = max(0.0, em - epg)
        frames.append(go.Frame(
            data=[
                go.Scatter(x=[xa], y=[ya + 0.3]),
                go.Bar(y=[ec, epg, em], text=[f"{ec:.1f}J", f"{epg:.1f}J", f"{em:.1f}J"])
            ],
            traces=[1, 2],
            name=f"f{i}"
        ))

    fig.frames = frames
    fig.update_layout(
        showlegend=False, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=10, r=10, t=40, b=10), height=400,
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.0, "y": 1.15,
            "buttons": [
                {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": duracao_ms, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}, "mode": "immediate"}]},
                {"label": "❚❚ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]}
            ]
        }]
    )
    fig.update_xaxes(range=[-6.5, 6.5], showgrid=False, zeroline=False, visible=False, row=1, col=1)
    fig.update_yaxes(range=[-1, altura_max + 2.5], showgrid=False, zeroline=False, visible=False, row=1, col=1)
    fig.update_yaxes(range=[0, max(10.0, em * 1.2)], title="Energia (Joules)", row=1, col=2)
    return fig, t_total


# 2. Massa-Mola
def gerar_figura_massa_mola(massa, k_mola, amplitude, duracao_ms):
    em = 0.5 * k_mola * (amplitude**2)
    limite_x = amplitude + 3.5

    omega = math.sqrt(k_mola / massa)
    t_periodo = 2.0 * math.pi / omega
    dt_sim = 0.002
    n_steps = int(t_periodo / dt_sim)

    x = amplitude
    v = 0.0
    xs = [x]
    
    def deriv(state):
        xi, vi = state
        acc = -(k_mola / massa) * xi
        return np.array([vi, acc])

    for _ in range(n_steps):
        state = np.array([x, v])
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt_sim * k1)
        k3 = deriv(state + 0.5 * dt_sim * k2)
        k4 = deriv(state + dt_sim * k3)
        state += (dt_sim / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        x, v = state[0], state[1]
        xs.append(x)

    n_frames = 80
    idxs = np.linspace(0, len(xs)-1, n_frames, dtype=int)
    xs_f = [xs[i] for i in idxs]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.68, 0.32], horizontal_spacing=0.08)

    fig.add_trace(go.Scatter(x=[-limite_x, limite_x, limite_x, -limite_x, -limite_x], y=[-0.8, -0.8, 0, 0, -0.8], fill="toself", fillcolor="#bdc3c7", line=dict(width=0), hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[-limite_x, -limite_x + 0.4, -limite_x + 0.4, -limite_x, -limite_x], y=[0, 0, 2.2, 2.2, 0], fill="toself", fillcolor="#95a5a6", line=dict(width=0), hoverinfo='skip'), row=1, col=1)

    x_ini = xs_f[0]
    xm, ym = criar_mola(-limite_x + 0.4, x_ini - 0.4, 0.4, n_voltas=12)
    bx, by = criar_bloco(x_ini, 0)
    epe_ini = 0.5 * k_mola * (x_ini**2)
    ec_ini = max(0.0, em - epe_ini)

    fig.add_trace(go.Scatter(x=xm, y=ym, mode='lines', line=dict(color='#7f8c8d', width=3), hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=bx, y=by, fill="toself", fillcolor="#3498db", line=dict(color="#2980b9", width=2), hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Bar(x=['Ec', 'Epe', 'Em'], y=[ec_ini, epe_ini, em], marker_color=[COR_EC, COR_EPE, COR_EM], text=[f"{ec_ini:.1f}J", f"{epe_ini:.1f}J", f"{em:.1f}J"], textposition='auto'), row=1, col=2)

    frames = []
    for i, x_a in enumerate(xs_f):
        epe = 0.5 * k_mola * (x_a**2)
        ec = max(0.0, em - epe)
        xm_f, ym_f = criar_mola(-limite_x + 0.4, x_a - 0.4, 0.4, n_voltas=12)
        bx_f, by_f = criar_bloco(x_a, 0)

        frames.append(go.Frame(
            data=[
                go.Scatter(x=xm_f, y=ym_f),
                go.Scatter(x=bx_f, y=by_f),
                go.Bar(y=[ec, epe, em], text=[f"{ec:.1f}J", f"{epe:.1f}J", f"{em:.1f}J"])
            ],
            traces=[2, 3, 4],
            name=f"f{i}"
        ))

    fig.frames = frames
    fig.update_layout(
        showlegend=False, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=10, r=10, t=40, b=10), height=400,
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.0, "y": 1.15,
            "buttons": [
                {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": duracao_ms, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}, "mode": "immediate"}]},
                {"label": "❚❚ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]}
            ]
        }]
    )
    fig.update_xaxes(range=[-limite_x - 0.5, limite_x + 0.5], showgrid=False, zeroline=False, visible=False, row=1, col=1)
    fig.update_yaxes(range=[-1.2, 2.8], showgrid=False, zeroline=False, visible=False, row=1, col=1)
    fig.update_yaxes(range=[0, max(10.0, em * 1.2)], title="Energia (Joules)", row=1, col=2)
    return fig, t_periodo


# 3. Rampa + Mola (Comprimento da Mola Ajustável e RK4 Unificado)
def gerar_figura_rampa_mola_ref(massa, h_max, k_mola, comprimento_mola, duracao_ms, gravidade=10):
    em_total = massa * gravidade * h_max
    x_max_comp = math.sqrt((2 * em_total) / k_mola)

    x_topo_rampa = -6.0
    x_base_rampa = -2.0
    x_parede = 5.0
    x_inicio_mola = x_parede - comprimento_mola
    
    a_r = h_max / ((x_topo_rampa - x_base_rampa)**2)

    # Núcleo RK4 Unificado (Rampa -> Plano -> Mola -> Retorno)
    def get_accel(x):
        if x < x_base_rampa:
            return -gravidade * 2.0 * a_r * (x - x_base_rampa)
        elif x < x_inicio_mola:
            return 0.0
        else:
            comp = x - x_inicio_mola
            return -(k_mola / massa) * comp

    dt = 0.001
    x = x_topo_rampa
    v = 0.0
    xs = [x]

    for _ in range(5000):
        def deriv(pos, vel):
            return vel, get_accel(pos)
        
        p1, v1 = x, v
        f1_p, f1_v = deriv(p1, v1)
        
        p2 = p1 + 0.5 * dt * f1_p
        v2 = v1 + 0.5 * dt * f1_v
        f2_p, f2_v = deriv(p2, v2)
        
        p3 = p1 + 0.5 * dt * f2_p
        v3 = v1 + 0.5 * dt * f2_v
        f3_p, f3_v = deriv(p3, v3)
        
        p4 = p1 + dt * f3_p
        v4 = v1 + dt * f3_v
        f4_p, f4_v = deriv(p4, v4)
        
        x += (dt / 6.0) * (f1_p + 2*f2_p + 2*f3_p + f4_p)
        v += (dt / 6.0) * (f1_v + 2*f2_v + 2*f3_v + f4_v)
        
        if x <= x_topo_rampa:
            x = x_topo_rampa
            xs.append(x)
            break
            
        xs.append(x)

    n_frames = 90
    idxs = np.linspace(0, len(xs)-1, n_frames, dtype=int)
    xs_f = [xs[i] for i in idxs]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.68, 0.32], horizontal_spacing=0.08)

    xr = np.linspace(x_topo_rampa, x_base_rampa, 30)
    yr = h_max * ((xr - x_base_rampa) / (x_topo_rampa - x_base_rampa))**2
    
    xp = np.concatenate([xr, np.linspace(x_base_rampa, x_parede, 40)])
    yp = np.concatenate([yr, np.zeros(40)])
    fig.add_trace(go.Scatter(x=xp, y=yp, mode='lines', line=dict(color='#7f8c8d', width=4), hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[x_parede, x_parede], y=[0, 1.8], mode='lines', line=dict(color='#95a5a6', width=6), hoverinfo='skip'), row=1, col=1)

    y_ini = a_r * (xs_f[0] - x_base_rampa)**2 if xs_f[0] < x_base_rampa else 0
    bx_ini, by_ini = criar_bloco(xs_f[0], y_ini, 0.7, 0.7)
    
    ponto_inicio_mola = max(x_inicio_mola, xs_f[0])
    xm_ini, ym_ini = criar_mola(ponto_inicio_mola, x_parede, 0.35, 12)

    fig.add_trace(go.Scatter(x=xm_ini, y=ym_ini, mode='lines', line=dict(color='#2ecc71', width=3), hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=bx_ini, y=by_ini, fill="toself", fillcolor="#e74c3c", line=dict(color="#c0392b", width=2), hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Bar(x=['Ec', 'Epg', 'Epe', 'Em'], y=[0.0, em_total, 0.0, em_total], marker_color=[COR_EC, COR_EPG, COR_EPE, COR_EM], text=[f"0.0J", f"{em_total:.1f}J", f"0.0J", f"{em_total:.1f}J"], textposition='auto'), row=1, col=2)

    frames = []
    for i, x_a in enumerate(xs_f):
        if x_a < x_base_rampa:
            y_a = a_r * (x_a - x_base_rampa)**2
            epg = massa * gravidade * y_a
            epe = 0.0
        elif x_a <= x_inicio_mola:
            y_a = 0.0
            epg = 0.0
            epe = 0.0
        else:
            y_a = 0.0
            epg = 0.0
            comp = x_a - x_inicio_mola
            epe = 0.5 * k_mola * (comp**2)

        ec = max(0.0, em_total - epg - epe)
        ponto_inicio_mola = max(x_inicio_mola, x_a)
        xm_f, ym_f = criar_mola(ponto_inicio_mola, x_parede, 0.35, 12)
        bx_f, by_f = criar_bloco(x_a, y_a, 0.7, 0.7)

        frames.append(go.Frame(
            data=[
                go.Scatter(x=xm_f, y=ym_f),
                go.Scatter(x=bx_f, y=by_f),
                go.Bar(y=[ec, epg, epe, em_total], text=[f"{ec:.1f}J", f"{epg:.1f}J", f"{epe:.1f}J", f"{em_total:.1f}J"])
            ],
            traces=[2, 3, 4],
            name=f"f{i}"
        ))

    fig.frames = frames
    fig.update_layout(
        showlegend=False, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=10, r=10, t=40, b=10), height=400,
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.0, "y": 1.15,
            "buttons": [
                {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": duracao_ms, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}, "mode": "immediate"}]},
                {"label": "❚❚ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]}
            ]
        }]
    )
    fig.update_xaxes(range=[x_topo_rampa - 1, x_parede + 1], showgrid=False, zeroline=False, visible=False, row=1, col=1)
    fig.update_yaxes(range=[-0.5, h_max + 1.5], showgrid=False, zeroline=False, visible=False, row=1, col=1)
    fig.update_yaxes(range=[0, max(10.0, em_total * 1.2)], title="Energia (Joules)", row=1, col=2)
    return fig, x_max_comp


# 4. Looping
def gerar_figura_looping_corrigido(massa, raio_loop, v_inicial, duracao_ms, gravidade=10):
    em_total = 0.5 * massa * (v_inicial**2) 
    v_min_topo = math.sqrt(raio_loop * gravidade)
    topo_loop_y = 2 * raio_loop
    
    h_max_energia = (v_inicial**2) / (2 * gravidade)
    consegue_passar = h_max_energia >= topo_loop_y

    fig = make_subplots(rows=1, cols=2, column_widths=[0.68, 0.32], horizontal_spacing=0.08)

    x_linha = np.linspace(-5.0, raio_loop, 40)
    y_linha = np.zeros(40)

    theta_loop = np.linspace(np.pi, -np.pi, 120)
    x_loop = raio_loop + raio_loop * np.sin(theta_loop)
    y_loop = raio_loop + raio_loop * np.cos(theta_loop)

    fig.add_trace(go.Scatter(x=np.concatenate([x_linha, x_loop]), y=np.concatenate([y_linha, y_loop]), mode='lines', line=dict(color='#7f8c8d', width=4), hoverinfo='skip'), row=1, col=1)

    cx_ini, cy_ini = criar_bloco(-5.0, 0.0, 0.6, 0.6)
    fig.add_trace(go.Scatter(x=cx_ini, y=cy_ini, fill="toself", fillcolor="#e74c3c", line=dict(color="#c0392b", width=2), hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Bar(x=['Ec', 'Epg', 'Em'], y=[em_total, 0.0, em_total], marker_color=[COR_EC, COR_EPG, COR_EM], text=[f"{em_total:.1f}J", f"0.0J", f"{em_total:.1f}J"], textposition='auto'), row=1, col=2)

    passos_x = np.linspace(-5.0, raio_loop, 30)
    
    if consegue_passar:
        passos_th = np.linspace(np.pi, -np.pi, 60)
        passos_saida = np.linspace(raio_loop, raio_loop + 4.0, 30)
        trajetoria = []
        for x in passos_x:
            trajetoria.append((x, 0.0, "linha"))
        for th in passos_th:
            xa = raio_loop + raio_loop * math.sin(th)
            ya = raio_loop + raio_loop * math.cos(th)
            trajetoria.append((xa, ya, "loop"))
        for x in passos_saida:
            trajetoria.append((x, 0.0, "saida"))
    else:
        cos_limite = min(1.0, max(-1.0, (h_max_energia / raio_loop) - 1))
        theta_limite = math.acos(cos_limite)
        passos_th_ida = np.linspace(np.pi, np.pi - theta_limite, 30)
        passos_th_volta = np.linspace(np.pi - theta_limite, np.pi, 30)
        passos_volta_linha = np.linspace(raio_loop, -5.0, 30)
        
        trajetoria = []
        for x in passos_x:
            trajetoria.append((x, 0.0, "linha"))
        for th in passos_th_ida:
            xa = raio_loop + raio_loop * math.sin(th)
            ya = raio_loop + raio_loop * math.cos(th)
            trajetoria.append((xa, ya, "loop"))
        for th in passos_th_volta:
            xa = raio_loop + raio_loop * math.sin(th)
            ya = raio_loop + raio_loop * math.cos(th)
            trajetoria.append((xa, ya, "loop"))
        for x in passos_volta_linha:
            trajetoria.append((x, 0.0, "linha"))

    n_frames = 90
    idxs = np.linspace(0, len(trajetoria)-1, n_frames, dtype=int)
    traj_f = [trajetoria[i] for i in idxs]

    frames = []
    for i, (xa, ya, tipo) in enumerate(traj_f):
        epg = massa * gravidade * ya
        ec = max(0.0, em_total - epg)
        cx_f, cy_f = criar_bloco(xa, ya, 0.6, 0.6)

        frames.append(go.Frame(
            data=[
                go.Scatter(x=cx_f, y=cy_f),
                go.Bar(y=[ec, epg, em_total], text=[f"{ec:.1f}J", f"{epg:.1f}J", f"{em_total:.1f}J"])
            ],
            traces=[1, 2],
            name=f"f{i}"
        ))

    fig.frames = frames
    fig.update_layout(
        showlegend=False, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=10, r=10, t=40, b=10), height=400,
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.0, "y": 1.15,
            "buttons": [
                {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": duracao_ms, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}, "mode": "immediate"}]},
                {"label": "❚❚ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]}
            ]
        }]
    )
    fig.update_xaxes(range=[-6, raio_loop * 4], showgrid=False, zeroline=False, visible=False, row=1, col=1)
    fig.update_yaxes(range=[-1, max(4.0, (2 * raio_loop) + 1.5)], showgrid=False, zeroline=False, visible=False, row=1, col=1)
    fig.update_yaxes(range=[0, max(10.0, em_total * 1.2)], title="Energia (Joules)", row=1, col=2)
    return fig, h_max_energia, v_min_topo, consegue_passar


# ============================================
# TÍTULO E NAVEGAÇÃO POR ABAS SUPERIORES
# ============================================
st.markdown('<div class="main-title">⚡ Sistemas Conservativos e Dinâmica</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Simulações físicas completas com integração numérica RK4 e conservação de energia</div>', unsafe_allow_html=True)

if 'velocidade_ms' not in st.session_state:
    st.session_state.velocidade_ms = 30

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Rampa em 'U' (Gravitacional)", 
    "2. Sistema Massa-Mola (Elástica)",
    "3. Rampa Inclinada + Mola",
    "4. Brinquedo Looping"
])

# ============================================
# ABA 1: RAMPA EM U
# ============================================
with tab1:
    st.markdown("""
    <div class="concept-card" style="border-left-color: #9b59b6;">
        <b>Princípio e Equações (RK4):</b> Esfera oscilando livremente em pista parabólica sem atrito. O núcleo integra numericamente as equações de movimento, garantindo que a esfera acelere corretamente no fundo e pare nos pontos de retorno.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(r"""
    $$ E_m = E_c + E_{pg} = \frac{1}{2}mv^2 + mgh = \text{constante} $$
    """)
    
    col_c1, col_c2 = st.columns([1, 2.5])
    with col_c1:
        st.markdown("<div class='param-box'>", unsafe_allow_html=True)
        massa_u = st.slider("Massa da esfera (kg)", 1.0, 10.0, 2.0, step=0.5, key='mu')
        h_max_u = st.slider("Altura inicial (m)", 2.0, 10.0, 5.0, step=0.5, key='hu')
        
        col_b1, col_b2 = st.columns(2)
        if col_b1.button("⏩ Mais Rápido", key='fast_u'):
            st.session_state.velocidade_ms = max(5, st.session_state.velocidade_ms - 10)
        if col_b2.button("⏪ Mais Lento", key='slow_u'):
            st.session_state.velocidade_ms = min(100, st.session_state.velocidade_ms + 10)
        st.markdown(f"<b>Velocidade atual:</b> {st.session_state.velocidade_ms} ms/quadro", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_c2:
        fig_u, periodo_u = gerar_figura_rampa_u(massa_u, h_max_u, st.session_state.velocidade_ms)
        st.plotly_chart(fig_u, use_container_width=True, config={'displayModeBar': False})
        st.metric("Período de Oscilação Estimado ($T$)", f"{periodo_u:.2f} s")

# ============================================
# ABA 2: SISTEMA MASSA-MOLA
# ============================================
with tab2:
    st.markdown("""
    <div class="concept-card" style="border-left-color: #2ecc71;">
        <b>Princípio e Equações (RK4):</b> Bloco oscilando horizontalmente com mola elástica. A aceleração varia linearmente com o deslocamento ($x'' = -\frac{k}{m}x$).
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(r"""
    $$ E_m = E_c + E_{pe} = \frac{1}{2}mv^2 + \frac{1}{2}kx^2 = \text{constante} $$
    """)
    
    col_m1, col_m2 = st.columns([1, 2.5])
    with col_m1:
        st.markdown("<div class='param-box'>", unsafe_allow_html=True)
        massa_m = st.slider("Massa do bloco (kg)", 1.0, 10.0, 2.0, step=0.5, key='mm')
        k_m = st.slider("Constante elástica (N/m)", 10, 100, 50, step=10, key='km')
        amp_m = st.slider("Amplitude (m)", 1.0, 5.0, 3.0, step=0.5, key='ampm')
        
        col_bm1, col_bm2 = st.columns(2)
        if col_bm1.button("⏩ Mais Rápido", key='fast_m'):
            st.session_state.velocidade_ms = max(5, st.session_state.velocidade_ms - 10)
        if col_bm2.button("⏪ Mais Lento", key='slow_m'):
            st.session_state.velocidade_ms = min(100, st.session_state.velocidade_ms + 10)
        st.markdown(f"<b>Velocidade atual:</b> {st.session_state.velocidade_ms} ms/quadro", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_m2:
        fig_m, periodo_m = gerar_figura_massa_mola(massa_m, k_m, amp_m, st.session_state.velocidade_ms)
        st.plotly_chart(fig_m, use_container_width=True, config={'displayModeBar': False})
        st.metric("Período Analítico ($T = 2\pi\sqrt{m/k}$)", f"{periodo_m:.2f} s")

# ============================================
# ABA 3: RAMPA + MOLA
# ============================================
with tab3:
    st.markdown("""
    <div class="concept-card" style="border-left-color: #3498db;">
        <b>Princípio e Equações:</b> Bloco solto do alto da rampa ($h$). A energia potencial gravitacional converte-se totalmente em energia elástica na compressão máxima da mola.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(r"""
    $$ mgh = \frac{1}{2}k x_{\text{max}}^2 \implies x_{\text{max}} = \sqrt{\frac{2mgh}{k}} $$
    """)
    
    col_r1, col_r2 = st.columns([1, 2.5])
    with col_r1:
        st.markdown("<div class='param-box'>", unsafe_allow_html=True)
        massa_rm = st.slider("Massa do bloco (m)", 1.0, 10.0, 2.0, step=0.5, key='m_rm')
        h_rm = st.slider("Altura inicial (h)", 1.0, 8.0, 4.0, step=0.5, key='h_rm')
        k_rm = st.slider("Constante da mola (k)", 20, 200, 100, step=10, key='k_rm')
        comprimento_mola = st.slider("Comprimento da mola (m)", 1.0, 4.0, 2.5, step=0.5, key='l_mola')
        
        col_brm1, col_brm2 = st.columns(2)
        if col_brm1.button("⏩ Mais Rápido", key='fast_rm'):
            st.session_state.velocidade_ms = max(5, st.session_state.velocidade_ms - 10)
        if col_brm2.button("⏪ Mais Lento", key='slow_rm'):
            st.session_state.velocidade_ms = min(100, st.session_state.velocidade_ms + 10)
        st.markdown(f"<b>Velocidade atual:</b> {st.session_state.velocidade_ms} ms/quadro", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r2:
        fig_rm, x_max_comp = gerar_figura_rampa_mola_ref(massa_rm, h_rm, k_rm, comprimento_mola, st.session_state.velocidade_ms)
        st.plotly_chart(fig_rm, use_container_width=True, config={'displayModeBar': False})
        st.metric("Compressão Máxima da Mola ($x_{max}$)", f"{x_max_comp:.2f} m")

# ============================================
# ABA 4: BRINQUEDO LOOPING
# ============================================
with tab4:
    st.markdown("""
    <div class="concept-card" style="border-left-color: #e74c3c;">
        <b>Princípio e Equações:</b> O carrinho inicia em linha reta com velocidade inicial $v_0$. A simulação verifica se há energia suficiente para transpor o topo do looping, garantindo conservação rigorosa.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(r"""
    $$ \frac{1}{2}mv_0^2 = \frac{1}{2}mv_{\text{topo}}^2 + mg(2R) \quad \text{e} \quad v_{\text{topo}} \ge \sqrt{Rg} $$
    """)
    
    col_l1, col_l2 = st.columns([1, 2.5])
    with col_l1:
        st.markdown("<div class='param-box'>", unsafe_allow_html=True)
        massa_l = st.slider("Massa do carrinho (kg)", 0.1, 5.0, 1.0, step=0.1, key='m_l')
        raio_l = st.slider("Raio do Looping (R)", 0.5, 3.0, 1.0, step=0.25, key='r_l')
        v_ini_l = st.slider("Velocidade inicial ($v_0$)", 1.0, 15.0, 6.0, step=0.5, key='v_ini_l')
        
        col_bl1, col_bl2 = st.columns(2)
        if col_bl1.button("⏩ Mais Rápido", key='fast_l'):
            st.session_state.velocidade_ms = max(5, st.session_state.velocidade_ms - 10)
        if col_bl2.button("⏪ Mais Lento", key='slow_l'):
            st.session_state.velocidade_ms = min(100, st.session_state.velocidade_ms + 10)
        st.markdown(f"<b>Velocidade atual:</b> {st.session_state.velocidade_ms} ms/quadro", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_l2:
        fig_l, h_max_eng, v_min_topo, viavel = gerar_figura_looping_corrigido(massa_l, raio_l, v_ini_l, st.session_state.velocidade_ms)
        st.plotly_chart(fig_l, use_container_width=True, config={'displayModeBar': False})
        
        st.subheader("📊 Relatório de Viabilidade do Looping")
        col_r1, col_r2 = st.columns(2)
        col_r1.metric("Altura Máxima Atingível ($h_{max}$)", f"{h_max_eng:.2f} m")
        col_r2.metric("Altura do Topo do Loop", f"{2 * raio_l:.2f} m")
        
        if viavel:
            st.success("✅ **Trajetória Viável!** O carrinho possui energia suficiente para transpor o looping completo.")
        else:
            st.error("❌ **Trajetória Inviável!** A velocidade inicial é insuficiente para o raio escolhido.")

# Rodapé
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.85rem; padding: 1rem;">
    ⚡ <b>Física Visual: Energia e Dinâmica</b> — Simulação com integração numérica RK4 otimizada.
</div>
""", unsafe_allow_html=True)
