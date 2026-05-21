"""
Interfaz Streamlit para el análisis de pórticos 2D.

Ejecución:
    streamlit run frame_solver/ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permite ejecutar `streamlit run frame_solver/ui/app.py` desde la raíz del repo.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from frame_solver.core.element import Element
from frame_solver.core.node import Node
from frame_solver.core.solver import Frame2DSolver


# ----------------------------------------------------------------------
# Ejemplo precargado: pórtico simple de 3 barras (2 columnas + 1 viga)
# ----------------------------------------------------------------------
def default_nodes_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # x, y, restricciones, cargas
            {"label": "N1", "x": 0.0, "y": 0.0,
             "fixed_u": True,  "fixed_v": True,  "fixed_theta": True,
             "fx": 0.0, "fy": 0.0, "m": 0.0},
            {"label": "N2", "x": 0.0, "y": 3.0,
             "fixed_u": False, "fixed_v": False, "fixed_theta": False,
             "fx": 10.0, "fy": 0.0, "m": 0.0},
            {"label": "N3", "x": 4.0, "y": 3.0,
             "fixed_u": False, "fixed_v": False, "fixed_theta": False,
             "fx": 0.0, "fy": -5.0, "m": 0.0},
            {"label": "N4", "x": 4.0, "y": 0.0,
             "fixed_u": True,  "fixed_v": True,  "fixed_theta": True,
             "fx": 0.0, "fy": 0.0, "m": 0.0},
        ]
    )


def default_elements_df() -> pd.DataFrame:
    # E y A típicos para acero / sección genérica.
    # E = 2.1e8 kN/m^2, A = 0.01 m^2, I = 8.33e-5 m^4 (perfil rectangular 10x10 cm)
    return pd.DataFrame(
        [
            {"label": "B1", "node_i": "N1", "node_j": "N2",
             "E": 2.1e8, "A": 0.01, "I": 8.33e-5},
            {"label": "B2", "node_i": "N2", "node_j": "N3",
             "E": 2.1e8, "A": 0.01, "I": 8.33e-5},
            {"label": "B3", "node_i": "N3", "node_j": "N4",
             "E": 2.1e8, "A": 0.01, "I": 8.33e-5},
        ]
    )


# ----------------------------------------------------------------------
# Construcción del modelo desde los DataFrames editados
# ----------------------------------------------------------------------
def build_model(nodes_df: pd.DataFrame, elements_df: pd.DataFrame):
    nodes_df = nodes_df.dropna(subset=["x", "y"]).reset_index(drop=True)
    elements_df = elements_df.dropna(subset=["node_i", "node_j"]).reset_index(drop=True)

    nodes: list[Node] = []
    label_to_index: dict[str, int] = {}
    for i, row in nodes_df.iterrows():
        node = Node(
            x=float(row["x"]),
            y=float(row["y"]),
            fixed_u=bool(row["fixed_u"]),
            fixed_v=bool(row["fixed_v"]),
            fixed_theta=bool(row["fixed_theta"]),
            fx=float(row.get("fx", 0.0) or 0.0),
            fy=float(row.get("fy", 0.0) or 0.0),
            m=float(row.get("m", 0.0) or 0.0),
            label=str(row.get("label", f"N{i+1}") or f"N{i+1}"),
        )
        nodes.append(node)
        label_to_index[node.label] = i

    elements: list[Element] = []
    for k, row in elements_df.iterrows():
        ni_label = str(row["node_i"])
        nj_label = str(row["node_j"])
        if ni_label not in label_to_index or nj_label not in label_to_index:
            raise ValueError(
                f"Elemento {row.get('label', k)}: etiquetas de nodos "
                f"'{ni_label}'/'{nj_label}' no existen en la tabla de nodos."
            )
        elements.append(
            Element(
                node_i=nodes[label_to_index[ni_label]],
                node_j=nodes[label_to_index[nj_label]],
                E=float(row["E"]),
                A=float(row["A"]),
                I=float(row["I"]),
                label=str(row.get("label", f"B{k+1}") or f"B{k+1}"),
            )
        )

    return nodes, elements


# ----------------------------------------------------------------------
# Visualización con Matplotlib
# ----------------------------------------------------------------------
def plot_structure(nodes, elements, results=None, scale: float = 100.0):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Estructura original (líneas grises)
    for elem in elements:
        ax.plot(
            [elem.node_i.x, elem.node_j.x],
            [elem.node_i.y, elem.node_j.y],
            color="#888", lw=2, zorder=1,
        )

    # Deformada (líneas azules)
    if results is not None:
        for idx_e, elem in enumerate(elements):
            i = next(i for i, n in enumerate(nodes) if n is elem.node_i)
            j = next(j for j, n in enumerate(nodes) if n is elem.node_j)
            ui = results.displacements[i]
            uj = results.displacements[j]
            xi = elem.node_i.x + scale * ui[0]
            yi = elem.node_i.y + scale * ui[1]
            xj = elem.node_j.x + scale * uj[0]
            yj = elem.node_j.y + scale * uj[1]
            ax.plot([xi, xj], [yi, yj], color="#1f77b4", lw=2, zorder=2,
                    label="Deformada" if idx_e == 0 else None)

    # Nodos
    for idx, node in enumerate(nodes):
        ax.scatter(node.x, node.y, s=60, color="black", zorder=3)
        ax.annotate(
            node.label or f"N{idx+1}",
            (node.x, node.y),
            textcoords="offset points", xytext=(8, 8), fontsize=10,
        )

        # Apoyos (símbolos básicos)
        if node.fixed_u and node.fixed_v and node.fixed_theta:
            ax.scatter(node.x, node.y, marker="s", s=200,
                       facecolors="none", edgecolors="red", lw=2, zorder=2)
        elif node.fixed_u and node.fixed_v:
            ax.scatter(node.x, node.y, marker="^", s=200,
                       facecolors="none", edgecolors="red", lw=2, zorder=2)
        elif node.fixed_v:
            ax.scatter(node.x, node.y, marker="o", s=200,
                       facecolors="none", edgecolors="red", lw=2, zorder=2)

    # Cargas aplicadas (flechas)
    max_coord = max(
        max((n.x for n in nodes), default=1),
        max((n.y for n in nodes), default=1),
        1.0,
    )
    arrow_scale = max_coord * 0.05
    max_force = max(
        (max(abs(n.fx), abs(n.fy)) for n in nodes), default=1.0
    ) or 1.0
    for node in nodes:
        if node.fx != 0:
            dx = arrow_scale * (node.fx / max_force) * 3
            ax.annotate(
                "", xy=(node.x, node.y),
                xytext=(node.x - dx, node.y),
                arrowprops=dict(arrowstyle="->", color="green", lw=2),
            )
        if node.fy != 0:
            dy = arrow_scale * (node.fy / max_force) * 3
            ax.annotate(
                "", xy=(node.x, node.y),
                xytext=(node.x, node.y - dy),
                arrowprops=dict(arrowstyle="->", color="green", lw=2),
            )

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Pórtico 2D" + ("  —  con deformada (escalada)" if results else ""))
    if results is not None:
        ax.legend(loc="best")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Streamlit app
# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Pórtico 2D — Método de la Rigidez",
                       layout="wide")

    st.title("Análisis de Pórticos 2D — Método Matricial de la Rigidez")
    st.caption(
        "Edite la tabla de nodos, elementos y cargas. Apriete **Calcular** "
        "para ejecutar el solver. El ejemplo precargado es un pórtico simple "
        "de 3 barras."
    )

    # ----- Estado inicial -----
    if "nodes_df" not in st.session_state:
        st.session_state["nodes_df"] = default_nodes_df()
    if "elements_df" not in st.session_state:
        st.session_state["elements_df"] = default_elements_df()

    # ----- Entrada -----
    col_in1, col_in2 = st.columns(2)

    with col_in1:
        st.subheader("Nodos")
        st.caption("Coordenadas, apoyos (`fixed_*` = True restringe) y cargas (fx, fy, m).")
        st.session_state["nodes_df"] = st.data_editor(
            st.session_state["nodes_df"],
            num_rows="dynamic",
            use_container_width=True,
            key="nodes_editor",
        )

    with col_in2:
        st.subheader("Elementos (barras)")
        st.caption("Cada barra con sus propiedades **E, A, I** independientes.")
        st.session_state["elements_df"] = st.data_editor(
            st.session_state["elements_df"],
            num_rows="dynamic",
            use_container_width=True,
            key="elements_editor",
        )

    st.divider()
    cols_btn = st.columns([1, 1, 1, 3])
    calc = cols_btn[0].button("Calcular", type="primary")
    if cols_btn[1].button("Restablecer ejemplo"):
        st.session_state["nodes_df"] = default_nodes_df()
        st.session_state["elements_df"] = default_elements_df()
        st.rerun()
    scale_factor = cols_btn[2].number_input(
        "Escala deformada", min_value=1, max_value=100000, value=100, step=10
    )

    if not calc:
        st.info("Edite los datos y presione **Calcular**.")
        st.pyplot(plot_structure(
            *build_model(st.session_state["nodes_df"], st.session_state["elements_df"])
        ))
        return

    # ----- Construcción del modelo y solver -----
    try:
        nodes, elements = build_model(
            st.session_state["nodes_df"], st.session_state["elements_df"]
        )
    except Exception as exc:
        st.error(f"Error en los datos de entrada: {exc}")
        return

    try:
        solver = Frame2DSolver(nodes, elements)
        results = solver.solve()
    except Exception as exc:
        st.error(f"Error en el cálculo: {exc}")
        return

    # ----- Visualización -----
    st.subheader("Visualización")
    st.pyplot(plot_structure(nodes, elements, results=results, scale=scale_factor))

    # ----- Resultados: desplazamientos -----
    st.subheader("Desplazamientos nodales")
    disp_rows = []
    for idx, node in enumerate(nodes):
        u, v, th = results.displacements[idx]
        disp_rows.append({
            "Nodo": node.label or f"N{idx+1}",
            "u (X)": u, "v (Y)": v, "θ (rot)": th,
        })
    st.dataframe(pd.DataFrame(disp_rows), use_container_width=True)

    # ----- Resultados: reacciones -----
    st.subheader("Reacciones en apoyos")
    react_rows = []
    for idx, node in enumerate(nodes):
        if any(node.restraints()):
            Rx, Ry, M = results.reactions[idx]
            react_rows.append({
                "Nodo": node.label or f"N{idx+1}",
                "Rx": Rx, "Ry": Ry, "M (reacción)": M,
            })
    if react_rows:
        st.dataframe(pd.DataFrame(react_rows), use_container_width=True)
    else:
        st.warning("No se detectaron apoyos restringidos.")

    # ----- Fuerzas internas por elemento -----
    st.subheader("Fuerzas internas por elemento")
    f_rows = []
    for idx, elem in enumerate(elements):
        N_i, V_i, M_i, N_j, V_j, M_j = results.element_forces_local[idx]
        N = results.element_axial[idx]
        f_rows.append({
            "Barra": elem.label or f"B{idx+1}",
            "L": elem.length,
            "N_i": N_i, "V_i": V_i, "M_i": M_i,
            "N_j": N_j, "V_j": V_j, "M_j": M_j,
            "Estado axial": Frame2DSolver.axial_state(N),
        })
    st.dataframe(pd.DataFrame(f_rows), use_container_width=True)

    # ----- Matrices (desplegable) -----
    with st.expander("Matrices intermedias del cálculo"):
        st.markdown("#### Matriz de rigidez global K")
        st.dataframe(pd.DataFrame(results.K_global), use_container_width=True)

        st.markdown("#### Vector de fuerzas global F")
        st.dataframe(pd.DataFrame({"F": results.F_global}))

        st.markdown("#### GDL libres / restringidos")
        st.write({"libres": results.free_dofs, "restringidos": results.fixed_dofs})

        for idx, elem in enumerate(elements):
            st.markdown(f"#### Elemento {elem.label or f'B{idx+1}'}")
            cols = st.columns(3)
            cols[0].markdown("**k' (local)**")
            cols[0].dataframe(pd.DataFrame(results.element_k_local[idx]))
            cols[1].markdown("**T (transformación)**")
            cols[1].dataframe(pd.DataFrame(results.element_T[idx]))
            cols[2].markdown("**k (global)**")
            cols[2].dataframe(pd.DataFrame(results.element_k_global[idx]))


if __name__ == "__main__":
    main()
