"""
Tests del solver de pórticos 2D.

Caso de prueba principal: pórtico simple de 3 barras (2 columnas + 1 viga),
empotrado en ambas bases, con cargas horizontales y verticales.
"""

import math
import os
import sys

import numpy as np
import pytest

# Agrega la raíz del repo a sys.path para que `from core.*` funcione.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.element import Element
from core.node import Node
from core.solver import Frame2DSolver


# ----------------------------------------------------------------------
# Construcción del caso de prueba
# ----------------------------------------------------------------------
def _portal_frame():
    """Pórtico 4 nodos, 3 barras, empotrado en N1 y N4. Carga lateral en N2."""
    E = 2.1e8       # kN/m^2  (acero)
    A = 0.01        # m^2
    I = 8.33e-5     # m^4

    n1 = Node(0.0, 0.0, fixed_u=True, fixed_v=True, fixed_theta=True, label="N1")
    n2 = Node(0.0, 3.0, fx=10.0, label="N2")
    n3 = Node(4.0, 3.0, fy=-5.0, label="N3")
    n4 = Node(4.0, 0.0, fixed_u=True, fixed_v=True, fixed_theta=True, label="N4")

    e1 = Element(n1, n2, E=E, A=A, I=I, label="B1")  # columna izq
    e2 = Element(n2, n3, E=E, A=A, I=I, label="B2")  # viga
    e3 = Element(n3, n4, E=E, A=A, I=I, label="B3")  # columna der

    return [n1, n2, n3, n4], [e1, e2, e3]


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_element_length_and_angle():
    n1 = Node(0.0, 0.0)
    n2 = Node(3.0, 4.0)
    e = Element(n1, n2, E=1.0, A=1.0, I=1.0)
    assert math.isclose(e.length, 5.0)
    assert math.isclose(e.angle, math.atan2(4.0, 3.0))


def test_k_local_is_symmetric():
    n1 = Node(0.0, 0.0)
    n2 = Node(3.0, 0.0)
    e = Element(n1, n2, E=2.1e8, A=0.01, I=8.33e-5)
    k = e.k_local()
    assert k.shape == (6, 6)
    assert np.allclose(k, k.T), "k_local debe ser simétrica"


def test_T_is_orthogonal():
    n1 = Node(0.0, 0.0)
    n2 = Node(3.0, 4.0)  # ángulo arbitrario
    e = Element(n1, n2, E=1.0, A=1.0, I=1.0)
    T = e.T()
    # T es ortogonal por bloques -> T T^T = I
    assert np.allclose(T @ T.T, np.eye(6), atol=1e-12)


def test_k_global_equivalent_when_horizontal():
    """Si el elemento es horizontal, k_global == k_local."""
    n1 = Node(0.0, 0.0)
    n2 = Node(5.0, 0.0)
    e = Element(n1, n2, E=2.1e8, A=0.01, I=8.33e-5)
    assert np.allclose(e.k_global(), e.k_local(), atol=1e-9)


def test_solver_reactions_equilibrium():
    """Las reacciones deben equilibrar a las cargas aplicadas."""
    nodes, elements = _portal_frame()
    solver = Frame2DSolver(nodes, elements)
    res = solver.solve()

    # Suma de cargas aplicadas
    Fx_app = sum(n.fx for n in nodes)
    Fy_app = sum(n.fy for n in nodes)

    # Suma de reacciones (solo en nodos restringidos)
    Rx_total = sum(res.reactions[i][0] for i, n in enumerate(nodes) if any(n.restraints()))
    Ry_total = sum(res.reactions[i][1] for i, n in enumerate(nodes) if any(n.restraints()))

    # Equilibrio: ΣF + ΣR = 0
    assert math.isclose(Fx_app + Rx_total, 0.0, abs_tol=1e-6), (
        f"Equilibrio en X falla: F={Fx_app}, R={Rx_total}"
    )
    assert math.isclose(Fy_app + Ry_total, 0.0, abs_tol=1e-6), (
        f"Equilibrio en Y falla: F={Fy_app}, R={Ry_total}"
    )


def test_solver_moment_equilibrium():
    """Equilibrio de momentos respecto al origen."""
    nodes, elements = _portal_frame()
    solver = Frame2DSolver(nodes, elements)
    res = solver.solve()

    # Momento de cargas aplicadas respecto al origen
    M_app = sum(n.x * n.fy - n.y * n.fx + n.m for n in nodes)

    # Momento de reacciones respecto al origen
    M_react = 0.0
    for i, n in enumerate(nodes):
        if any(n.restraints()):
            Rx, Ry, Mz = res.reactions[i]
            M_react += n.x * Ry - n.y * Rx + Mz

    assert math.isclose(M_app + M_react, 0.0, abs_tol=1e-6), (
        f"Equilibrio de momentos falla: M_app={M_app}, M_react={M_react}"
    )


def test_solver_fixed_dofs_have_zero_displacement():
    """Los GDL restringidos deben tener desplazamiento cero."""
    nodes, elements = _portal_frame()
    solver = Frame2DSolver(nodes, elements)
    res = solver.solve()
    for gdl in res.fixed_dofs:
        assert math.isclose(res.U[gdl], 0.0, abs_tol=1e-12)


def test_solver_lateral_load_produces_horizontal_displacement():
    """Con carga horizontal positiva en N2, el desplazamiento horizontal de N2 debe ser positivo."""
    nodes, elements = _portal_frame()
    solver = Frame2DSolver(nodes, elements)
    res = solver.solve()
    u_n2 = res.displacements[1][0]  # N2 índice 1, GDL u
    assert u_n2 > 0.0, f"u(N2) debería ser positivo, obtuvo {u_n2}"


def test_solver_K_is_symmetric():
    nodes, elements = _portal_frame()
    solver = Frame2DSolver(nodes, elements)
    res = solver.solve()
    assert np.allclose(res.K_global, res.K_global.T, atol=1e-6)


def test_singular_structure_raises():
    """Estructura sin apoyos suficientes -> matriz singular."""
    n1 = Node(0.0, 0.0)  # sin restricciones
    n2 = Node(3.0, 0.0)
    e = Element(n1, n2, E=1.0, A=1.0, I=1.0)
    solver = Frame2DSolver([n1, n2], [e])
    with pytest.raises(RuntimeError, match="singular"):
        solver.solve()


def test_axial_state_classification():
    assert Frame2DSolver.axial_state(10.0) == "Tracción"
    assert Frame2DSolver.axial_state(-10.0) == "Compresión"
    assert Frame2DSolver.axial_state(0.0) == "Nulo"
