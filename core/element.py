"""Elemento de pórtico 2D (viga-columna Euler-Bernoulli)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from core.node import Node


@dataclass
class Element:
    """
    Elemento estructural entre dos nodos.

    Atributos:
        node_i, node_j: nodos extremo.
        E: módulo de elasticidad.
        A: área de sección transversal.
        I: momento de inercia.
        label: etiqueta opcional.
    """

    node_i: Node
    node_j: Node
    E: float
    A: float
    I: float
    label: str = ""

    # ---------- Propiedades geométricas ----------
    @property
    def length(self) -> float:
        """Longitud del elemento (L)."""
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        return math.hypot(dx, dy)

    @property
    def angle(self) -> float:
        """Ángulo en radianes desde el eje X global hasta el eje del elemento."""
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        return math.atan2(dy, dx)

    @property
    def cos_sin(self):
        a = self.angle
        return math.cos(a), math.sin(a)

    # ---------- Paso 1: matriz local k' ----------
    def k_local(self) -> np.ndarray:
        """
        Matriz de rigidez del elemento en coordenadas locales (6x6).

        GDL locales en orden: [u_i', v_i', theta_i, u_j', v_j', theta_j]
            - u'  : desplazamiento axial (a lo largo del elemento)
            - v'  : desplazamiento transversal
            - theta: rotación

        Combina rigidez axial (EA/L) y flexional (Euler-Bernoulli, EI/L^3).
        """
        E, A, I, L = self.E, self.A, self.I, self.length
        EA_L = E * A / L
        EI = E * I
        L2 = L * L
        L3 = L2 * L

        k = np.array(
            [
                [ EA_L,         0,            0,         -EA_L,        0,            0       ],
                [ 0,        12 * EI / L3,   6 * EI / L2,  0,       -12 * EI / L3,   6 * EI / L2],
                [ 0,         6 * EI / L2,   4 * EI / L,   0,        -6 * EI / L2,   2 * EI / L ],
                [-EA_L,         0,            0,          EA_L,        0,            0       ],
                [ 0,       -12 * EI / L3,  -6 * EI / L2,  0,        12 * EI / L3,  -6 * EI / L2],
                [ 0,         6 * EI / L2,   2 * EI / L,   0,        -6 * EI / L2,   4 * EI / L ],
            ],
            dtype=float,
        )
        return k

    # ---------- Paso 2: matriz de transformación T ----------
    def T(self) -> np.ndarray:
        """
        Matriz de transformación de coordenadas globales a locales (6x6).

        Rota el vector de desplazamientos del nodo desde el sistema global
        (u, v, theta) al sistema local del elemento (u', v', theta).
        La rotación theta no cambia con la rotación del sistema (escalar).
        """
        c, s = self.cos_sin
        T = np.array(
            [
                [ c,  s, 0,  0,  0, 0],
                [-s,  c, 0,  0,  0, 0],
                [ 0,  0, 1,  0,  0, 0],
                [ 0,  0, 0,  c,  s, 0],
                [ 0,  0, 0, -s,  c, 0],
                [ 0,  0, 0,  0,  0, 1],
            ],
            dtype=float,
        )
        return T

    # ---------- Paso 3: matriz global del elemento k = T^T k' T ----------
    def k_global(self) -> np.ndarray:
        """Matriz de rigidez del elemento expresada en coordenadas globales."""
        T = self.T()
        return T.T @ self.k_local() @ T

    # ---------- Paso 7: fuerzas internas en coordenadas locales ----------
    def internal_forces(self, u_global_element: np.ndarray) -> np.ndarray:
        """
        Calcula el vector de fuerzas internas del elemento en coordenadas LOCALES.

        Parámetros:
            u_global_element: vector 6x1 de desplazamientos globales del elemento
                              [u_i, v_i, theta_i, u_j, v_j, theta_j].

        Retorna:
            f_local: vector 6x1 [N_i, V_i, M_i, N_j, V_j, M_j] en sistema local.
                     N positivo = tracción en extremo j (compresión en i).
        """
        u_local = self.T() @ u_global_element
        return self.k_local() @ u_local
