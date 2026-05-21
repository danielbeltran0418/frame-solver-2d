"""Solver de pórticos 2D por el Método Matricial de la Rigidez."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from frame_solver.core.element import Element
from frame_solver.core.node import Node


# Cada nodo tiene 3 GDL: u (X), v (Y), theta (rotación).
DOF_PER_NODE = 3


@dataclass
class SolverResults:
    """Resultados completos del análisis del pórtico."""

    K_global: np.ndarray                 # Matriz de rigidez ensamblada
    F_global: np.ndarray                 # Vector de fuerzas global
    U: np.ndarray                        # Vector de desplazamientos completo
    R: np.ndarray                        # Vector de fuerzas (incluye reacciones)
    reactions: Dict[int, np.ndarray]     # node_index -> [Rx, Ry, M]
    displacements: Dict[int, np.ndarray] # node_index -> [u, v, theta]
    element_forces_local: List[np.ndarray]   # [N_i, V_i, M_i, N_j, V_j, M_j] por elemento
    element_axial: List[float]               # Esfuerzo axial (N positivo = tracción)
    free_dofs: List[int]
    fixed_dofs: List[int]
    element_k_local: List[np.ndarray]
    element_T: List[np.ndarray]
    element_k_global: List[np.ndarray]


class Frame2DSolver:
    """
    Implementa los pasos del Método Matricial de la Rigidez:

      1. Numeración de GDL (3 por nodo).
      2. k' local de cada elemento.
      3. T (matriz de transformación) de cada elemento.
      4. k_global del elemento = T^T k' T.
      5. Ensamblaje de K global.
      6. F global (cargas nodales).
      7. Partición y resolución: U_L = K_LL^{-1} (F_L - K_LR U_R).
      8. Reacciones y fuerzas internas.
    """

    def __init__(self, nodes: List[Node], elements: List[Element]):
        if len(nodes) < 2:
            raise ValueError("Se requieren al menos 2 nodos.")
        if len(elements) < 1:
            raise ValueError("Se requiere al menos 1 elemento.")
        self.nodes = nodes
        self.elements = elements
        self._node_index = {id(n): i for i, n in enumerate(nodes)}

    # ---------- Paso 1: numeración de GDL ----------
    def dof_indices(self, node_index: int) -> Tuple[int, int, int]:
        """GDL globales (índices) asociados a un nodo: (u, v, theta)."""
        base = node_index * DOF_PER_NODE
        return (base, base + 1, base + 2)

    def element_dof_map(self, element: Element) -> List[int]:
        """Mapea los 6 GDL locales del elemento a los GDL globales correspondientes."""
        i = self._node_index[id(element.node_i)]
        j = self._node_index[id(element.node_j)]
        return list(self.dof_indices(i)) + list(self.dof_indices(j))

    def classify_dofs(self) -> Tuple[List[int], List[int]]:
        """Devuelve (gdl_libres, gdl_restringidos) según las restricciones de los nodos."""
        free, fixed = [], []
        for idx, node in enumerate(self.nodes):
            base = idx * DOF_PER_NODE
            for offset, is_fixed in enumerate(node.restraints()):
                gdl = base + offset
                (fixed if is_fixed else free).append(gdl)
        return free, fixed

    # ---------- Pasos 2-4: matrices de elemento ----------
    def _element_matrices(self):
        """Calcula k_local, T y k_global para cada elemento."""
        k_locals = [e.k_local() for e in self.elements]
        Ts = [e.T() for e in self.elements]
        k_globals = [T.T @ kl @ T for kl, T in zip(k_locals, Ts)]
        return k_locals, Ts, k_globals

    # ---------- Paso 5: ensamblaje de K global ----------
    def assemble_K(self, k_globals_elem: List[np.ndarray]) -> np.ndarray:
        """
        Ensambla la matriz de rigidez global sumando las contribuciones de cada
        elemento en los GDL globales correspondientes.
        """
        n_dof = len(self.nodes) * DOF_PER_NODE
        K = np.zeros((n_dof, n_dof), dtype=float)
        for elem, k_g in zip(self.elements, k_globals_elem):
            dof_map = self.element_dof_map(elem)
            for a in range(6):
                for b in range(6):
                    K[dof_map[a], dof_map[b]] += k_g[a, b]
        return K

    # ---------- Paso 6: vector de fuerzas global ----------
    def assemble_F(self) -> np.ndarray:
        """Construye F a partir de las cargas nodales (fx, fy, m)."""
        n_dof = len(self.nodes) * DOF_PER_NODE
        F = np.zeros(n_dof, dtype=float)
        for idx, node in enumerate(self.nodes):
            base = idx * DOF_PER_NODE
            F[base]     = node.fx
            F[base + 1] = node.fy
            F[base + 2] = node.m
        return F

    # ---------- Paso 7-8: resolución completa ----------
    def solve(self) -> SolverResults:
        # 2-4. Matrices por elemento
        k_locals, Ts, k_globals_elem = self._element_matrices()

        # 5. K global
        K = self.assemble_K(k_globals_elem)

        # 6. F global
        F = self.assemble_F()

        # 1. Clasificación de GDL
        free, fixed = self.classify_dofs()

        # 7. Partición K = [[K_LL, K_LR], [K_RL, K_RR]] y resolución
        # Asumimos asentamientos prescritos U_R = 0.
        K_LL = K[np.ix_(free, free)]
        K_LR = K[np.ix_(free, fixed)]
        K_RL = K[np.ix_(fixed, free)]
        F_L  = F[free]

        # U_L = K_LL^{-1} F_L  (con U_R = 0)
        try:
            U_L = np.linalg.solve(K_LL, F_L)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "La matriz K_LL es singular. Verifique apoyos y estabilidad de la "
                "estructura (puede ser un mecanismo)."
            ) from exc

        n_dof = K.shape[0]
        U = np.zeros(n_dof, dtype=float)
        U[free] = U_L
        # U[fixed] = 0  (asentamientos nulos)

        # 8. Reacciones: R = K U - F_aplicado en GDL restringidos
        R_full = K @ U - F  # negativo de cargas externas en libres (debería ser ~0); reacciones en restringidos
        # En GDL libres por equilibrio R_full ≈ 0.
        reactions = {}
        for idx in range(len(self.nodes)):
            base = idx * DOF_PER_NODE
            reactions[idx] = R_full[base:base + 3].copy()

        # Desplazamientos por nodo
        displacements = {}
        for idx in range(len(self.nodes)):
            base = idx * DOF_PER_NODE
            displacements[idx] = U[base:base + 3].copy()

        # Fuerzas internas por elemento (en coordenadas locales)
        element_forces_local: List[np.ndarray] = []
        element_axial: List[float] = []
        for elem in self.elements:
            dof_map = self.element_dof_map(elem)
            u_elem_global = U[dof_map]
            f_local = elem.internal_forces(u_elem_global)
            element_forces_local.append(f_local)
            # Por convención, N positivo = tracción. En extremo j: N_j = f_local[3]
            # (axial positivo cuando el elemento se estira).
            element_axial.append(f_local[3])

        return SolverResults(
            K_global=K,
            F_global=F,
            U=U,
            R=R_full,
            reactions=reactions,
            displacements=displacements,
            element_forces_local=element_forces_local,
            element_axial=element_axial,
            free_dofs=free,
            fixed_dofs=fixed,
            element_k_local=k_locals,
            element_T=Ts,
            element_k_global=k_globals_elem,
        )

    # ---------- Utilidades ----------
    @staticmethod
    def axial_state(N: float, tol: float = 1e-6) -> str:
        """Devuelve 'Tracción', 'Compresión' o 'Nulo' según el signo del axial."""
        if abs(N) < tol:
            return "Nulo"
        return "Tracción" if N > 0 else "Compresión"
