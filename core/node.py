"""Nodo de un pórtico 2D — 3 grados de libertad: u, v, theta."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Node:
    """
    Representa un nodo del pórtico.

    Atributos:
        x, y: coordenadas en el sistema global.
        fixed_u, fixed_v, fixed_theta: True si el GDL está restringido (apoyo).
        fx, fy, m: cargas nodales aplicadas (fuerza horizontal, vertical, momento).
        label: etiqueta opcional (p.ej. "N1").
    """

    x: float
    y: float
    fixed_u: bool = False
    fixed_v: bool = False
    fixed_theta: bool = False
    fx: float = 0.0
    fy: float = 0.0
    m: float = 0.0
    label: str = ""

    def restraints(self) -> Tuple[bool, bool, bool]:
        return (self.fixed_u, self.fixed_v, self.fixed_theta)

    def loads(self) -> Tuple[float, float, float]:
        return (self.fx, self.fy, self.m)
