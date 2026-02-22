"""
portals.py - классы объектов поля для симуляции потенциала.

Functions:
  Portal                базовый портал с маской и цветом
  FixedPotentialPortal  жёсткое граничное условие Дирихле (φ = const)
  CouplePortal          пара порталов с равным потенциалом
  PotentialAnchor       якорь фонового поля (φ = const)
  MaterialObject        твёрдый объект
  ConductorObject       твёрдый объект проводник (проводит потенциал)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from masks import Mask


class Portal:
    def __init__(self, mask: "Mask",
                 color: Tuple[int, int, int],
                 active: bool = True) -> None:
        self.mask   = mask
        self.color  = color
        self.active = active

    def __repr__(self) -> str:
        return f"Portal(mask={self.mask!r}, color={self.color})"

    def __contains__(self, point: Tuple[float, float]) -> bool:
        return self.active and (point in self.mask)

    def get_mask(self, X: "np.ndarray", Y: "np.ndarray") -> "np.ndarray":
        if self.active:
            return self.mask(X, Y)
        return np.zeros(X.shape, dtype=bool)


class FixedPotentialPortal(Portal):
    """φ = const (граничное условие Дирихле)"""

    def __init__(self, mask: "Mask",
                 potential_value: float,
                 color: Optional[Tuple[int, int, int]] = None,
                 active: bool = True) -> None:
        if color is None:
            color = (220, 50, 50) if potential_value >= 0.5 else (50, 100, 220)
        super().__init__(mask, color, active)
        self.potential_value = potential_value

    def __repr__(self) -> str:
        return f"FixedPotentialPortal(φ={self.potential_value}, mask={self.mask!r})"


@dataclass
class CouplePortal:
    """Пара порталов с одинаковым потенциалом"""
    p1: Portal
    p2: Portal

    def get_combined_mask(self, X: "np.ndarray",
                          Y: "np.ndarray") -> "np.ndarray":
        return self.p1.get_mask(X, Y) | self.p2.get_mask(X, Y)

    def __repr__(self) -> str:
        return f"CouplePortal(p1={self.p1!r}, p2={self.p2!r})"


class PotentialAnchor:
    """
    Якорь потенциала: область с фиксированным потенциалом

    Используется для задания фонового поля
    """

    def __init__(self, mask: "Mask",
                 potential_value: float,
                 color: Optional[Tuple[int, int, int]] = None,
                 active: bool = True) -> None:
        self.mask            = mask
        self.potential_value = potential_value
        self.active          = active
        if color is None:
            color = (255, 210, 0) if potential_value >= 0.5 else (0, 190, 230)
        self.color = color

    def __repr__(self) -> str:
        return f"PotentialAnchor(φ={self.potential_value:.2f}, mask={self.mask!r})"

    def __contains__(self, point: Tuple[float, float]) -> bool:
        return self.active and (point in self.mask)

    def get_mask(self, X: "np.ndarray",
                 Y: "np.ndarray") -> "np.ndarray":
        if self.active:
            return self.mask(X, Y)
        return np.zeros(X.shape, dtype=bool)


class MaterialObject:
    """
    Твёрдый объект

    Args:
        mask:   форма объекта
        color:  цвет отображения
        pinned: если True - нельзя переместить
        label:  имя для UI
        active: если False - игнорируется
    """

    def __init__(self, mask: "Mask",
                 color: Tuple[int, int, int] = (180, 180, 180),
                 pinned: bool = False,
                 label: str = "Object",
                 mass: float = 1.0,
                 active: bool = True) -> None:
        self.mask   = mask
        self.color  = color
        self.pinned = pinned
        self.label  = label
        self.mass   = float(mass)
        self.active = active
        self.vx: float = 0.0
        self.vy: float = 0.0

    def __repr__(self) -> str:
        return f"MaterialObject({self.label!r}, pinned={self.pinned}, mask={self.mask!r})"

    def __contains__(self, point: Tuple[float, float]) -> bool:
        return self.active and not self.pinned and (point in self.mask)

    def get_mask(self, X: "np.ndarray", Y: "np.ndarray") -> "np.ndarray":
        if self.active:
            return self.mask(X, Y)
        return np.zeros(X.shape, dtype=bool)


class ConductorObject:
    """
    Плавающий проводник

    Args:
        mask:   форма объекта
        color:  цвет отображения
        pinned: если True - нельзя перетащить
        label:  имя для UI
        active: если False - игнорируется
    """

    def __init__(self, mask: "Mask",
                 color: Tuple[int, int, int] = (220, 180, 60),
                 pinned: bool = False,
                 label: str = "Conductor",
                 mass: float = 1.0,
                 active: bool = True) -> None:
        self.mask   = mask
        self.color  = color
        self.pinned = pinned
        self.label  = label
        self.mass   = float(mass)
        self.active = active
        self.vx: float = 0.0
        self.vy: float = 0.0

    def __repr__(self) -> str:
        return f"ConductorObject({self.label!r}, pinned={self.pinned}, mask={self.mask!r})"

    def __contains__(self, point: Tuple[float, float]) -> bool:
        return self.active and not self.pinned and (point in self.mask)

    def get_mask(self, X: "np.ndarray", Y: "np.ndarray") -> "np.ndarray":
        if self.active:
            return self.mask(X, Y)
        return np.zeros(X.shape, dtype=bool)
