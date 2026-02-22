import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple


class ColorMapper(ABC):
    """Абстрактный цветовой мапер: values -> RGB uint8 array"""

    @abstractmethod
    def __call__(self, values: np.ndarray) -> np.ndarray:
        """Возвращает массив uint8 - цвета RGB"""


class SingleColorMapper(ColorMapper):
    """Маппер, возвращающий один и тот же цвет для всех значений"""

    def __init__(self, color: Tuple[int, int, int]) -> None:
        self.color = np.array(color, dtype=np.uint8)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        result = np.empty(values.shape + (3,), dtype=np.uint8)
        result[:] = self.color
        return result


class GradientColorMapper(ColorMapper):
    """
    Линейный градиент между опорными точками

    Args:
        points: список пар (value, (R, G, B)), отсортированных по value
    """

    def __init__(self, points: List[Tuple[float, Tuple[int, int, int]]]) -> None:
        self.points = sorted(points, key=lambda p: p[0])
        self.values = np.array([p[0] for p in self.points], dtype=np.float32)
        self.colors = np.array([p[1] for p in self.points], dtype=np.float32)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        v = np.clip(values, self.values[0], self.values[-1]).astype(np.float32)
        idx = np.searchsorted(self.values, v, side="right") - 1
        idx = np.clip(idx, 0, len(self.points) - 2)

        v0 = self.values[idx]
        v1 = self.values[idx + 1]
        t = (v - v0) / np.where(v1 != v0, v1 - v0, 1.0)
        t = t[..., np.newaxis]  # broadcast для RGB

        interpolated = self.colors[idx] * (1.0 - t) + self.colors[idx + 1] * t
        return np.clip(interpolated, 0, 255).astype(np.uint8)


# region Цветовые схемы

def colormap_plasma() -> GradientColorMapper:
    return GradientColorMapper([
        (0.0, (13, 8, 135)),  # тёмно-фиолетовый
        (0.25, (84, 2, 163)),  # фиолетовый
        (0.5, (139, 10, 165)),  # пурпурный
        (0.75, (211, 97, 107)),  # розовато-оранжевый
        (1.0, (240, 249, 33)),  # жёлтый
    ])


def colormap_electric() -> GradientColorMapper:
    return GradientColorMapper([
        (0.0, (0, 0, 0)),  # чёрный
        (0.3, (20, 20, 180)),  # тёмно-синий
        (0.6, (70, 130, 255)),  # синий
        (0.85, (180, 220, 255)),  # голубой
        (1.0, (255, 255, 255)),  # белый
    ])



def colormap_fire() -> GradientColorMapper:
    return GradientColorMapper([
        (0.0, (0, 0, 0)),  # чёрный
        (0.25, (128, 0, 0)),  # тёмно-красный
        (0.5, (255, 69, 0)),  # оранжево-красный
        (0.75, (255, 215, 0)),  # золотистый
        (1.0, (255, 255, 255)),  # белый
    ])


def default_color_mapper() -> GradientColorMapper:
    return GradientColorMapper([
        (0.0, (0, 0, 139)),  # тёмно-синий
        (1.0, (0, 150, 255)),  # ярко-синий
        (2.5, (50, 205, 50)),  # зелёный
        (5.0, (255, 215, 0)),  # жёлтый
        (7.5, (255, 140, 0)),  # оранжевый
        (10.0, (255, 60, 60)),  # красный (без белого, чтобы не выгорало)
    ])


def extra_mapper() -> GradientColorMapper:
    return GradientColorMapper([
        (0.0, (0, 0, 139)),  # тёмно-синий
        (0.25, (30, 144, 255)),  # синий
        (0.5, (0, 200, 200)),  # бирюзовый
        (0.75, (255, 220, 0)),  # жёлтый
        (1.0, (220, 20, 60)),  # красный
    ])


def default_potential_mapper() -> GradientColorMapper:
    return GradientColorMapper([
        (0.0, (30, 30, 200)),   # синий
        (0.5, (60, 200, 80)),   # зелёный
        (1.0, (210, 30, 30)),   # красный
    ])
# endregion


# Список цветовых схем
COLOR_SCHEMES = {
    "Default": default_color_mapper,
    "Potential": default_potential_mapper,
    "Extra": extra_mapper,
    "Plasma": colormap_plasma,
    "Electric": colormap_electric,
    "Fire": colormap_fire,
}
