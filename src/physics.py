"""
physics.py - физическое ядро симуляции

Решает уравнение Лапласа методом Red-Black SOR
(подробнее в explanation/explanation.pdf

  Объекты:
    PotentialAnchor       φ = const (якорь, граничное условие Дирихле)
    FixedPotentialPortal  φ = const (якорь, граничное условие Дирихле. Обладает физическими свойствами)
    CouplePortal          переносят гравитацию
    MultiPortal           несколько порталов, соединенных вместе (не спрашивайте)
    MaterialObject        непрозрачное препятствие
    ConductorObject       плавающий проводник
"""

import numpy as np
from typing import List, Optional
from portals import *


class PhysicsEngine:
    def __init__(self, width: int, height: int,
                 field: list, sor_omega: float = 1.7, default_grad=(0.0, 0.0)) -> None:
        self.width = width
        self.height = height
        self.field = field
        self.sor_omega = sor_omega

        self.Y, self.X = np.ogrid[:height, :width]

        # Заранее применяем градиент потецниала (от 1.0 до 0.0). Т.к. это не совместимо с некоторыми сценами,
        # по умолчанию стоит (0.0, 0.0)
        y_lin = np.linspace(default_grad[0], default_grad[1], height)
        self.potential = np.tile(y_lin[:, np.newaxis], (1, width)).astype(np.float64)
        self.grad_x    = np.zeros_like(self.potential)
        self.grad_y    = np.zeros_like(self.potential)
        self.g_force   = np.zeros_like(self.potential)

        # Red-Black маски внутренних ячеек
        i_idx, j_idx = np.meshgrid(np.arange(1, height - 1),
                                    np.arange(1, width - 1), indexing="ij")
        checker = (i_idx + j_idx) % 2
        self._red_inner = checker == 0
        self._black_inner = checker == 1

        # Кэш масок
        self._cache_dirty = True
        self._portal_mask_cache:   Optional[np.ndarray] = None
        self._active_couples_cache: Optional[list]  = None
        self._active_multies_cache: Optional[list] = None
        self._material_cache: Optional[list] = None
        self._conductor_cache: Optional[list] = None
        self._dirichlet_cache: Optional[list] = None

        self._apply_dirichlet()
        self.compute_gradients()

    def invalidate_mask_cache(self) -> None:
        self._cache_dirty = True

    def _iter_dirichlet(self):
        for obj in self.field:
            if isinstance(obj, (FixedPotentialPortal, PotentialAnchor)) and obj.active:
                yield obj

    def _rebuild_cache(self) -> None:
        active_couples: list = []
        for obj in self.field:
            if not isinstance(obj, CouplePortal):
                continue
            m1 = obj.p1.get_mask(self.X, self.Y)
            m2 = obj.p2.get_mask(self.X, self.Y)
            n1, n2 = int(np.sum(m1)), int(np.sum(m2))
            if n1 == 0 or n2 == 0:
                continue
            active_couples.append((obj, m1, m2))

        active_multies: list = []
        for obj in self.field:
            if not isinstance(obj, MultiPortal):
                continue

            active_multies.append((obj, tuple(p.get_mask(self.X, self.Y) for p in obj.args)))

        material_list: list = []
        for obj in self.field:
            if isinstance(obj, MaterialObject) and obj.active:
                m = obj.get_mask(self.X, self.Y)
                if np.any(m):
                    material_list.append((obj, m))

        conductor_list: list = []
        for obj in self.field:
            if isinstance(obj, ConductorObject) and obj.active:
                m = obj.get_mask(self.X, self.Y)
                if np.any(m):
                    conductor_list.append((obj, m))

        dirichlet_cache: list = []
        for obj in self._iter_dirichlet():
            m = obj.get_mask(self.X, self.Y)
            if np.any(m):
                dirichlet_cache.append((m, float(obj.potential_value)))

        frozen = np.zeros((self.height, self.width), dtype=bool)
        for _, m in material_list:
            frozen |= m
        for _, m in conductor_list:
            frozen |= m
        for m, _ in dirichlet_cache:
            frozen |= m

        self._active_couples_cache = active_couples
        self._active_multies_cache = active_multies
        self._material_cache = material_list
        self._conductor_cache = conductor_list
        self._dirichlet_cache = dirichlet_cache
        self._portal_mask_cache = frozen
        self._cache_dirty = False

    def _apply_dirichlet(self) -> None:
        for obj in self._iter_dirichlet():
            m = obj.get_mask(self.X, self.Y)
            self.potential[m] = obj.potential_value

    def step(self) -> float:
        """Red-Black SOR"""

        if self._cache_dirty:
            self._rebuild_cache()

        p            = self.potential
        omega        = self.sor_omega
        frozen_inner = self._portal_mask_cache[1:-1, 1:-1]

        # Red pass
        avg = 0.25 * (p[0:-2, 1:-1] + p[2:, 1:-1] +
                       p[1:-1, 0:-2] + p[1:-1, 2:])
        sor = (1.0 - omega) * p[1:-1, 1:-1] + omega * avg
        upd = self._red_inner & ~frozen_inner
        p[1:-1, 1:-1] = np.where(upd, sor, p[1:-1, 1:-1])

        # Black pass
        avg = 0.25 * (p[0:-2, 1:-1] + p[2:, 1:-1] +
                       p[1:-1, 0:-2] + p[1:-1, 2:])
        sor = (1.0 - omega) * p[1:-1, 1:-1] + omega * avg
        upd = self._black_inner & ~frozen_inner
        p[1:-1, 1:-1] = np.where(upd, sor, p[1:-1, 1:-1])

        # Граничные условия
        p[:, 0]   = p[:, 1] # Нейман боковые
        p[:, -1]  = p[:, -2]

        # CouplePortal
        for _, m1, m2 in self._active_couples_cache:
            combined = m1 | m2
            p[combined] = float(np.mean(p[combined]))

        # MultiPortal
        for _, generator in self._active_multies_cache:
            if not len(generator):
                continue

            combined = generator[0]
            for i in generator[1:]:
                combined |= i

            p[combined] = float(np.mean(p[combined]))

        # ConductorObject
        for _, m in self._conductor_cache:
            ext = self._exterior_neighbors(m)
            if np.any(ext):
                p[m] = float(np.mean(p[ext]))

        # FixedPotentialPortal
        for m, val in self._dirichlet_cache:
            p[m] = val

        residual = (0.25 * (p[0:-2, 1:-1] + p[2:, 1:-1] +
                            p[1:-1, 0:-2] + p[1:-1, 2:])
                    - p[1:-1, 1:-1])
        return float(np.mean(np.abs(residual)))

    @staticmethod
    def _exterior_neighbors(mask: np.ndarray) -> np.ndarray:
        d = np.zeros_like(mask)
        d[1:, :] |= mask[:-1, :]
        d[:-1, :] |= mask[1:, :]
        d[:, 1:] |= mask[:, :-1]
        d[:, :-1] |= mask[:, 1:]
        return d & ~mask

    def run_steps(self, n_steps: int,
                  diff_threshold: float = 1e-4) -> float:
        diff = float("inf")
        for _ in range(n_steps):
            diff = self.step()
            if diff < diff_threshold:
                break
        return diff

    def compute_gradients(self) -> None:
        self.grad_y, self.grad_x = np.gradient(self.potential)
        self.g_force = np.sqrt(self.grad_x ** 2 + self.grad_y ** 2) * self.height

    def compute_isolines(self, data: np.ndarray, n_levels: int) -> np.ndarray:
        d_min, d_max = float(np.min(data)), float(np.max(data))
        if d_max - d_min < 1e-9:
            return np.zeros(data.shape, dtype=bool)
        levels = np.linspace(d_min, d_max, n_levels + 2)[1:-1]
        mask = np.zeros(data.shape, dtype=bool)
        for level in levels:
            above = data > level
            ch = above[:-1, :] ^ above[1:, :]
            mask[:-1, :] |= ch;  mask[1:, :] |= ch
            cv = above[:, :-1] ^ above[:, 1:]
            mask[:, :-1] |= cv;  mask[:, 1:] |= cv
        return mask
