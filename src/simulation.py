"""
simulation.py - главный файл всей симуляции, объединяющий всю логику
"""


import numpy as np
import pygame
from typing import List, Optional, Tuple

from portals import (Portal, CouplePortal, FixedPotentialPortal,
                     PotentialAnchor, MaterialObject, ConductorObject)
from masks import (Mask, RectangleMask, CircleMask, LineMask, PointMask)
from physics import PhysicsEngine
from colors import GradientColorMapper, default_color_mapper, COLOR_SCHEMES
from ui import (
    TabbedPanel, Label, Button, Toggle, Stepper, Slider,
    Divider, SectionHeader,
    CLR_BG, CLR_TEXT, CLR_ACCENT, CLR_TEXT_DIM, CLR_BORDER,
    _init_fonts,
)


UI_WIDTH = 250


class Simulation:
    def __init__(self, *field,
                 sim_width: int,
                 sim_height: int,
                 px_scale: float,
                 iterations_per_frame: int = 50,
                 diff_threshold: float = 1e-4,
                 view_mode: str = "g_force",
                 show_vectors: bool = False,
                 show_isolines: bool = False,
                 isoline_count: int = 10,
                 fps: int = 60,
                 sor_omega: float = 1.7,
                 color_mapper: Optional[GradientColorMapper] = None) -> None:

        pygame.init()
        pygame.display.set_caption("Portals Engine")

        self.sim_width = sim_width
        self.sim_height = sim_height
        self.px_scale = px_scale
        self.field: List = list(field)

        self.sor_omega = sor_omega
        self.iterations_per_frame = iterations_per_frame
        self.diff_threshold = diff_threshold
        self.diff: float = 0.0
        self.total_iterations: int = 0

        self.view_mode = view_mode
        self.show_vectors = show_vectors
        self.show_isolines = show_isolines
        self.isoline_count = isoline_count

        self._color_scheme_names = list(COLOR_SCHEMES.keys())
        self._color_scheme_idx = 0
        self.color_mapper = color_mapper or default_color_mapper()

        win_w = int(sim_width * px_scale) + UI_WIDTH
        win_h = int(sim_height * px_scale)
        self.screen = pygame.display.set_mode((win_w, win_h))
        self.sim_surface = pygame.Surface((int(sim_width * px_scale), win_h))
        self.clock = pygame.time.Clock()
        self.fps = fps

        self.Y, self.X = np.ogrid[:sim_height, :sim_width]

        self._engine = PhysicsEngine(
            sim_width, sim_height, self.field, sor_omega=self.sor_omega)
        self.potential = self._engine.potential
        self.grad_x = self._engine.grad_x
        self.grad_y = self._engine.grad_y
        self.g_force = self._engine.g_force

        # Drag
        self._dragging_mask: Optional[Mask] = None
        self._dragging_obj = None

        # Кэш изолиний
        self._isolines_cache: Optional[np.ndarray] = None
        self._isolines_dirty = True

        # Буферы рендеринга
        sim_w_px = int(sim_width  * px_scale)
        sim_h_px = int(sim_height * px_scale)
        self._sim_w_px = sim_w_px
        self._sim_h_px = sim_h_px
        self._px_int = max(1, int(px_scale))

        # Поле: RGB-Surface без alpha (быстрее чем make_surface + transform.scale)
        self._field_surf = pygame.Surface((sim_w_px, sim_h_px))

        self._overlay_surf = pygame.Surface((sim_w_px, sim_h_px), pygame.SRCALPHA)

        self._portal_render_cache: Optional[list] = None
        self._portal_render_dirty = True

        self._fonts = _init_fonts()
        self._panel = self._build_panel()

    def update_physics(self) -> None:
        self._engine.sor_omega = self.sor_omega
        self.diff = self._engine.run_steps(
            self.iterations_per_frame, self.diff_threshold)
        self.total_iterations += self.iterations_per_frame
        self._engine.compute_gradients()
        self.potential = self._engine.potential
        self.grad_x = self._engine.grad_x
        self.grad_y = self._engine.grad_y
        self.g_force = self._engine.g_force
        self._update_material_dynamics()
        self._isolines_dirty = True

    def _update_material_dynamics(self) -> None:
        """
        Обновляет скорость и позицию динамических MaterialObject
        """

        needs_invalidate = False
        for obj in self.field:
            if not isinstance(obj, MaterialObject):
                continue
            if obj.pinned or not obj.active:
                continue
            mask = obj.get_mask(self.X, self.Y)
            if not np.any(mask):
                continue

            # Среднее значение градиента внутри объекта
            fx = float(-np.mean(self.grad_x[mask]) / obj.mass)
            fy = float(-np.mean(self.grad_y[mask]) / obj.mass)

            obj.vx = (obj.vx + fx) * 0.92  # демпфирование
            obj.vy = (obj.vy + fy) * 0.92

            # Ограничиваем максимальную скорость
            speed = (obj.vx**2 + obj.vy**2) ** 0.5
            max_speed = 2.0
            if speed > max_speed:
                obj.vx = obj.vx / speed * max_speed
                obj.vy = obj.vy / speed * max_speed

            if abs(obj.vx) > 0.01 or abs(obj.vy) > 0.01:
                obj.mask.translate(obj.vx, obj.vy)
                needs_invalidate = True

        if needs_invalidate:
            self._invalidate_caches()
            self._isolines_dirty = True

    def _invalidate_caches(self) -> None:
        """Инвалидирует кэш физики и кэш рендеринга порталов одним вызовом"""
        self._engine.invalidate_mask_cache()
        self._portal_render_dirty = True
        self._isolines_dirty      = True

    def _reset_engine(self) -> None:
        self._engine = PhysicsEngine(
            self.sim_width, self.sim_height, self.field,
            sor_omega=self.sor_omega)
        self.potential = self._engine.potential
        self.grad_x = self._engine.grad_x
        self.grad_y = self._engine.grad_y
        self.g_force = self._engine.g_force
        self.total_iterations = 0
        self._isolines_dirty = True
        self._portal_render_dirty = True

    def _render_field(self) -> None:
        """Рендерит поле в предаллоцированную Surface без аллокаций"""
        data = self.potential if self.view_mode == "potential" else self.g_force
        rgb = self.color_mapper(data)  # (H, W, 3) uint8
        ps = self._px_int
        rgbT = np.ascontiguousarray(rgb.transpose(1, 0, 2))  # (W, H, 3)
        up = np.repeat(np.repeat(rgbT, ps, axis=1), ps, axis=0)  # (W_px, H_px, 3)
        up = up[:self._sim_w_px, :self._sim_h_px]
        pxa = pygame.surfarray.pixels3d(self._field_surf)
        pxa[:] = up
        del pxa
        self.sim_surface.blit(self._field_surf, (0, 0))

    def _render_isolines(self) -> None:
        if not self.show_isolines:
            return
        data = self.potential if self.view_mode == "potential" else self.g_force
        if self._isolines_dirty or self._isolines_cache is None:
            self._isolines_cache = self._compute_isolines(data)
            self._isolines_dirty = False

        ps = max(1, int(self.px_scale))
        sim_w_px = int(self.sim_width  * self.px_scale)
        sim_h_px = int(self.sim_height * self.px_scale)
        up = np.repeat(np.repeat(self._isolines_cache, ps, 0), ps, 1)
        up = up[:sim_h_px, :sim_w_px]

        ov = self._overlay_surf
        ov.fill((0, 0, 0, 0))
        p3 = pygame.surfarray.pixels3d(ov)
        pa = pygame.surfarray.pixels_alpha(ov)
        ut = up.T
        p3[ut] = (255, 255, 255)
        pa[ut] = 80
        del p3, pa
        self.sim_surface.blit(ov, (0, 0))

    def _compute_isolines(self, data: np.ndarray) -> np.ndarray:
        d_min, d_max = float(np.min(data)), float(np.max(data))
        if d_max - d_min < 1e-9:
            return np.zeros(data.shape, dtype=bool)
        levels = np.linspace(d_min, d_max, self.isoline_count + 2)[1:-1]
        mask   = np.zeros(data.shape, dtype=bool)
        for level in levels:
            above = data > level
            ch = above[:-1, :] ^ above[1:, :]
            mask[:-1, :] |= ch;  mask[1:, :] |= ch
            cv = above[:, :-1] ^ above[:, 1:]
            mask[:, :-1] |= cv;  mask[:, 1:] |= cv
        return mask

    def _render_portals(self) -> None:
        """Кэшированный пиксельный рендер"""

        if self._portal_render_dirty or self._portal_render_cache is None:
            self._portal_render_cache = self._build_portal_render_cache()
            self._portal_render_dirty = False

        if not self._portal_render_cache:
            return

        ov = self._overlay_surf
        ov.fill((0, 0, 0, 0))
        p3 = pygame.surfarray.pixels3d(ov)
        pa = pygame.surfarray.pixels_alpha(ov)
        for ut, color in self._portal_render_cache:
            p3[ut] = color
            pa[ut] = 210
        del p3, pa
        self.sim_surface.blit(ov, (0, 0))

    def _build_portal_render_cache(self) -> list:
        """Вычисляет upscaled маски всех объектов. Вызывается редко."""
        ps = self._px_int
        sim_w_px = self._sim_w_px
        sim_h_px = self._sim_h_px
        result = []

        to_draw: list = []
        for obj in self.field:
            if isinstance(obj, CouplePortal):
                to_draw += [(obj.p1.mask, obj.p1.color),
                             (obj.p2.mask, obj.p2.color)]
            elif isinstance(obj, (FixedPotentialPortal, PotentialAnchor,
                                   MaterialObject, ConductorObject)):
                to_draw.append((obj.mask, obj.color))

        for mask, color in to_draw:
            m = mask(self.X, self.Y)  # (H_sim, W_sim) bool
            if not np.any(m):
                continue
            up = np.repeat(np.repeat(m, ps, axis=0), ps, axis=1)
            up = up[:sim_h_px, :sim_w_px]
            result.append((up.T.copy(), color))  # T: column-major для surfarray

        return result


    def _render_vectors(self) -> None:
        """Векторное поле"""
        if not self.show_vectors:
            return

        step = 5
        max_len = step * self.px_scale * 2.0
        head_len = max_len * 0.12
        lw  = max(1, int(head_len * 0.2))
        color = (60, 60, 70)
        ps = self.px_scale

        grad_mag = np.sqrt(self.grad_x ** 2 + self.grad_y ** 2)
        max_mag  = float(np.max(grad_mag))
        if max_mag < 1e-12:
            return
        scale = max_len / max_mag

        rows = np.arange(0, self.sim_height, step)
        cols = np.arange(0, self.sim_width,  step)
        R, C = np.meshgrid(rows, cols, indexing='ij')  # (Nr, Nc)

        gx = -self.grad_x[R, C]  # (Nr, Nc)
        gy = -self.grad_y[R, C]
        mag = grad_mag[R, C]

        valid = mag > 1e-12
        if not np.any(valid):
            return

        sx = (C * ps + ps / 2).astype(float)
        sy = (R * ps + ps / 2).astype(float)

        ex = sx + gx * scale
        ey = sy + gy * scale

        angles = np.arctan2(gy, gx)  # (Nr, Nc)

        vi = np.argwhere(valid)  # (N_valid, 2)
        for r, c in vi:
            pygame.draw.line(self.sim_surface, color,
                             (sx[r, c], sy[r, c]), (ex[r, c], ey[r, c]), lw)
            a = angles[r, c]
            for wa in (a + 2.618, a - 2.618):
                pygame.draw.line(self.sim_surface, color,
                                 (ex[r, c], ey[r, c]),
                                 (ex[r, c] + np.cos(wa) * head_len,
                                  ey[r, c] + np.sin(wa) * head_len), lw)


    def draw(self) -> None:
        self._render_field()
        self._render_isolines()
        self._render_portals()
        self._render_vectors()
        self.screen.blit(self.sim_surface, (0, 0))
        self._panel.draw(self.screen, self._fonts)
        pygame.display.flip()

    def handle_input(self) -> bool:
        ps = self.px_scale
        sim_w_px = int(self.sim_width * ps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    self.view_mode = ("g_force" if self.view_mode == "potential"
                                      else "potential")
                elif event.key == pygame.K_v:
                    self.show_vectors = not self.show_vectors
                elif event.key == pygame.K_i:
                    self.show_isolines = not self.show_isolines

            mouse_in_sim = (hasattr(event, "pos") and
                            event.pos[0] < sim_w_px)

            if not mouse_in_sim:
                self._panel.handle_event(event)
                continue

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos[0] / ps, event.pos[1] / ps
                self._dragging_mask, self._dragging_obj = \
                    self._find_draggable(mx, my)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                mx, my = event.pos[0] / ps, event.pos[1] / ps
                obj = self._find_obj_at(mx, my)
                if obj is not None:
                    self._panel._active = 1
                    self._open_inspector(obj)

            elif event.type == pygame.MOUSEBUTTONUP:
                self._dragging_mask = None
                self._dragging_obj  = None

            elif (event.type == pygame.MOUSEMOTION
                  and self._dragging_mask is not None):
                dx = event.rel[0] / ps
                dy = event.rel[1] / ps
                self._dragging_mask.translate(dx, dy)
                self._invalidate_caches()
                self._isolines_dirty = True
                self._panel.invalidate_tab("SCENE")

        return True

    def _find_draggable(self, mx, my) -> Tuple[Optional[Mask], object]:
        pt = (mx, my)
        for obj in self.field:
            if isinstance(obj, CouplePortal):
                if pt in obj.p1:
                    return obj.p1.mask, obj.p1
                if pt in obj.p2:
                    return obj.p2.mask, obj.p2
            elif isinstance(obj, MaterialObject):
                if pt in obj:
                    return obj.mask, obj
            elif isinstance(obj, (FixedPotentialPortal, PotentialAnchor)):
                if pt in obj:
                    return obj.mask, obj
        return None, None

    def _find_obj_at(self, mx: float, my: float):
        """Возвращает объект под курсором"""
        pt = (mx, my)
        for obj in self.field:
            if isinstance(obj, CouplePortal):
                if pt in obj.p1.mask:
                    return obj.p1
                if pt in obj.p2.mask:
                    return obj.p2
            elif isinstance(obj, (MaterialObject, FixedPotentialPortal,
                                   PotentialAnchor)):
                if obj.active and pt in obj.mask:
                    return obj
        return None

    def _build_panel(self) -> TabbedPanel:
        sim_w_px = int(self.sim_width * self.px_scale)
        sim_h_px = int(self.sim_height * self.px_scale)
        panel = TabbedPanel(sim_w_px, 0, UI_WIDTH, sim_h_px, ["SIMULATION", "INSPECTOR"])

        T = "SIMULATION"

        # region Объекты интерфейса
        panel.add(T, SectionHeader(0, 0, 0, "Portals Engine"))

        panel.add(T, SectionHeader(0, 0, 0, "Info"))
        panel.add(T, Label(0, 0, 0, "FPS",
                           value_fn=lambda: f"{self.clock.get_fps():.0f}"))
        panel.add(T, Label(0, 0, 0, "Diff",
                           value_fn=lambda: f"{self.diff:.2e}"))
        panel.add(T, Label(0, 0, 0, "Iterations",
                           value_fn=lambda: str(self.total_iterations)))
        panel.add(T, Label(0, 0, 0, "Grid",
                           value_fn=lambda: f"{self.sim_width}×{self.sim_height}"))
        panel.add(T, Divider(0, 0, 0))

        panel.add(T, SectionHeader(0, 0, 0, "View Mode"))
        panel.add(T, Button(0, 0, 0, 26, "Force",
                            callback=lambda: setattr(self, "view_mode", "g_force"),
                            active_fn=lambda: self.view_mode == "g_force"))
        panel.add(T, Button(0, 0, 0, 26, "Potential",
                            callback=lambda: setattr(self, "view_mode", "potential"),
                            active_fn=lambda: self.view_mode == "potential"))
        panel.add(T, Divider(0, 0, 0))

        panel.add(T, SectionHeader(0, 0, 0, "Display"))
        panel.add(T, Toggle(0, 0, 0, "Vectors",
                            getter=lambda: self.show_vectors,
                            setter=lambda v: setattr(self, "show_vectors", v)))
        panel.add(T, Toggle(0, 0, 0, "Isolines",
                            getter=lambda: self.show_isolines,
                            setter=lambda v: setattr(self, "show_isolines", v)))
        panel.add(T, Stepper(0, 0, 0, "Isolines count",
                             getter=lambda: self.isoline_count,
                             setter=lambda v: self._set_isoline_count(int(v)),
                             step=1, fmt="{:.0f}", min_val=2, max_val=30))
        panel.add(T, Divider(0, 0, 0))

        panel.add(T, SectionHeader(0, 0, 0, "Color Scheme"))
        panel.add(T, Label(0, 0, 0, "Scheme",
                           value_fn=lambda: self._color_scheme_names[
                               self._color_scheme_idx]))
        panel.add(T, Button(0, 0, 0, 26, "Previous",
                            callback=lambda: self._cycle_color(-1)))
        panel.add(T, Button(0, 0, 0, 26, "Next",
                            callback=lambda: self._cycle_color(+1)))
        panel.add(T, Divider(0, 0, 0))

        panel.add(T, SectionHeader(0, 0, 0, "Physics"))
        panel.add(T, Stepper(0, 0, 0, "Iter/frame",
                             getter=lambda: self.iterations_per_frame,
                             setter=lambda v: setattr(self, "iterations_per_frame",
                                                       int(v)),
                             step=5, fmt="{:.0f}", min_val=1, max_val=500))
        panel.add(T, Stepper(0, 0, 0, "SOR ω",
                             getter=lambda: self.sor_omega,
                             setter=lambda v: setattr(self, "sor_omega",
                                                       round(v, 2)),
                             step=0.05, fmt="{:.2f}", min_val=1.0, max_val=1.99))
        panel.add(T, Divider(0, 0, 0))

        panel.add(T, SectionHeader(0, 0, 0, "Hotkeys"))

        for line in ["M - change view mode", "V - show vectors",
                     "I - show isolines", "", ":3"]:
            lbl = Label(0, 0, 0, line, font_key="small")
            lbl.rect.h = 16
            panel.add(T, lbl)

        panel.set_tab_factory("INSPECTOR", self._build_scene_widgets)
        # endregion

        return panel

    def _build_scene_widgets(self) -> List:
        w: List = []

        w.append(SectionHeader(0, 0, 0, "Objects"))

        if not self.field:
            lbl = Label(0, 0, 0, "(empty scene)", font_key="small")
            lbl.rect.h = 16
            w.append(lbl)

        for i, obj in enumerate(self.field):
            if isinstance(obj, CouplePortal):
                w.append(SectionHeader(0, 0, 0, "Portal Pair"))
                w.append(Button(0, 0, 0, 24,
                                f"  Portal 1 ({_mask_type(obj.p1.mask)})",
                                callback=lambda o=obj.p1: self._open_inspector(o)))
                w.append(Button(0, 0, 0, 24,
                                f"  Portal 2 ({_mask_type(obj.p2.mask)})",
                                callback=lambda o=obj.p2: self._open_inspector(o)))
            elif isinstance(obj, FixedPotentialPortal):
                w.append(Button(0, 0, 0, 24,
                                f"Fixed  φ={obj.potential_value:.2f}"
                                f"  ({_mask_type(obj.mask)})",
                                callback=lambda o=obj: self._open_inspector(o)))
            elif isinstance(obj, PotentialAnchor):
                w.append(Button(0, 0, 0, 24,
                                f"Anchor φ={obj.potential_value:.2f}"
                                f"  ({_mask_type(obj.mask)})",
                                callback=lambda o=obj: self._open_inspector(o)))
            elif isinstance(obj, MaterialObject):
                pin = "pinned" if obj.pinned else "dynamic"
                w.append(Button(0, 0, 0, 24,
                                f"{pin} {obj.label}  ({_mask_type(obj.mask)})",
                                callback=lambda o=obj: self._open_inspector(o)))

        w.append(Divider(0, 0, 0))
        w.append(SectionHeader(0, 0, 0, "Add"))
        w.append(Button(0, 0, 0, 24, "+ Portal Pair",
                        callback=self._add_couple_portal))
        w.append(Button(0, 0, 0, 24, "+ Fixed Potential",
                        callback=self._add_fixed_potential))
        w.append(Button(0, 0, 0, 24, "+ Anchor",
                        callback=self._add_anchor))
        w.append(Button(0, 0, 0, 24, "+ Cube",
                        callback=lambda: self._add_material("Cube",
                            RectangleMask, (180, 180, 180))))
        w.append(Button(0, 0, 0, 24, "+ Planet",
                        callback=lambda: self._add_material("Planet",
                            CircleMask, (140, 90, 60))))
        w.append(Divider(0, 0, 0))

        w.append(SectionHeader(0, 0, 0, "Presets"))
        for name, fn in self._presets().items():
            w.append(Button(0, 0, 0, 24, name, callback=fn))

        return w

    def _cx(self): return self.sim_width // 2
    def _cy(self): return self.sim_height // 2

    def _add_couple_portal(self) -> None:
        cx, cy = self._cx(), self._cy()
        p1 = Portal(RectangleMask(cx-20, cx+20, cy-15, cy-13), (255, 153, 0))
        p2 = Portal(RectangleMask(cx-20, cx+20, cy+13, cy+15), (0, 204, 255))
        self.field.append(CouplePortal(p1, p2))
        self._refresh_field()

    def _add_fixed_potential(self) -> None:
        cx, cy = self._cx(), self._cy()
        self.field.append(
            FixedPotentialPortal(RectangleMask(cx-10, cx+10, cy-3, cy+3), 0.7))
        self._refresh_field()

    def _add_anchor(self) -> None:
        cx, cy = self._cx(), self._cy()
        self.field.append(PotentialAnchor(CircleMask(cx, cy, 4), 0.5))
        self._refresh_field()

    def _add_material(self, label: str, mask_cls, color: tuple) -> None:
        cx, cy = self._cx(), self._cy()
        if mask_cls is RectangleMask:
            mask = RectangleMask(cx-10, cx+10, cy-10, cy+10)
        else:
            mask = CircleMask(cx, cy, 10)
        self.field.append(MaterialObject(mask, color=color, label=label))
        self._refresh_field()

    def _refresh_field(self) -> None:
        self._invalidate_caches()
        self._panel.invalidate_tab("SCENE")

    # region Пресеты
    def _presets(self) -> dict:
        return {
            "Couple Portals": self._preset_couple_portals,
            "Couple Circles":  self._preset_couple_circles,
            "Advanced":        self._preset_advanced,
            "Dynamic Ball":    self._preset_dynamic,
            "Clear Scene":     self._preset_clear,
        }

    def _load_preset(self, field_objs: list) -> None:
        self.field = field_objs
        self._reset_engine()
        self._panel.invalidate_tab("SCENE")
        self._panel.close_inspector()

    def _preset_couple_portals(self) -> None:
        p1 = Portal(RectangleMask(25, 75, 25, 26), (255, 153, 0))
        p2 = Portal(RectangleMask(25, 75, 74, 75), (0, 204, 255))
        a_hi = PotentialAnchor(RectangleMask(0, self.sim_width, 0, 1),   1.0)
        a_lo = PotentialAnchor(RectangleMask(0, self.sim_width,
                                             self.sim_height-1,
                                             self.sim_height), 0.0)
        self._load_preset([a_hi, a_lo, CouplePortal(p1, p2)])

    def _preset_couple_circles(self) -> None:
        p1 = Portal(CircleMask(35, self.sim_height//2, 10), (255, 100, 50))
        p2 = Portal(CircleMask(self.sim_width-35, self.sim_height//2, 10),
                    (50, 200, 255))
        a_hi = PotentialAnchor(RectangleMask(0, self.sim_width, 0, 1),   1.0)
        a_lo = PotentialAnchor(RectangleMask(0, self.sim_width,
                                             self.sim_height-1,
                                             self.sim_height), 0.0)
        self._load_preset([a_hi, a_lo, CouplePortal(p1, p2)])

    def _preset_advanced(self) -> None:
        W, H = self.sim_width, self.sim_height
        top = FixedPotentialPortal(
            RectangleMask(20, W-20, H//3, H//3 + 3), 0.8, (255, 80, 80))
        bot = FixedPotentialPortal(
            RectangleMask(20, W-20, 2*H//3 - 3, 2*H//3), 0.2, (80, 120, 255))
        obs = MaterialObject(CircleMask(W//2, H//2, 8),
                             color=(80, 220, 120), pinned=True, label="Conductor")
        self._load_preset([top, bot, obs])

    def _preset_dynamic(self) -> None:
        W, H = self.sim_width, self.sim_height
        a_hi = PotentialAnchor(RectangleMask(0, W, 0, 1),   1.0)
        a_lo = PotentialAnchor(RectangleMask(0, W, H-1, H), 0.0)
        ball = MaterialObject(CircleMask(W//2, H//4, 7),
                              color=(220, 160, 60), pinned=False,
                              label="Ball", mass=1.0)
        self._load_preset([a_hi, a_lo, ball])

    def _preset_clear(self) -> None:
        self._load_preset([])
    # endregion

    def _open_inspector(self, obj) -> None:
        self._panel.show_inspector(self._build_inspector(obj))

    def _close_inspector(self) -> None:
        self._panel.close_inspector()
        self._panel.invalidate_tab("SCENE")

    def _build_inspector(self, obj) -> List:
        w: List = []
        w.append(Button(0, 0, 0, 26, "<- Back",
                        callback=self._close_inspector))
        w.append(SectionHeader(0, 0, 0, type(obj).__name__))

        # Удалить объект
        w.append(Button(0, 0, 0, 24, "Remove from scene",
                        callback=lambda o=obj: self._remove_obj(o)))

        # Потенциал
        if isinstance(obj, (FixedPotentialPortal, PotentialAnchor)):
            w.append(SectionHeader(0, 0, 0, "Potential"))
            w.append(Slider(0, 0, 0, "φ value",
                            getter=lambda o=obj: o.potential_value,
                            setter=lambda v, o=obj: (
                                setattr(o, "potential_value", round(v, 3)),
                                self._invalidate_caches()),
                            min_val=0.0, max_val=1.0, fmt="{:.3f}"))

        # MaterialObject
        if isinstance(obj, MaterialObject):
            w.append(SectionHeader(0, 0, 0, "Object"))
            w.append(Toggle(0, 0, 0, "Pinned",
                            getter=lambda o=obj: o.pinned,
                            setter=lambda v, o=obj: setattr(o, "pinned", v)))
            w.append(Stepper(0, 0, 0, "Mass",
                             getter=lambda o=obj: o.mass,
                             setter=lambda v, o=obj: setattr(o, "mass", max(0.1, v)),
                             step=0.5, fmt="{:.1f}", min_val=0.1, max_val=100.0))
            w.append(Label(0, 0, 0, "Speed",
                           value_fn=lambda o=obj: f"{(o.vx**2+o.vy**2)**0.5:.2f}"))
            w.append(Button(0, 0, 0, 24, "Stop (reset velocity)",
                            callback=lambda o=obj: (
                                setattr(o, "vx", 0.0),
                                setattr(o, "vy", 0.0))))

        # Маска
        if hasattr(obj, "mask"):
            w.append(SectionHeader(0, 0, 0, "Mask type"))
            w.append(Button(0, 0, 0, 24, "→ Rectangle",
                            callback=lambda o=obj: self._change_mask(
                                o, "rect"),
                            active_fn=lambda o=obj: isinstance(
                                o.mask, RectangleMask)))
            w.append(Button(0, 0, 0, 24, "* Circle",
                            callback=lambda o=obj: self._change_mask(
                                o, "circle"),
                            active_fn=lambda o=obj: isinstance(
                                o.mask, CircleMask)))
            w.append(Button(0, 0, 0, 24, "* Point",
                            callback=lambda o=obj: self._change_mask(
                                o, "point"),
                            active_fn=lambda o=obj: isinstance(
                                o.mask, PointMask)))
            w.append(Button(0, 0, 0, 24, "* Line",
                            callback=lambda o=obj: self._change_mask(
                                o, "line"),
                            active_fn=lambda o=obj: isinstance(
                                o.mask, LineMask)))
            w += self._mask_widgets(obj)

        # Цвет
        if hasattr(obj, "color"):
            w.append(SectionHeader(0, 0, 0, "Color"))
            w += self._color_widgets(obj)

        return w

    # Смена типа маски

    def _change_mask(self, obj, mask_type: str) -> None:
        """Заменяет маску объекта на новую"""
        old = obj.mask
        cx = cy = None
        if isinstance(old, RectangleMask):
            cx = (old.x_min + old.x_max) / 2
            cy = (old.y_min + old.y_max) / 2
        elif isinstance(old, CircleMask):
            cx, cy = old.cx, old.cy
        elif isinstance(old, PointMask):
            cx, cy = old.x, old.y
        elif isinstance(old, LineMask):
            cx = (old.x1 + old.x2) / 2
            cy = (old.y1 + old.y2) / 2
        if cx is None:
            cx, cy = self._cx(), self._cy()

        if mask_type == "rect":
            obj.mask = RectangleMask(cx-10, cx+10, cy-5, cy+5)
        elif mask_type == "circle":
            obj.mask = CircleMask(cx, cy, 8)
        elif mask_type == "point":
            obj.mask = PointMask(cx, cy)
        elif mask_type == "line":
            obj.mask = LineMask(cx-10, cy, cx+10, cy, 1.0)

        self._invalidate_caches()
        self._panel.show_inspector(self._build_inspector(obj))

    # Параметры маски
    def _mask_widgets(self, obj) -> List:
        mask = obj.mask
        w: List = []
        w.append(SectionHeader(0, 0, 0, "Mask params"))

        def mk(label, getter, setter, step=1.0, mn=-1e6, mx=1e6):
            def _set(v, s=setter):
                s(v)
                self._on_mask_changed()
            return Stepper(0, 0, 0, label, getter=getter, setter=_set,
                           step=step, fmt="{:.1f}", min_val=mn, max_val=mx)

        if isinstance(mask, RectangleMask):
            w.append(mk("X min", lambda m=mask: m.x_min,
                         lambda v, m=mask: setattr(m, "x_min", v),
                         mn=0, mx=self.sim_width))
            w.append(mk("X max", lambda m=mask: m.x_max,
                         lambda v, m=mask: setattr(m, "x_max", v),
                         mn=0, mx=self.sim_width))
            w.append(mk("Y min", lambda m=mask: m.y_min,
                         lambda v, m=mask: setattr(m, "y_min", v),
                         mn=0, mx=self.sim_height))
            w.append(mk("Y max", lambda m=mask: m.y_max,
                         lambda v, m=mask: setattr(m, "y_max", v),
                         mn=0, mx=self.sim_height))

        elif isinstance(mask, CircleMask):
            w.append(mk("CX", lambda m=mask: m.cx,
                         lambda v, m=mask: setattr(m, "cx", v),
                         mn=0, mx=self.sim_width))
            w.append(mk("CY", lambda m=mask: m.cy,
                         lambda v, m=mask: setattr(m, "cy", v),
                         mn=0, mx=self.sim_height))
            w.append(mk("Radius", lambda m=mask: m.radius,
                         lambda v, m=mask: setattr(m, "radius", max(1, v)),
                         mn=1, mx=min(self.sim_width, self.sim_height)//2))

        elif isinstance(mask, PointMask):
            w.append(mk("X", lambda m=mask: m.x,
                         lambda v, m=mask: setattr(m, "x", v),
                         mn=0, mx=self.sim_width))
            w.append(mk("Y", lambda m=mask: m.y,
                         lambda v, m=mask: setattr(m, "y", v),
                         mn=0, mx=self.sim_height))

        elif isinstance(mask, LineMask):
            w.append(mk("X1", lambda m=mask: m.x1,
                         lambda v, m=mask: (setattr(m, "x1", v), m._update_cache())))
            w.append(mk("Y1", lambda m=mask: m.y1,
                         lambda v, m=mask: (setattr(m, "y1", v), m._update_cache())))
            w.append(mk("X2", lambda m=mask: m.x2,
                         lambda v, m=mask: (setattr(m, "x2", v), m._update_cache())))
            w.append(mk("Y2", lambda m=mask: m.y2,
                         lambda v, m=mask: (setattr(m, "y2", v), m._update_cache())))
            w.append(mk("Thickness", lambda m=mask: m.thickness,
                         lambda v, m=mask: setattr(m, "thickness", max(0.5, v)),
                         mn=0.5, mx=20))
        return w

    def _on_mask_changed(self) -> None:
        self._invalidate_caches()
        self._isolines_dirty = True

    # Цвет
    def _color_widgets(self, obj) -> List:
        def set_ch(o, ch, v):
            c = list(o.color)
            c[ch] = int(max(0, min(255, v)))
            o.color = tuple(c)

        return [
            Stepper(0, 0, 0, "R", getter=lambda o=obj: o.color[0],
                    setter=lambda v, o=obj: set_ch(o, 0, v),
                    step=5, fmt="{:.0f}", min_val=0, max_val=255),
            Stepper(0, 0, 0, "G", getter=lambda o=obj: o.color[1],
                    setter=lambda v, o=obj: set_ch(o, 1, v),
                    step=5, fmt="{:.0f}", min_val=0, max_val=255),
            Stepper(0, 0, 0, "B", getter=lambda o=obj: o.color[2],
                    setter=lambda v, o=obj: set_ch(o, 2, v),
                    step=5, fmt="{:.0f}", min_val=0, max_val=255),
        ]

    def _remove_obj(self, obj) -> None:
        """Удаляет объект или портал, содержащий obj, из self.field"""
        to_remove = None
        for f in self.field:
            if f is obj:
                to_remove = f
                break
            if isinstance(f, CouplePortal) and (f.p1 is obj or f.p2 is obj):
                to_remove = f
                break
        if to_remove is not None:
            self.field.remove(to_remove)
        self._invalidate_caches()
        self._panel.invalidate_tab("SCENE")
        self._close_inspector()

    def _set_isoline_count(self, n: int) -> None:
        self.isoline_count   = n
        self._isolines_dirty = True

    def _cycle_color(self, d: int) -> None:
        self._color_scheme_idx = (
            (self._color_scheme_idx + d) % len(self._color_scheme_names))
        self.color_mapper = COLOR_SCHEMES[
            self._color_scheme_names[self._color_scheme_idx]]()

    def run(self) -> None:
        running = True
        try:
            while running:
                running = self.handle_input()
                self.update_physics()
                self.draw()
                self.clock.tick(self.fps)
        finally:
            pygame.quit()


def _mask_type(mask) -> str:
    return type(mask).__name__.replace("Mask", "")
