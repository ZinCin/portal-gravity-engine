"""
ui.py - функции и объекты виджетов для пользовательского интерфейса

Widgets:
  Label, Button, Toggle, Stepper, Slider, Divider, SectionHeader

  TabbedPanel - панель с вкладками, скроллом и поддержкой инспектора, хранящий виджеты
"""

import pygame
from typing import Callable, List, Optional, Tuple


# region Стиль
FONT_SIZE_TITLE  = 15
FONT_SIZE_LABEL  = 13
FONT_SIZE_SMALL  = 11

CLR_BG           = (22,  24,  28)
CLR_SECTION      = (32,  35,  42)
CLR_BORDER       = (55,  60,  72)
CLR_ACCENT       = (80, 160, 255)
CLR_ACCENT_HOVER = (110, 185, 255)
CLR_BTN_ACTIVE   = (50, 130, 230)
CLR_BTN_INACTIVE = (42,  46,  58)
CLR_TEXT         = (220, 225, 235)
CLR_TEXT_DIM     = (130, 138, 155)
CLR_GREEN        = (60, 200, 100)
CLR_RED          = (220,  70,  70)

_BTN_W = 24   # ширина кнопок изменения
_BTN_H = 20   # высота кнопок изменения
_VAL_W = 38   # ширина поля значения
_GAP = 3    # пропуск между элементами
# endregion


def _init_fonts() -> dict:
    return {
        "title": pygame.font.SysFont("Segoe UI", FONT_SIZE_TITLE, bold=True),
        "label": pygame.font.SysFont("Segoe UI", FONT_SIZE_LABEL),
        "small": pygame.font.SysFont("Segoe UI", FONT_SIZE_SMALL),
    }


class Widget:
    """Абстрактный виджет: создает виджет на экране"""

    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        self.rect = pygame.Rect(x, y, w, h)

    def draw(self, surface: pygame.Surface, fonts: dict) -> None:
        """Отрисовка виджета"""
        pass

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Определяет управления (нажатие, перенос и т.п.)"""
        return False

    def move_to(self, x: int, y: int) -> None:
        """Перемещение виджета"""
        self.rect.topleft = (x, y)


# region Widgets
class Label(Widget):
    def __init__(self, x, y, w, text: str,
                 value_fn: Optional[Callable[[], str]] = None,
                 color: Tuple[int, int, int] = CLR_TEXT,
                 font_key: str = "label") -> None:
        super().__init__(x, y, w, 20)
        self.text = text
        self.value_fn = value_fn
        self.color = color
        self.font_key = font_key

    def draw(self, surface: pygame.Surface, fonts: dict) -> None:
        font = fonts[self.font_key]
        lbl = font.render(self.text, True, CLR_TEXT_DIM)
        surface.blit(lbl, (self.rect.x, self.rect.y))
        if self.value_fn is not None:
            val = font.render(self.value_fn(), True, self.color)
            surface.blit(val, (self.rect.right - val.get_width(), self.rect.y))


class Button(Widget):
    def __init__(self, x, y, w, h, text: str,
                 callback: Callable[[], None],
                 active_fn: Optional[Callable[[], bool]] = None) -> None:
        super().__init__(x, y, w, h)
        self.text = text
        self.callback = callback
        self.active_fn = active_fn
        self._hovered = False

    def draw(self, surface: pygame.Surface, fonts: dict) -> None:
        is_active = self.active_fn() if self.active_fn else False
        if is_active:    bg, tc = CLR_BTN_ACTIVE, (255, 255, 255)
        elif self._hovered: bg, tc = CLR_ACCENT_HOVER, (10, 10, 10)
        else:            bg, tc = CLR_BTN_INACTIVE, CLR_TEXT
        pygame.draw.rect(surface, bg, self.rect, border_radius=5)
        pygame.draw.rect(surface, CLR_BORDER, self.rect, 1, border_radius=5)
        font = fonts["label"]
        lbl = font.render(self.text, True, tc)
        surface.blit(lbl, (self.rect.centerx - lbl.get_width() // 2,
                            self.rect.centery - lbl.get_height() // 2))

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self._hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()
                return True
        return False


class Toggle(Widget):
    def __init__(self, x, y, w, text: str,
                 getter: Callable[[], bool],
                 setter: Callable[[bool], None]) -> None:
        super().__init__(x, y, w, 24)
        self.text = text
        self.getter = getter
        self.setter = setter

    def draw(self, surface: pygame.Surface, fonts: dict) -> None:
        val = self.getter()
        font = fonts["label"]
        lbl = font.render(self.text, True, CLR_TEXT if val else CLR_TEXT_DIM)
        surface.blit(lbl, (self.rect.x, self.rect.centery - lbl.get_height() // 2))
        tw, th = 36, 18
        tr = pygame.Rect(self.rect.right - tw, self.rect.centery - th // 2, tw, th)
        pygame.draw.rect(surface, CLR_GREEN if val else (70, 75, 90), tr, border_radius=9)
        cx = tr.right - 11 if val else tr.left + 11
        pygame.draw.circle(surface, (255, 255, 255), (cx, tr.centery), 7)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.setter(not self.getter())
                return True
        return False


class Stepper(Widget):
    def __init__(self, x, y, w, text: str,
                 getter: Callable[[], float],
                 setter: Callable[[float], None],
                 step: float = 1.0,
                 fmt: str = "{:.0f}",
                 min_val: float = -1e9,
                 max_val: float =  1e9) -> None:
        super().__init__(x, y, w, _BTN_H + 6)
        self.text = text
        self.getter = getter
        self.setter = setter
        self.step = step
        self.fmt = fmt
        self.min_val = min_val
        self.max_val = max_val

    def _get_btn_minus_rect(self) -> pygame.Rect:
        ry = self.rect.y + 3
        # minus левее, чем plus
        plus_left = self.rect.right - _BTN_W - 2
        minus_left = plus_left - _GAP - _BTN_W
        return pygame.Rect(minus_left, ry, _BTN_W, _BTN_H)

    def _get_btn_plus_rect(self) -> pygame.Rect:
        ry = self.rect.y + 3
        plus_left = self.rect.right - _BTN_W - 2
        return pygame.Rect(plus_left, ry, _BTN_W, _BTN_H)

    def _get_val_rect(self) -> pygame.Rect:
        ry = self.rect.y + 3
        plus_left = self.rect.right - _BTN_W - 2
        minus_left = plus_left - _GAP - _BTN_W
        val_left = minus_left - _GAP - _VAL_W
        return pygame.Rect(val_left, ry, _VAL_W, _BTN_H)

    def move_to(self, x: int, y: int) -> None:
        super().move_to(x, y)

    def draw(self, surface: pygame.Surface, fonts: dict) -> None:
        font = fonts["label"]
        lbl = font.render(self.text, True, CLR_TEXT_DIM)
        surface.blit(lbl, (self.rect.x, self.rect.centery - lbl.get_height() // 2))

        # -
        btn_minus = self._get_btn_minus_rect()
        pygame.draw.rect(surface, CLR_BTN_INACTIVE, btn_minus, border_radius=4)
        pygame.draw.rect(surface, CLR_BORDER, btn_minus, 1, border_radius=4)
        s_minus = font.render("−", True, CLR_TEXT)
        surface.blit(s_minus, (btn_minus.centerx - s_minus.get_width() // 2,
                               btn_minus.centery - s_minus.get_height() // 2))

        # +
        btn_plus = self._get_btn_plus_rect()
        pygame.draw.rect(surface, CLR_BTN_INACTIVE, btn_plus, border_radius=4)
        pygame.draw.rect(surface, CLR_BORDER, btn_plus, 1, border_radius=4)
        s_plus = font.render("+", True, CLR_TEXT)
        surface.blit(s_plus, (btn_plus.centerx - s_plus.get_width() // 2,
                              btn_plus.centery - s_plus.get_height() // 2))

        val_rect = self._get_val_rect()
        vs = fonts["small"].render(self.fmt.format(self.getter()), True, CLR_ACCENT)
        surface.blit(vs, (val_rect.centerx - vs.get_width() // 2,
                          val_rect.centery - vs.get_height() // 2))

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._get_btn_minus_rect().collidepoint(event.pos):
                self.setter(max(self.min_val, self.getter() - self.step))
                return True
            if self._get_btn_plus_rect().collidepoint(event.pos):
                self.setter(min(self.max_val, self.getter() + self.step))
                return True
        return False


class Slider(Widget):
    _TRACK_H = 5
    _THUMB_R = 6
    _LABEL_H = 17

    def __init__(self, x, y, w, text: str,
                 getter: Callable[[], float],
                 setter: Callable[[float], None],
                 min_val: float = 0.0,
                 max_val: float = 1.0,
                 fmt: str = "{:.2f}") -> None:
        super().__init__(x, y, w, Slider._LABEL_H + Slider._THUMB_R * 2 + 6)
        self.text = text
        self.getter = getter
        self.setter = setter
        self.min_val = min_val
        self.max_val = max_val
        self.fmt = fmt
        self._dragging = False

    def _track_rect(self) -> pygame.Rect:
        r = self._THUMB_R
        cy = self.rect.y + self._LABEL_H + r + 1
        return pygame.Rect(self.rect.x + r, cy - self._TRACK_H // 2,
                           self.rect.w - r * 2, self._TRACK_H)

    def _thumb_x(self, tr: pygame.Rect) -> int:
        span = self.max_val - self.min_val
        t = (self.getter() - self.min_val) / span if span > 1e-9 else 0.0
        return tr.x + int(max(0.0, min(1.0, t)) * tr.w)

    def draw(self, surface: pygame.Surface, fonts: dict) -> None:
        font = fonts["label"]
        surface.blit(font.render(self.text, True, CLR_TEXT_DIM),
                     (self.rect.x, self.rect.y))
        vs = fonts["small"].render(self.fmt.format(self.getter()), True, CLR_ACCENT)
        surface.blit(vs, (self.rect.right - vs.get_width(), self.rect.y))
        tr = self._track_rect()
        tx = self._thumb_x(tr)
        pygame.draw.rect(surface, CLR_BTN_INACTIVE, tr, border_radius=3)
        if tx > tr.x:
            pygame.draw.rect(surface, CLR_ACCENT,
                             pygame.Rect(tr.x, tr.y, tx - tr.x, tr.h), border_radius=3)
        cy = tr.y + tr.h // 2
        pygame.draw.circle(surface, CLR_BORDER, (tx, cy), self._THUMB_R)
        pygame.draw.circle(surface, CLR_ACCENT, (tx, cy), self._THUMB_R - 2)

    def handle_event(self, event: pygame.event.Event) -> bool:
        tr = self._track_rect()
        hit = tr.inflate(0, self._THUMB_R * 2)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if hit.collidepoint(event.pos):
                self._dragging = True
                self._set_from_mouse(event.pos[0], tr)
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._dragging = False
        elif event.type == pygame.MOUSEMOTION and self._dragging:
            self._set_from_mouse(event.pos[0], tr)
            return True
        return False

    def _set_from_mouse(self, mx: int, tr: pygame.Rect) -> None:
        t = max(0.0, min(1.0, (mx - tr.x) / max(tr.w, 1)))
        self.setter(self.min_val + t * (self.max_val - self.min_val))


class Divider(Widget):
    def __init__(self, x, y, w) -> None:
        super().__init__(x, y, w, 10)

    def draw(self, surface: pygame.Surface, fonts: dict) -> None:
        cy = self.rect.centery
        pygame.draw.line(surface, CLR_BORDER, (self.rect.x, cy), (self.rect.right, cy))


class SectionHeader(Widget):
    def __init__(self, x, y, w, text: str) -> None:
        super().__init__(x, y, w, 22)
        self.text = text

    def draw(self, surface: pygame.Surface, fonts: dict) -> None:
        lbl = fonts["title"].render(self.text.upper(), True, CLR_TEXT_DIM)
        surface.blit(lbl, (self.rect.x, self.rect.y))
        cy = self.rect.bottom - 1
        pygame.draw.line(surface, CLR_BORDER, (self.rect.x, cy), (self.rect.right, cy))
# endregion


class TabbedPanel:
    """Панель с вкладками"""

    PADDING = 10
    ITEM_GAP = 6
    TAB_H = 28
    SCROLL_W  = 6  # ширина полосы прокрутки
    SCROLL_SPD = 25  # пикселей за тик колёсика

    def __init__(self, x: int, y: int, w: int, h: int,
                 tab_names: List[str]) -> None:
        self.rect = pygame.Rect(x, y, w, h)
        self.tab_names = tab_names
        self._active = 0

        self._tabs: dict = {n: [] for n in tab_names}
        self._factories: dict = {}
        self._factory_cache: dict = {}

        self._inspector: Optional[List[Widget]] = None
        self._scroll_y = 0  # текущее смещение прокрутки
        self._content_h = 0  # высота контента (для скроллбара)
        self._surface = pygame.Surface((w, h))

        self._sb_dragging = False
        self._sb_drag_y0 = 0
        self._sb_scroll_y0 = 0

    def add(self, tab_name: str, widget: Widget) -> None:
        self._tabs[tab_name].append(widget)

    def set_tab_factory(self, tab_name: str, factory: Callable) -> None:
        self._factories[tab_name] = factory
        self._factory_cache.pop(tab_name, None)

    def invalidate_tab(self, tab_name: str) -> None:
        self._factory_cache.pop(tab_name, None)

    def show_inspector(self, widgets: List[Widget]) -> None:
        self._inspector = widgets
        self._scroll_y  = 0

    def close_inspector(self) -> None:
        self._inspector = None
        self._scroll_y  = 0

    @property
    def _active_widgets(self) -> List[Widget]:
        if self._inspector is not None:
            return self._inspector
        name = self.tab_names[self._active]
        if name in self._factories:
            if name not in self._factory_cache:
                self._factory_cache[name] = self._factories[name]()
            return self._factory_cache[name]
        return self._tabs.get(name, [])

    def _content_top(self) -> int:
        return self.PADDING // 2 + self.TAB_H + self.ITEM_GAP

    def _content_area_h(self) -> int:
        return self.rect.h - self._content_top()

    def _layout(self, widgets: List[Widget]) -> int:
        y        = self._content_top() + self.PADDING
        inner_w  = self.rect.w - self.PADDING * 2 - self.SCROLL_W - 2
        for w in widgets:
            w.rect.w = inner_w
            w.move_to(self.PADDING, y)
            y += w.rect.h + self.ITEM_GAP
        return y

    def _max_scroll(self) -> int:
        return max(0, self._content_h - self._content_area_h())

    def draw(self, screen: pygame.Surface, fonts: dict) -> None:
        self._surface.fill(CLR_BG)
        pygame.draw.rect(self._surface, CLR_BORDER,
                         pygame.Rect(0, 0, self.rect.w, self.rect.h), 1)

        # Вкладки
        self._draw_tabs(fonts)

        # Виджеты
        widgets = self._active_widgets
        self._content_h = self._layout(widgets) + self.PADDING  # total content height

        # Clip region
        ct = self._content_top()
        clip = pygame.Rect(0, ct, self.rect.w, self._content_area_h())
        self._surface.set_clip(clip)

        for w in widgets:
            saved = w.rect.y
            w.rect.y -= self._scroll_y
            w.draw(self._surface, fonts)
            w.rect.y = saved

        self._surface.set_clip(None)

        self._draw_scrollbar()

        screen.blit(self._surface, (self.rect.x, self.rect.y))

    def _draw_tabs(self, fonts: dict) -> None:
        n = len(self.tab_names)
        tw = (self.rect.w - self.PADDING * 2) // n
        ty = self.PADDING // 2
        for i, name in enumerate(self.tab_names):
            r = pygame.Rect(self.PADDING + i * tw, ty, tw - 2, self.TAB_H)
            is_active = (i == self._active) and self._inspector is None
            if self._inspector is not None and i == self._active:
                bg = (38, 42, 52)  # затемняем активную вкладку
            else:
                bg = CLR_ACCENT if is_active else CLR_BTN_INACTIVE
            pygame.draw.rect(self._surface, bg, r, border_radius=4)
            lbl = fonts["label"].render(name, True,
                                        (255, 255, 255) if is_active else CLR_TEXT_DIM)
            self._surface.blit(lbl, (r.centerx - lbl.get_width() // 2,
                                      r.centery - lbl.get_height() // 2))

    def _draw_scrollbar(self) -> None:
        max_s = self._max_scroll()
        if max_s <= 0:
            return
        ct   = self._content_top()
        cah  = self._content_area_h()
        sb_x = self.rect.w - self.SCROLL_W - 1
        # Трек
        pygame.draw.rect(self._surface, (35, 38, 48),
                         pygame.Rect(sb_x, ct, self.SCROLL_W, cah), border_radius=3)
        # Ползунок
        ratio    = cah / max(self._content_h, 1)
        thumb_h  = max(20, int(cah * ratio))
        thumb_y  = ct + int(self._scroll_y / max_s * (cah - thumb_h))
        pygame.draw.rect(self._surface, CLR_BORDER,
                         pygame.Rect(sb_x, thumb_y, self.SCROLL_W, thumb_h), border_radius=3)

    def handle_event(self, event: pygame.event.Event) -> bool:
        offset    = pygame.Vector2(self.rect.topleft)
        translated = _translate_event(event, -offset)
        if translated is None:
            return False

        lx = translated.pos[0] if hasattr(translated, "pos") else 0
        ly = translated.pos[1] if hasattr(translated, "pos") else 0
        ct  = self._content_top()
        max_s = self._max_scroll()

        sb_x = self.rect.w - self.SCROLL_W - 1
        sb_rect = pygame.Rect(sb_x, ct, self.SCROLL_W + 4, self._content_area_h())

        if translated.type == pygame.MOUSEBUTTONDOWN and translated.button == 1:
            if sb_rect.collidepoint(lx, ly):
                self._sb_dragging  = True
                self._sb_drag_y0   = ly
                self._sb_scroll_y0 = self._scroll_y
                return True

        if translated.type == pygame.MOUSEBUTTONUP and getattr(translated, "button", 0) == 1:
            self._sb_dragging = False

        if translated.type == pygame.MOUSEMOTION and self._sb_dragging:
            cah   = self._content_area_h()
            ratio = self._content_h / max(cah, 1)
            self._scroll_y = int(self._sb_scroll_y0 + (ly - self._sb_drag_y0) * ratio)
            self._scroll_y = max(0, min(max_s, self._scroll_y))
            return True

        # Колёсико
        if translated.type == pygame.MOUSEWHEEL:
            self._scroll_y -= translated.y * self.SCROLL_SPD
            self._scroll_y  = max(0, min(max_s, self._scroll_y))
            return True

        # Переключение вкладок
        n  = len(self.tab_names)
        tw = (self.rect.w - self.PADDING * 2) // n
        ty = self.PADDING // 2
        if self._inspector is None and translated.type == pygame.MOUSEBUTTONDOWN and translated.button == 1:
            for i in range(n):
                r = pygame.Rect(self.PADDING + i * tw, ty, tw - 2, self.TAB_H)
                if r.collidepoint(lx, ly):
                    if i != self._active:
                        self._active   = i
                        self._scroll_y = 0
                    return True

        # Виджеты
        if ly < ct:
            return False  # клик в зону вкладок
        for w in self._active_widgets:
            saved    = w.rect.y
            w.rect.y -= self._scroll_y
            visible  = w.rect.bottom > ct and w.rect.top < ct + self._content_area_h()
            result   = w.handle_event(translated) if visible else False
            w.rect.y = saved
            if result:
                return True
        return False


def _translate_event(event: pygame.event.Event,
                     offset: pygame.Vector2) -> Optional[pygame.event.Event]:
    if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP,
                      pygame.MOUSEMOTION, pygame.MOUSEWHEEL):
        d = event.__dict__.copy()
        if "pos" in d:
            d["pos"] = (d["pos"][0] + int(offset.x), d["pos"][1] + int(offset.y))
        return pygame.event.Event(event.type, d)
    return None
