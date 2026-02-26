"""
Основной скрипт симуляции

Objects:
  CouplePortal(p1, p2)           пара порталов с равным потенциалом
  FixedPotentialPortal(mask, v)  область с фиксированным потенциалом
  PotentialAnchor(mask, v)       якорь потенциала

Masks:
  RectangleMask(x_min, x_max, y_min, y_max)
  CircleMask(cx, cy, radius)
  PointMask(x, y)
  LineMask(x1, y1, x2, y2, thickness)
  PolygonMask([(x0,y0), ...])
  FunctionMask("выражение(x, y)")

Controls:
  M - режима отображения гравитационное ускорение / потенциал
  V - векторы вкл/выкл
  I - изолинии вкл/выкл

  Drag - перетащить любой портал мышью
  Вкладка SIM - параметры рендера и физики
  Вкладка SCENE - объекты сцены, добавление, пресеты, инспектор

Подробнее в README.md
"""


from scenes import *


def main() -> None:
    #sim = example_couple_portals()  # Стартовая сцена
    sim = triple_portals()
    sim.run()


if __name__ == "__main__":
    main()
