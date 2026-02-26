"""
Это файл с примерами сцен, которые могут объяснить некоторые функции этой программы
"""


from simulation import Simulation
from portals import *
from masks import *


def _anchors(sim_width, sim_height):
    """Специальный пресет с якорями потенциал так, чтобы гравитация была как на земле"""
    return [
        PotentialAnchor(RectangleMask(0, sim_width, 0, 1), 1.0),
        PotentialAnchor(RectangleMask(0, sim_width, sim_height-1, sim_height), 0.0),
    ]


def example_couple_portals() -> Simulation:
    """Два параллельных портала"""
    W, H = 120, 120
    p1 = Portal(RectangleMask(40, 80, 40, 40), color=(255, 153, 0))
    p2 = Portal(RectangleMask(40, 80, 80, 80), color=(0, 204, 255))

    return Simulation(
        *_anchors(W, H), CouplePortal(p1, p2),
        sim_width=W, sim_height=H,
        px_scale=6,
        iterations_per_frame=40,
        sor_omega=1.7,
    )


def example_advanced() -> Simulation:
    """FixedPotential + объект"""
    W, H = 120, 100
    top = FixedPotentialPortal(RectangleMask(20, 100, 30, 33), 0.8, (255, 80, 80))
    bot = FixedPotentialPortal(RectangleMask(20, 100, 67, 70), 0.2, (80, 120, 255))
    obs = MaterialObject(CircleMask(60, 50, 8),
                         color=(80, 220, 120), pinned=True, label="Conductor")
    return Simulation(
        top, bot, obs,
        sim_width=W, sim_height=H,
        px_scale=6,
        iterations_per_frame=60,
        sor_omega=1.75,
    )


def example_couple_circles() -> Simulation:
    """Два круглых портала"""
    W, H = 120, 100
    p1 = Portal(CircleMask(cx=35, cy=50, radius=10), color=(255, 100, 50))
    p2 = Portal(CircleMask(cx=85, cy=50, radius=10), color=(50, 200, 255))

    return Simulation(
        *_anchors(W, H), CouplePortal(p1, p2),
        sim_width=W, sim_height=H,
        px_scale=6,
        iterations_per_frame=50,
        sor_omega=1.8,
        isoline_count=15,
    )


def triple_portals() -> Simulation:
    W, H = 120, 100
    p1 = Portal(RectangleMask(40, 60, 30, 30), (255, 0, 0))
    p2 = Portal(RectangleMask(40, 60, 60, 60), (0, 255, 0))
    p3 = Portal(RectangleMask(40, 60, 90, 90), (0, 0, 255))

    return Simulation(
        *_anchors(W, H), MultiPortal((p1, p2, p3)),
        sim_width=W, sim_height=H,
        px_scale=6,
        iterations_per_frame=50,
        sor_omega=1.8,
        isoline_count=15,
    )
