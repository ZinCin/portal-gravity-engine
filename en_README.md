# Portal Gravity Engine

Interactive potential field simulation. Solves the Laplace equation using the SOR (Successive Over-Relaxation) method. See `examples/explanation.pdf` for details.

You can place anchors, portals, and obstacles on the grid and observe the result.

The project was created more as an experiment with a new computation method — Black-Red SOR — than as a general portal sandbox.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Physics](#physics)
- [Project Architecture](#project-architecture)
- [Field Objects](#field-objects)
- [Masks](#masks)
- [Simulation Parameters](#simulation-parameters)
- [Controls](#controls)
- [Color Schemes](#color-schemes)
- [Writing Scenes](#writing-scenes)

---

## Quick Start

**Requirements:** Python 3.9+, numpy 2.0.2+, pygame 2.6.1+

**To run normally, clone the repository (or just the `./src` folder), then launch `main.py` from the `./src` directory:**

```bash
pip install numpy pygame
python main.py
```

To run a different scene, open `main.py` and uncomment the desired line:

```python
def main() -> None:
    sim = example_couple_portals()   # <- active scene
    # sim = example_advanced()
    # sim = example_couple_circles()
    sim.run()
```

or switch the scene in the inspector.

---

## Physics

The simulation solves the Laplace equation:

```
∇²φ = 0
```

Method: **Red-Black Gauss-Seidel SOR** — alternates between "red" and "black" checkerboard cells, which doubles convergence speed compared to plain Jacobi.

### Boundary Conditions

| Boundary | Type | Value |
|---|---|---|
| Top of grid | Dirichlet | φ = 1.0 |
| Bottom of grid | Dirichlet | φ = 0.0 |
| Left / right | Neumann | ∂φ/∂n = 0 |

The top and bottom edges are constant source and sink. The side boundaries are transparent: the field flows freely left and right. (This implements gravity similar to Earth's.)

### Initial State

The potential is initialized with a linear gradient φ = 1 (top) → 0 (bottom). This is the exact solution to Laplace's equation under the chosen boundary conditions, so without any objects the field will not change. Objects create permanent deviations from this background.

### Parameter ω (SOR)

Controls convergence speed:

- `ω = 1.0` — slow but accurate convergence
- `ω = 1.5–1.9` — fast but less accurate
- `ω → 2.0` — near-instant but unstable

---

## Project Architecture

```
main.py          — entry point
scenes.py        — predefined scenes (examples demonstrating how portals work)
simulation.py    — main class, game loop, rendering, UI
physics.py       — SOR solver
portals.py       — field object classes
masks.py         — geometric masks
colors.py        — color schemes
ui.py            — sidebar widgets
```

> `physics.py` has no Pygame dependency, for use in standalone tests.

---

## Field Objects

All objects are passed into `Simulation(*field, ...)` in any order and quantity.

---

### `PotentialAnchor(mask, potential_value)`

An anchor: rigidly fixes the potential in its area and holds it constant. Creates a field source or sink at an arbitrary point on the grid.

```python
# Source at center-top
PotentialAnchor(CircleMask(60, 20, 5), potential_value=1.0)

# Sink at center-bottom
PotentialAnchor(CircleMask(60, 80, 5), potential_value=0.0)
```

An anchor with φ above the background creates a potential gradient hill; with φ below — a potential gradient valley. The field decays from the anchor in all directions according to the Laplace equation.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `mask` | `Mask` | Shape of the anchor area |
| `potential_value` | `float` | Fixed φ value ∈ [0, 1] |
| `color` | `tuple` | RGB color |
| `active` | `bool` | Enabled / disabled |

---

### `FixedPotentialPortal(mask, potential_value)`

Functionally identical to `PotentialAnchor` — fixes the potential in its area. Used for potential shielding.

```python
FixedPotentialPortal(RectangleMask(20, 100, 30, 33), potential_value=0.8)
```

---

### `CouplePortal(p1, p2)`

A pair of linked portals with **equal potential**. At each step, the potential across the entire linked area is averaged.

```python
p1 = Portal(RectangleMask(25, 75, 25, 26), color=(255, 153, 0))
p2 = Portal(RectangleMask(25, 75, 94, 95), color=(0, 204, 255))
couple = CouplePortal(p1, p2)
```

**`Portal` Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `mask` | `Mask` | Shape of the portal |
| `color` | `tuple` | RGB color |
| `active` | `bool` | Enabled / disabled |

---

### `MaterialObject(mask, ...)`

A solid obstacle. Cells inside are **excluded from SOR** — the field wraps around the object from outside. The potential inside is fixed at the initial value and is not updated.

```python
# Pinned block
MaterialObject(
    RectangleMask(40, 80, 40, 60),
    color=(180, 180, 180),
    pinned=True,
    label="Block"
)

# Free-floating ball
MaterialObject(
    CircleMask(cx=60, cy=50, radius=8),
    pinned=False,
    mass=1.5
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mask` | `Mask` | — | Shape of the object |
| `color` | `tuple` | `(180, 180, 180)` | RGB color |
| `pinned` | `bool` | `False` | If `True`, cannot be dragged |
| `label` | `str` | `"Object"` | Name in UI |
| `mass` | `float` | `1.0` | Mass |
| `active` | `bool` | `True` | Enabled / disabled |

---

### `ConductorObject(mask, ...)`

A floating conductor. At each step, the potential inside the object is equalized to the average potential of its outer neighboring cells. Creates the field-line distortion characteristic of a conductor.

```python
ConductorObject(
    CircleMask(cx=60, cy=50, radius=10),
    color=(220, 180, 60),
    pinned=True,
    label="Sphere"
)
```

Parameters are identical to `MaterialObject`.

---

## Masks

Masks define the geometric shape of an object on the simulation grid. Any object accepts any mask.

### `RectangleMask(x_min, x_max, y_min, y_max)`

```python
RectangleMask(10, 90, 20, 21)   # thin horizontal strip
RectangleMask(0, 120, 0, 1)     # anchor along the entire top edge
```

### `CircleMask(cx, cy, radius)`

```python
CircleMask(cx=60, cy=50, radius=15)
```

### `PointMask(x, y)`

A single grid cell.

```python
PointMask(60, 50)
```

### `LineMask(x1, y1, x2, y2, thickness)`

A line segment between two points with a given thickness.

```python
LineMask(10, 10, 110, 90, thickness=2.0)
```

### `PolygonMask([(x0,y0), (x1,y1), ...])`

An arbitrary polygon.

```python
PolygonMask([(30, 20), (90, 20), (60, 80)])  # triangle
```

### `FunctionMask("expression")`

A mask defined by an arbitrary expression (function). Variables: `x`, `y`, `np`. The expression must return a boolean scalar or array.

```python
FunctionMask("(x - 60)**2 + (y - 50)**2 < 15**2")  # circle
FunctionMask("np.abs(x - 60) < 5")                  # vertical strip
FunctionMask("(x > 30) & (x < 90) & (y > 40) & (y < 60)")  # rectangle
```

> ⚠️ `PolygonMask` and `FunctionMask` are slower than the others due to complex calculations — avoid using them unless necessary.

---

## Simulation Parameters

```python
Simulation(
    *field,                          # field objects (any number)
    sim_width: int,                  # grid width in cells
    sim_height: int,                 # grid height in cells
    px_scale: float,                 # pixels per cell
    iterations_per_frame: int = 50,  # SOR iterations per frame
    diff_threshold: float = 1e-6,    # early stopping threshold
    view_mode: str = "g_force",      # "g_force" | "potential"
    show_vectors: bool = True,       # gradient vectors
    show_isolines: bool = True,      # isolines
    isoline_count: int = 10,         # number of isolines
    fps: int = 60,                   # target FPS
    sor_omega: float = 1.7,          # SOR parameter [1.0, 2.0)
    color_mapper = None,             # custom color scheme
)
```

**Recommended grid sizes:**

| Resolution | `px_scale` | Load |
|---|---|---|
| 80 × 80 | 8 | light |
| 120 × 120 | 6 | medium (default) |
| 200 × 150 | 4 | heavy |

---

## Controls

### Keyboard

| Key | Action |
|---|---|
| `M` | Toggle display mode: gravitational acceleration / potential |
| `V` | Show / hide gradient vectors |
| `I` | Show / hide isolines |

### Mouse

| Action | Result |
|---|---|
| **LMB + drag** on a portal or object | Move the object |
| **RMB** on an object | Open the inspector in the SCENE panel |

### Sidebar

**SIMULATION tab** — rendering and physics parameters in real time:

- Display mode (force / potential)
- Vectors, isolines, isoline count
- Color scheme
- Iterations per frame, SOR ω

**INSPECTOR tab** — object management:

- Add portal, anchor, object, conductor
- Scene presets
- Selected object inspector (mask parameters, color, φ value)

---

## Color Schemes

Available in `COLOR_SCHEMES`, switchable in the SIMULATION panel:

| Name | Description |
|---|---|
| `Default` | Blue → green → yellow → red (for g_force) |
| `Potential` | Blue → green → red (for potential) |
| `Plasma` | Purple → pink → yellow |
| `Electric` | Black → blue → white |
| `Fire` | Black → red → gold → white |
| `Extra` | Blue → cyan → yellow → red |

A custom scheme is set via `color_mapper`:

```python
from colors import GradientColorMapper

my_mapper = GradientColorMapper([
    (0.0, (0,   0,  50)),   # dark blue at φ = 0
    (0.5, (255, 255, 0)),   # yellow at φ = 0.5
    (1.0, (255,  50, 50)),  # red at φ = 1
])

sim = Simulation(..., color_mapper=my_mapper)
```

---

## Writing Scenes

All scenes are defined in `scenes.py`. A minimal scene:

```python
from simulation import Simulation
from portals import PotentialAnchor, CouplePortal, Portal
from masks import RectangleMask, CircleMask

def my_scene() -> Simulation:
    W, H = 120, 120

    # Anchors define the background field
    top    = PotentialAnchor(RectangleMask(0, W, 0, 1),      1.0)
    bottom = PotentialAnchor(RectangleMask(0, W, H-1, H),    0.0)

    # Two linked portals
    p1 = Portal(CircleMask(30, 60, 10), color=(255, 153, 0))
    p2 = Portal(CircleMask(90, 60, 10), color=(0, 204, 255))

    return Simulation(
        top, bottom, CouplePortal(p1, p2),
        sim_width=W, sim_height=H,
        px_scale=6,
        iterations_per_frame=40,
        sor_omega=1.7,
    )
```

Then in `main.py`:

```python
from scenes import my_scene

def main():
    my_scene().run()
```

### Recipes

**Two anchors instead of grid boundary conditions:**

```python
# Source at top, sink at bottom — field directed downward
PotentialAnchor(RectangleMask(0, W, 0, 1),   1.0)
PotentialAnchor(RectangleMask(0, W, H-1, H), 0.0)
```

**Two parallel screens:**

```python
FixedPotentialPortal(RectangleMask(20, 100, 30, 33), 0.8)
FixedPotentialPortal(RectangleMask(20, 100, 67, 70), 0.2)
```

**Conductor in a screen field:**

```python
ConductorObject(CircleMask(60, 50, 10), pinned=True, label="Sphere")
```

**Non-conducting obstacle:**

```python
MaterialObject(RectangleMask(50, 70, 30, 70), pinned=True, label="Wall")
```

---

**For a detailed explanation of the simulation, read explanation.pdf**
