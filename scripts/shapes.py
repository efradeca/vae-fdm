"""
Prescribed shapes for the shell and tower tasks.
"""

# ===============================================================================
# Shell task - These shapes need of a `bezier_symmetric_double` generator.
# ===============================================================================

# pillow
BEZIER_PILLOW = [
    [0.0, 0.0, 10.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0]
]

# circular dome
BEZIER_DOME = [
    [0.0, 0.0, 10.0],
    [2.75, 0.0, 0.0],
    [0.0, 2.75, 0.0],
    [0.0, 0.0, 0.0]
]

# cute saddle
BEZIER_SADDLE = [
    [0.0, 0.0, 1.5],
    [-1.25, 0.0, 5.0],
    [0.0, -2.5, 0.0],
    [0.0, 0.0, 0.0]
]

# cute hypar
BEZIER_HYPAR = [
    [0.0, 0.0, 1.5],
    [-1.25, 0.0, 7.5],
    [0.0, 1.25, 0.0],
    [0.0, 0.0, 0.0]
]

# cute pringle
BEZIER_PRINGLE = [
    [0.0, 0.0, 1.5],
    [1.25, 1.25, 0.0],
    [-1.25, 0.0, 7.5],
    [0.0, 0.0, 0.0]
]

# cannon vault
BEZIER_CANNON = [
    [0.0, 0.0, 6.0],
    [0.0, 0.0, 6.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0]
]

BEZIERS = {
    "pillow": BEZIER_PILLOW,
    "dome": BEZIER_DOME,
    "saddle": BEZIER_SADDLE,
    "hypar": BEZIER_HYPAR,
    "pringle": BEZIER_PRINGLE,
    "cannon": BEZIER_CANNON,
}

# ===============================================================================
# Tower task
# ===============================================================================

TOWER_ANGLES = [0.0, 0.0, 0.0]
TOWER_RADII_FIXED = [0.75, 0.75]
TOWER_RADII = [TOWER_RADII_FIXED, [0.75, 0.75], TOWER_RADII_FIXED]

TOWERS = {
    -30: [TOWER_RADII, [0.0, -30.0, 0.0]],
    -22: [TOWER_RADII, [0.0, -22.0, 0.0]],
    -15: [TOWER_RADII, [0.0, -15.0, 0.0]],
    -7: [TOWER_RADII, [0.0, -7, 0.0]],
    0: [TOWER_RADII, [0.0, 0.0, 0.0]],
    7: [TOWER_RADII, [0.0, 7.0, 0.0]],
    15: [TOWER_RADII, [0.0, 15.0, 0.0]],
    22: [TOWER_RADII, [0.0, 22.0, 0.0]],
    30: [TOWER_RADII, [0.0, 30.0, 0.0]],
    0.5: [[TOWER_RADII_FIXED, [0.5, 0.5], TOWER_RADII_FIXED], TOWER_ANGLES],
    0.75: [[TOWER_RADII_FIXED, [0.75, 0.75], TOWER_RADII_FIXED], TOWER_ANGLES],
    1.0: [[TOWER_RADII_FIXED, [1.0, 1.0], TOWER_RADII_FIXED], TOWER_ANGLES],
    1.25: [[TOWER_RADII_FIXED, [1.25, 1.25], TOWER_RADII_FIXED], TOWER_ANGLES],
    1.5: [[TOWER_RADII_FIXED, [1.5, 1.5], TOWER_RADII_FIXED], TOWER_ANGLES],

}
