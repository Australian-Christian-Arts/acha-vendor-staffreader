"""AChA shim: build a staffreader Layout from Score Engine's own staff geometry.

staffreader's row-profile staff finder gives up on photocopy-grade pages where our
container/page_geometry.py still finds every staff. This module takes our staff tops and
bottoms, synthesises the five line heights, then reuses staffreader's own system, part and
barline finders on its binarised ink so the rest of the reader runs unchanged.

Enable with STAFFREADER_ACHA_GEOMETRY=1 (and ACHA_CONTAINER_DIR pointing at the Score Engine
container directory). Prototype for the 2026-11-09 benchmark; not shipped.
"""
import os
import sys

import cv2
import numpy as np

from . import layout as L


def _acha_geometry():
    container = os.environ.get('ACHA_CONTAINER_DIR')
    if container and container not in sys.path:
        sys.path.insert(0, container)
    import page_geometry  # noqa: WPS433  (Score Engine module)
    return page_geometry


def read(gray: np.ndarray, image_path: str | None = None) -> L.Layout:
    pg = _acha_geometry()
    if image_path is None:
        tmp = os.path.join(L._scratch(), 'acha-geom-page.png')
        cv2.imwrite(tmp, gray)
        image_path = tmp
    geo = pg.read_page(image_path)
    ink, angle = L.straighten(L.binarize(gray))
    line = max(1, int(geo.get('line_thickness_px') or 2))
    staves = []
    h, w = gray.shape
    for top, bottom in sorted(geo['staves']):
        lines = list(np.linspace(float(top), float(bottom), 5))
        # Horizontal extent: where this staff band actually carries ink.
        band = ink[int(max(0, top - 2)):int(min(h, bottom + 3)), :]
        cols = np.flatnonzero(band.sum(axis=0) > 0)
        x0, x1 = (float(cols[0]), float(cols[-1])) if len(cols) else (0.0, float(w))
        staves.append(L.Staff(lines, x0, x1))
    systems = L.find_systems(ink, staves, line)
    # Fall back to our own grouping when the bracket test fails on noise.
    ours = geo.get('systems') or []
    if ours and len(systems) != len(ours):
        systems = [L.System(list(s)) for s in ours]
    L.find_parts(ink, staves, systems)
    L.find_barlines(ink, staves, systems, line)
    return L.Layout(w, h, staves, systems, angle, None)
