"""AChA: attach lyric syllables to notes by their printed position (opt in: STAFFREADER_ACHA_LYRICS=hybrid).

Measured 2026-11-10 on the Score Engine benchmark, clean page 1 of the Hallelujah fixture: the
reader's own in-order `lyrics.attach` puts 85% of syllables on the right note, positional alone 75%,
the hybrid below 81%. So in-order stays the default and this module is opt in; it is kept because it
is the only path that can survive a staff whose note marks the reader missed.

staffreader's `lyrics.attach` zips a verse's syllables onto the staff's 'ly'-marked notes in
order, so one missed or extra note mark shifts every later syllable in the verse. Printed lyrics
carry their own alignment: each syllable is set under the note it is sung to. This module reads
each lyric row as separate word-boxes (connected components across the row, merged by the gaps
between letters), reads every box with staffreader's own text reader, splits the text into
syllables, and gives each syllable to the nearest placed notehead whose x is at or after the
syllable's left edge, in order, never going backwards.

Needs the notes to carry 'x' (positions.place has run). Falls back to lyrics.attach for a
staff whose notes have no positions.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from . import lyrics, text


def word_boxes(level: np.ndarray, staff, row: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    """Boxes of the printed words on one lyric row, left to right, in page pixels."""
    top, bottom = row
    x0, x1 = int(staff.x0), int(staff.x1)
    band = (level[top:bottom, x0:x1] < 128).astype(np.uint8)
    sp = staff.space
    # Join letters of one word, keep words apart: close across about a third of a space.
    gap = max(2, int(0.35 * sp))
    joined = cv2.morphologyEx(band, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (gap, 1)))
    # Also join a word's letters vertically (dots on i, accents) without joining to the next row.
    joined = cv2.morphologyEx(joined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(2, int(0.5 * sp)))))
    n, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    boxes = []
    for k in range(1, n):
        bx, by, bw, bh, area = stats[k]
        if bw < 0.25 * sp or area < 0.1 * sp * sp:
            continue  # a speck or a stray hyphen remnant
        boxes.append((x0 + int(bx), top + int(by), x0 + int(bx + bw), top + int(by + bh)))
    boxes.sort()
    # Merge boxes that overlap or nearly touch (a hyphenated syllable's hyphen counts with its syllable).
    merged: list[list[int]] = []
    for b in boxes:
        if merged and b[0] - merged[-1][2] < 0.2 * sp:
            merged[-1][2] = max(merged[-1][2], b[2]); merged[-1][1] = min(merged[-1][1], b[1]); merged[-1][3] = max(merged[-1][3], b[3])
        else:
            merged.append(list(b))
    return [tuple(m) for m in merged]


def joins_after(level: np.ndarray, staff, row: tuple[int, int], boxes) -> list[str]:
    """For each box, '-' when a hyphen stands between it and the next box, '_' when an extender
    line does (longer than a hyphen), else ''. A hyphen is a short flat run of ink near the row's
    x-height centre in the gap between two words; printed lyrics hyphenate every split word."""
    top, bottom = row
    out = []
    sp = staff.space
    for k, (x0, y0, x1, y1) in enumerate(boxes):
        nxt = boxes[k + 1][0] if k + 1 < len(boxes) else int(staff.x1)
        gap = level[y0:y1, x1 + 1:max(x1 + 2, nxt - 1)] < 128
        if gap.size == 0:
            out.append(''); continue
        rows_ink = gap.sum(axis=1)
        if rows_ink.max() == 0:
            out.append(''); continue
        # a hyphen: ink confined to a thin band of rows, at least a few pixels long, in the middle part
        h = gap.shape[0]
        band_rows = np.flatnonzero(rows_ink > 0)
        thickness = band_rows.max() - band_rows.min() + 1
        length = gap.any(axis=0).sum()
        mid = 0.25 * h <= band_rows.mean() <= 0.8 * h
        if thickness <= max(2, 0.25 * sp) and mid and length >= 0.15 * sp:
            out.append('_' if length >= 1.5 * sp else '-')
        else:
            out.append('')
    return out


def read_boxes(level: np.ndarray, staff, boxes) -> list[str]:
    net = lyrics.text_reader()
    if net is None or not boxes:
        return []
    pad = int(0.4 * staff.space)
    views = [text.view(level[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]) for x0, y0, x1, y1 in boxes]
    x, w = text.batch(views, next(net.parameters()).device)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        return net.read(x, w)


def attach_by_position(bars, level: np.ndarray, staff, rows: list[tuple[int, int]], barlines=None) -> dict:
    """Bar-anchored: each syllable's x falls in a bar (between barlines); inside the bar, syllables go to the
    bar's singable notes in order. A bar that holds more syllables than notes keeps the first ones; a bar
    with fewer leaves its last notes bare (a melisma). Notes need no x positions. Returns counts."""
    stats = {'rows': len(rows), 'boxes': 0, 'syllables': 0, 'placed': 0, 'fallback': False}
    ends = list(barlines or [])
    if not ends:
        stats['fallback'] = True
        lyrics.attach(bars, lyrics.read_rows(level, staff, rows))
        return stats
    starts = [staff.x0] + ends[:-1]
    sp = staff.space
    for v, row in enumerate(rows, start=1):
        boxes = word_boxes(level, staff, row)
        texts = read_boxes(level, staff, boxes)
        joins = joins_after(level, staff, row, boxes)
        stats['boxes'] += len(boxes)
        # A box that is only punctuation (a comma or full stop the finder split off, often read as
        # 'y' or '.') belongs to the word before it; a box the hyphen detector explains as a hyphen
        # read as a dot loses that dot.
        items_in: list[tuple[tuple, str, str]] = []
        for (box, t, j) in zip(boxes, texts, joins):
            t = t.strip()
            if not t or not any(c.isalnum() for c in t):
                # a hyphen or extender the finder boxed on its own: it joins the word before
                if items_in and '-' in t:
                    pb, pt, pj = items_in[-1]; items_in[-1] = (pb, pt, '-')
                elif items_in and '_' in t:
                    pb, pt, pj = items_in[-1]; items_in[-1] = (pb, pt, '_')
                continue
            narrow = (box[2] - box[0]) < 0.7 * sp
            if narrow and items_in and (not any(c.isalpha() for c in t) or t in ('y', 'i', 'l', '1')):
                pb, pt, pj = items_in[-1]
                mark = t if not any(c.isalpha() for c in t) else ','
                items_in[-1] = ((pb[0], pb[1], box[2], pb[3]), pt + mark, j)
                continue
            if j == '-' and t.endswith('.'):
                t = t[:-1]
            items_in.append((box, t, j))
        per_bar: dict[int, list] = {}
        for (x0, _, x1, _), t, j in items_in:
            if j == '-' and not t.endswith('-'):
                t += '-'
            elif j == '_' and not t.endswith('_'):
                t += '_'
            syls = lyrics.syllables(t)
            if not syls:
                continue
            step = (x1 - x0) / max(1, len(syls))
            for k, syl in enumerate(syls):
                x = x0 + k * step + 0.3 * sp  # the syllable's text starts a little left of its note
                b = next((i for i, e in enumerate(ends) if x < e), len(ends) - 1)
                b = max(0, min(b, len(bars) - 1))
                per_bar.setdefault(b, []).append((x, syl))
        for b, items in per_bar.items():
            items.sort(key=lambda it: it[0])
            stats['syllables'] += len(items)
            # Singable notes: pitched, not grace, and not the continuation of a tie from the note before.
            notes, prev_tied = [], (b > 0 and any(n.get('tie') for n in bars[b - 1].notes[-1:]))
            for n in bars[b].notes:
                if n.get('rest') or n.get('grace') or not n.get('pitch'):
                    continue
                if not prev_tied:
                    notes.append(n)
                prev_tied = bool(n.get('tie'))
            # Prefer the notes the reader itself marked as taking a syllable, when it marked enough of them.
            marked = [n for n in notes if 'ly' in n.get('marks', [])]
            targets = marked if len(marked) >= len(items) else notes
            for (x, (word, syllabic, held)), n in zip(items, targets):
                n.setdefault('lyrics', []).append((v, word, syllabic, held))
                stats['placed'] += 1
    # Fix syllabic labels from the hyphen flags, in note order per verse.
    for v in range(1, len(rows) + 1):
        going = False
        for bar in bars:
            for n in bar.notes:
                for idx, (vv, word, syllabic, held) in enumerate(n.get('lyrics', [])):
                    if vv != v:
                        continue
                    more = syllabic in ('begin', 'middle')
                    fixed = ('middle' if more else 'end') if going else ('begin' if more else 'single')
                    n['lyrics'][idx] = (vv, word, fixed, held)
                    going = more
    return stats


def attach_hybrid(bars, level: np.ndarray, staff, rows: list[tuple[int, int]], barlines=None) -> dict:
    """lyrics.attach first (its in-order placement is right where the reader marked every sung note);
    then bars where the printed syllable count under the bar differs from what landed there are
    re-attached from the printed positions. Measured 2026-11-10 on the benchmark's clean page 1:
    in-order 85% of syllables on the right note, positional alone 75%; the hybrid keeps the best of
    each bar."""
    import copy
    verses = lyrics.read_rows(level, staff, rows)
    lyrics.attach(bars, verses)
    stats = {'rows': len(rows), 'bars_repositioned': 0, 'fallback': False}
    ends = list(barlines or [])
    if not ends:
        stats['fallback'] = True
        return stats
    # positional reading into a scratch copy
    scratch = copy.deepcopy(bars)
    for bar in scratch:
        for n in bar.notes:
            n.pop('lyrics', None)
    pos = attach_by_position(scratch, level, staff, rows, barlines=ends)
    stats['positional'] = pos
    for b, (bar, alt) in enumerate(zip(bars, scratch)):
        have = sum(1 for n in bar.notes for l in n.get('lyrics', []) if l[0] == 1)
        want = sum(1 for n in alt.notes for l in n.get('lyrics', []) if l[0] == 1)
        if want and have != want:
            for n, m in zip(bar.notes, alt.notes):
                if 'lyrics' in m:
                    n['lyrics'] = m['lyrics']
                else:
                    n.pop('lyrics', None)
            stats['bars_repositioned'] += 1
    # relabel syllabic in order across the staff
    going = False
    for bar in bars:
        for n in bar.notes:
            for idx, (vv, word, syllabic, held) in enumerate(n.get('lyrics', [])):
                if vv != 1:
                    continue
                more = syllabic in ('begin', 'middle')
                n['lyrics'][idx] = (vv, word, ('middle' if more else 'end') if going else ('begin' if more else 'single'), held)
                going = more
    return stats
