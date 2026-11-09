"""Our own reader: the layout stage finds the staves, the staff reader
reads each one, and the staves are joined into parts.

The model is the checkpoint STAFFREADER_MODEL names, or the best one of
the newest training run. Its name and step are part of NAME, so the
suite caches each model's readings apart from every other's.

A page's staves are read top to bottom. The layout groups each system's
staves into parts (a braced grand staff is one part on two staves), and
every system of a page is taken to hold the same parts in the same
order, so the k-th group of each system is the k-th part; each staff's
bars follow on, system after system."""

from __future__ import annotations

import collections
import os
import pathlib
from fractions import Fraction

import cv2
import numpy as np
import torch

from .. import settings
from .. import layout, layout_suite, staffcrop, tokens
from .. import lyrics
from .. import model as M
from ..render import CACHE


# The model that reads unless STAFFREADER_MODEL names another: the v3 run
# of 2026-10-16, which reads token format 3 and learned from scans paired
# with PDMX transcriptions. Against v2-mix on every test set it was better
# beyond noise in 24 of 72 slices and worse in none (real solo piano 80.5%
# to 83.6% of bars), and real ladder pages read as before (eval-v3-mix).
DEFAULT = CACHE / 'models/v3-mix/last.pt'


def _checkpoint() -> pathlib.Path | None:
    """The model to read with, or None on a machine that has none: the
    module's own logic still serves, and reading says what is missing."""
    named = os.environ.get('STAFFREADER_MODEL')
    if named:
        return pathlib.Path(named)
    return DEFAULT if DEFAULT.exists() else None


CHECKPOINT = _checkpoint()
_STATE = torch.load(CHECKPOINT, map_location='cpu', weights_only=True) if CHECKPOINT else {}
STEP = _STATE.get('step', 0)
# A checkpoint trained on both tasks reads each staff twice, the notes
# alone and then the marks, and the marks are put on the first reading's
# notes: weaving marks into a reading cost note accuracy (E8, E8c).
TASKS = bool(_STATE.get('tasks'))
del _STATE
# Greedy, no bar past twice its time signature; braced staves one part; one
# key per system; as many bars as the layout counts; an overfull bar read
# again as triplets (Reader.fits) within TRIPLET_MARGIN.
TRIPLET_MARGIN = float(settings.get('triplet_margin'))
# The word finder (words.py) the reader uses, where the word_finder setting
# names one: the directions in words on a page are then found and read by
# the text reader in place of the staff reader's spelling.
WORD_FINDER = settings.path('word_finder')
WORDS_REPLACE_ALL = settings.get('words_replace_all') in ('1', 'true', 'yes')

DECODER = 'bars2-parts-key2-count' + ('-tasks' if TASKS else '') + (f'-triplets{TRIPLET_MARGIN:g}' if TRIPLET_MARGIN >= 0 else '') + '-norepeat-labels2-repeats-mend-meter-pagemeter' + (f'-words{"all" if WORDS_REPLACE_ALL else ""}-{WORD_FINDER.parent.name}' if WORD_FINDER else '')
# Nats a triplet reading of an overfull bar may fall short of the reading it
# replaces. The model cannot see a "3" that is not printed, so it always
# finds the triplets less likely; E11 (experiments/e11_refit.py) chose 12 on
# held-out staves: quiet triplets 46.8% to 58.6% of bars exact, real staves
# level. The triplet_margin setting changes it.
NAME = f'staffreader-{CHECKPOINT.parent.name if CHECKPOINT else "none"}-{STEP}-{DECODER}'
_NET = None


def net() -> M.Reader:
    global _NET
    if _NET is None:
        if CHECKPOINT is None:
            raise RuntimeError(f'no model at {DEFAULT}: run `staffreader fetch`, or set STAFFREADER_MODEL')
        _NET = M.load(CHECKPOINT)  # on the device setting's device (model.DEVICE)
    return _NET


def agree_on_key(crops: list, seqs: list[list[str]], limits: list | None = None, before: list | None = None,
                 floor: float = 0.05, known: float = 0.005, counts: list | None = None,
                 task: str | None = None) -> list[list[str]]:
    """One key across a system's staves. A staff whose key differs from
    the one most staves read is read again with that key, but only where
    the model itself gave that key a fair chance: [floor], or the lower
    [known] where the same part had that key in the system before
    ([before]). A transposing instrument, whose own key the model reads
    with confidence and keeps from system to system, keeps it."""
    keys = [next((t for t in seq if t.startswith('key:')), None) for seq in seqs]
    counts = collections.Counter(k for k in keys if k)
    if not counts:
        return seqs
    key, n = counts.most_common(1)[0]
    if n < 2 or n <= len(seqs) / 2:
        return seqs
    odd = [i for i, k in enumerate(keys) if k and k != key and seqs[i] and seqs[i][0].startswith('clef:')]
    if not odd:
        return seqs
    reader = net()
    device = next(reader.parameters()).device
    views = [M.window(crops[i]) for i in odd]
    x, widths = M.batch_images(views, device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        lead = [f'task:{task}'] if task else []
        chance = reader.chances(x, widths, [lead + [seqs[i][0]] for i in odd])[:, M.INDEX[key]]
    redo = [(i, k) for k, i in enumerate(odd)
            if chance[k].item() >= (known if before and before[i] == key else floor)]
    if not redo:
        return seqs
    x, widths = M.batch_images([views[k] for _, k in redo], device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        again = reader.read(x, widths, bar_limits=[limits[i] for i, _ in redo] if limits else None,
                            prefixes=[lead + [seqs[i][0], key] for i, _ in redo],
                            bar_counts=[counts[i] for i, _ in redo] if counts else None)
    out = list(seqs)
    for (i, _), seq in zip(redo, again):
        out[i] = [t for t in seq if not t.startswith('task:')]
    return out


def triplets(crops: list, seqs: list[list[str]], limits: list | None = None, task: str | None = None,
             margin: float | None = None) -> list[list[str]]:
    """Each reading with its overfull bars read again as triplets, where
    the model finds that within [margin] of what it read (Reader.fits)."""
    reader = net()
    device = next(reader.parameters()).device
    x, widths = M.batch_images([M.window(c) for c in crops], device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        fits = reader.fits(x, widths, seqs, limits=limits, lead=[[f'task:{task}']] * len(seqs) if task else None,
                           most=1.0, skip=M.meter_changes(seqs, limits))
    return [M.apply_fits(seq, f, TRIPLET_MARGIN if margin is None else margin) for seq, f in zip(seqs, fits)]


def read_staves(crops: list, batch: int = 12, limits: list | None = None,
                counts: list | None = None, task: str | None = None) -> list[list[str]]:
    """Each stored-size staff crop's tokens; [limits] are the bar lengths
    in force where each staff begins, where known, and [counts] the bars
    the layout found on each."""
    reader = net()
    device = next(reader.parameters()).device
    views = [M.window(c) for c in crops]
    order = sorted(range(len(views)), key=lambda i: views[i].shape[1])  # like widths together
    out: list[list[str] | None] = [None] * len(views)
    for k in range(0, len(order), batch):
        idx = order[k:k + batch]
        x, widths = M.batch_images([views[i] for i in idx], device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            got = reader.read(x, widths, bar_limits=[limits[i] for i in idx] if limits else None,
                              bar_counts=[counts[i] for i in idx] if counts else None,
                              prefixes=[[f'task:{task}']] * len(idx) if task else None)
            got = [[t for t in g if not t.startswith('task:')] for g in got]
        for i, seq in zip(idx, got):
            out[i] = seq
    return out


class Work:
    """What reading a work carries from page to page: its parts' bars
    (part -> staff -> bars), the instruments its labels named, each staff's
    time signature and key, the parts the last system showed, and how many
    bars have been read."""

    def __init__(self):
        self.parts: dict[int, dict[int, list[tokens.Bar]]] = {}
        self.named: list | None = None
        self.times: dict[tuple[int, int], object] = {}
        self.keys: dict[tuple[int, int], str] = {}
        self.shown: set[int] = set()
        self.bars = 0
        self.where: list[tuple] = []
        self.seqs: list[list[str]] = []
        self.unplaced = 1000  # an unlabelled staff no part fits is a part of its own


def _rests(n: int) -> list[tokens.Bar]:
    return [tokens.Bar(notes=[{'rest': True, 'measure': True, 'space': False, 'pitch': None, 'type': 'whole',
                               'dots': 0, 'tuplet': None, 'tie': False, 'voice': 0}]) for _ in range(n)]


def read_page(image: pathlib.Path, page: int | None, w: Work, number: int = 0,
              names: list[list[str]] | None = None) -> None:
    """Read one page into the work [w]: each staff's bars onto its part, a
    part the page's systems leave out resting through them. [names], where
    given, names each system's staves top to bottom, in place of the labels
    the text reader would read."""
    gray = layout_suite.gray_page(image, page)
    if os.environ.get('STAFFREADER_ACHA_GEOMETRY') == '1':
        from .. import acha_layout
        lay = acha_layout.read(gray, str(image) if page is None else None)
    else:
        lay = layout.read(gray)
    level = layout.level(gray, lay.angle, lay.bend)
    if not lay.systems:
        raise RuntimeError('no staves found on the page')
    # With a text reader, each staff's part comes from its label, matched
    # against the parts the work's earlier pages named, so a system that
    # leaves out resting staves keeps every staff on its part.
    plan, named = _given_plan(lay, names, w.named) if names else _plan(level, lay, w.named, w.shown)
    if named:
        w.named = named
    for s_i, system in enumerate(lay.systems):
        groups = system.parts or [[st] for st in system.staves]
        ids = plan[s_i] if plan else list(range(len(groups)))
        fixed = []
        for g, k in enumerate(ids):
            if k is None:
                k, w.unplaced = w.unplaced, w.unplaced + 1
            fixed.append(k)
        places = [(fixed[g], n) for g, group in enumerate(groups) for n in range(len(group))]
        staves = [st for group in groups for st in group]
        crops = [staffcrop.crop(level, lay.staves[st].lines, lay.staves[st].x0, lay.staves[st].x1) for st in staves]
        limits = [w.times.get(place) for place in places]
        counts = [system.measures] * len(crops)
        task = 'notes' if TASKS else None
        got = agree_on_key(crops, read_staves(crops, limits=limits, counts=counts, task=task), limits,
                           [w.keys.get(place) for place in places], counts=counts, task=task)
        if TRIPLET_MARGIN >= 0:
            # A page that continues a work prints no time signature: its bars
            # are taken to fill the length most of them fill (tokens._meter).
            meter = tokens._meter([tokens.parse(seq) for seq in got])
            guess = None
            if meter:
                beats, unit, _ = tokens._time_value(meter)
                guess = Fraction(4 * beats, unit)
            # A staff whose bar length is not the one most of the system's bars
            # fill has a time signature that is wrong or not yet known: left alone.
            got = triplets(crops, got, [guess if lim is None or lim == guess else None for lim in limits], task)
        marked = read_staves(crops, limits=limits, counts=counts, task='marks') if TASKS else got
        read = []
        for st, seq, full in zip(staves, got, marked):
            bars = tokens.parse(seq)
            if TASKS:
                bars = tokens.merge_marks(bars, tokens.parse(full))
            if 'ly' in full and lyrics.text_reader() is not None:
                staff = lay.staves[st]
                below = lay.staves[st + 1].top if st + 1 < len(lay.staves) else level.shape[0]
                if os.environ.get('STAFFREADER_ACHA_LYRICS') == 'hybrid':
                    # AChA: place notes first, then attach syllables by printed x position.
                    from .. import acha_lyrics
                    acha_lyrics.attach_hybrid(bars, level, staff, lyrics.rows(level, staff, below),
                                                   barlines=system.barlines)
                else:
                    verses = lyrics.read_rows(level, staff, lyrics.rows(level, staff, below))
                    lyrics.attach(bars, verses)
            read.append(bars)
        if WORD_FINDER is not None:
            _words(level, lay, system, staves, read)
        for place, seq, bars in zip(places, got, read):
            for b in bars:
                if b.time:
                    beats, unit, _ = tokens._time_value(b.time)
                    w.times[place] = Fraction(4 * beats, unit)
            w.keys[place] = next((t for t in seq if t.startswith('key:')), w.keys.get(place))
            if place[0] not in w.parts:
                w.parts[place[0]] = {}
            if place[1] not in w.parts[place[0]]:  # a part or staff first seen here rests through what came before
                w.parts[place[0]][place[1]] = _rests(w.bars)
            w.parts[place[0]][place[1]].extend(bars)
            w.where.append((number, s_i, *place))
            w.seqs.append(seq)
        w.bars += system.measures
        # A part this system leaves out rests through it; so does a staff of
        # its read with fewer bars than the system holds.
        for k in w.parts:
            for n in w.parts[k]:
                short = w.bars - len(w.parts[k][n])
                if short > 0:
                    w.parts[k][n] += _rests(short)
        w.shown = {k for k, _ in places}


_FINDER = None
_PAGE_WORDS: dict[int, list] = {}


def _words(level, lay, system, staves, read) -> None:
    """The directions in words of one system, found on the page and read by
    the text reader, put on its staves in place of the staff reader's."""
    global _FINDER
    import torch
    from .. import positions, text, words
    if _FINDER is None:
        _FINDER = words.load(WORD_FINDER)
    key = id(level)
    if key not in _PAGE_WORDS:  # once a page: its boxes, read
        _PAGE_WORDS.clear()
        space = float(np.median([s.space for s in lay.staves]))
        boxes = words.find(_FINDER, level, space)
        net = lyrics.text_reader()
        texts = []
        if boxes and net is not None:
            pad = int(0.3 * space)
            crops = [level[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] for x0, y0, x1, y1 in boxes]
            x, wd = text.batch([text.view(c) for c in crops], next(net.parameters()).device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                texts = net.read(x, wd)
        _PAGE_WORDS[key] = [list(zip(boxes, texts)), positions.heads(level, lay)]
    found, heads = _PAGE_WORDS[key]
    top, bottom = lay.staves[staves[0]].top, lay.staves[staves[-1]].bottom
    space = lay.staves[staves[0]].space
    mine = [f for f in found if top - 6 * space <= (f[0][1] + f[0][3]) / 2 <= bottom + 6 * space]
    for st, bars in zip(staves, read):
        positions.place(bars, lay.staves[st], heads[st], system.barlines)
    words.place_words(mine, [lay.staves[st] for st in staves], read, system.barlines, WORDS_REPLACE_ALL)


def write_work(w: Work, out: pathlib.Path) -> pathlib.Path:
    """The work read so far as MusicXML."""
    order = sorted(w.parts)
    named = w.named
    for k in order:
        if named and k < len(named) and named[k].name == 'tenor':
            tenor_clef([w.parts[k][n] for n in sorted(w.parts[k])])
    out.write_bytes(tokens.to_musicxml([[w.parts[k][n] for n in sorted(w.parts[k])] for k in order],
                                       names=[named[k].key if named and k < len(named) else f'Part {i + 1}'
                                              for i, k in enumerate(order)],
                                       transpose=[named[k].transpose if named and k < len(named) else 0
                                                  for k in order]))
    return out


def read_work(pages: list[tuple[pathlib.Path, int | None]], work: pathlib.Path,
              names: list[list[list[str]] | None] | None = None) -> pathlib.Path:
    """The pages of one work, read in order into one score: the parts its
    first page names, matched on every later page by their labels, each
    part's time signature and key carried over, and a part a page leaves
    out resting there. [names], where given, names each page's staves
    (see read_page)."""
    work.mkdir(parents=True, exist_ok=True)
    w = Work()
    for number, (image, page) in enumerate(pages):
        read_page(image, page, w, number, names[number] if names else None)
    (work / 'tokens.txt').write_text('\n'.join(f'{x}: {" ".join(s)}' for x, s in zip(w.where, w.seqs)))
    return write_work(w, work / 'reading.musicxml')


def read(image: pathlib.Path, page: int | None, work: pathlib.Path,
         names: list[list[str]] | None = None) -> pathlib.Path:
    """One page on its own."""
    return read_work([(image, page)], work, [names] if names else None)


def tenor_clef(staves: list[list[tokens.Bar]]) -> None:
    """A tenor part printed in a plain treble clef, as older editions print
    one, sounds an octave below what it shows. It is written as the
    transcriptions write it: in the treble clef with an 8 below, an octave
    lower. A staff whose treble clef already shows its 8 is left as read.
    It applies where the text reader names the part, which on a page that
    continues a song needs the names carried from where they are printed."""
    for bars in staves:
        lower = False
        for bar in bars:
            if bar.clef:
                lower = bar.clef == 'G2'
                if lower:
                    bar.clef = 'G2-8'
            if lower:
                for n in bar.notes:
                    if n.get('pitch'):
                        n['pitch'] = n['pitch'][0] + str(int(n['pitch'][1:]) - 1)


def _plan(level, lay, known: list | None = None,
          shown: set[int] | None = None) -> tuple[list[list[int | None]] | None, list | None]:
    """Each system's part for each of its staff groups, from the labels
    the text reader reads at the system's left; or (None, None) without
    a text reader, or where no label is read at all. The work's parts are
    those [known] from its pages before, if any ([shown]: those the last
    system showed), else those this page's first system names. A label is read line
    by line ('Cor.' over '(C)'), and one printed between two staff groups
    names both, numbered in order ('Vl.' for the first and second
    violins). A braced pair whose label names no keyboard or harp is two
    players' staves, so each becomes a part of its own."""
    from .. import instruments
    net = lyrics.text_reader()
    if net is None:
        return None, None
    import torch
    from .. import text
    found, crops, owners = [], [], []
    for s_i, system in enumerate(lay.systems):
        groups = system.parts or [[st] for st in system.staves]
        labels = _margin_labels(level, lay, system, groups)
        found.append(labels)
        for k, (_, lines) in enumerate(labels):
            for line in lines:
                crops.append(text.view(line))
                owners.append((s_i, k))
    read = collections.defaultdict(list)
    got: dict[int, str] = {}
    order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1])  # like widths together: little padding
    for k in range(0, len(order), 32):
        idx = order[k:k + 32]
        x, w = text.batch([crops[i] for i in idx], next(net.parameters()).device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            got.update(zip(idx, net.read(x, w)))
    for i, owner in enumerate(owners):  # each label's lines in order, top to bottom
        read[owner].append(got[i])
    systems = []
    for s_i, system in enumerate(lay.systems):
        groups = system.parts or [[st] for st in system.staves]
        names: list[str | None] = [None] * len(groups)
        own: dict[int, str] = {}  # what stands beside one staff group alone, such as a horn's key
        for k, (covered, _) in enumerate(found[s_i]):
            label = ' '.join(t for t in read[(s_i, k)] if t).strip() or None
            if label and len(covered) == 1 and instruments.instrument(label) is None:
                own[covered[0]] = label
                continue
            for n, g in enumerate(covered):
                names[g] = f'{label} {n + 1}' if label and len(covered) > 1 else label
        for g, extra in own.items():  # 'Hr.' between the horns, 'C' and 'F' beside each
            names[g] = f'{names[g]} {extra}' if names[g] else extra
        # A braced pair that is no keyboard or harp: two players, two parts.
        split_groups, split_names = [], []
        for g, name in zip(groups, names):
            inst = instruments.instrument(name)
            if len(g) > 1 and inst is not None and inst.name not in ('piano', 'organ', 'harp'):
                split_groups += [[st] for st in g]
                split_names += [f'{name} {n + 1}' for n in range(len(g))]
            else:
                split_groups.append(g)
                split_names.append(name)
        if len(split_groups) != len(groups):
            system.parts = split_groups
        systems.append(split_names)
    if not any(instruments.instrument(l) for names in systems for l in names):
        return None, None
    parts, plan = instruments.assign(systems, known, shown)
    return plan, parts


def _given_plan(lay, names: list[list[str]], known: list | None = None) -> tuple[list[list[int]], list]:
    """Each system's part for each of its staff groups, from names someone
    gave for each system's staves, top to bottom (a reviewer reading the
    printed labels where the text reader cannot). Neighbouring staves with
    the same name are one part, as a piano's are. The parts are the work's
    [known] parts, if any, then each new name before the next part
    printed below it, so a part a first system leaves out still takes its
    place in score order (after the known parts, whose numbers the work's
    bars already use)."""
    from .. import instruments
    if len(names) != len(lay.systems):
        raise ValueError(f'names given for {len(names)} systems; the page has {len(lay.systems)} '
                         f'({", ".join(str(len(s.staves)) for s in lay.systems)} staves)')
    order = [p.key for p in known] if known else []
    for s_i, (system, given) in enumerate(zip(lay.systems, names)):
        if len(given) != len(system.staves):
            raise ValueError(f'system {s_i + 1}: {len(given)} names given for {len(system.staves)} staves')
        seen = list(dict.fromkeys(given))
        for k, name in enumerate(seen):
            if name in order:
                continue
            # A new part goes before the next part printed below it, or last;
            # parts already read keep their numbers, so it goes after them.
            below = next((order.index(n) for n in seen[k + 1:] if n in order), len(order))
            order.insert(max(below, len(known or [])), name)
    plan = []
    for system, given in zip(lay.systems, names):
        groups, ids = [], []
        for st, name in zip(system.staves, given):
            if groups and ids[-1] == order.index(name):
                groups[-1].append(st)
            else:
                groups.append([st])
                ids.append(order.index(name))
        system.parts = groups
        plan.append(ids)
    by_key = {p.key: p for p in known or []}
    parts = []
    for name in order:
        read = instruments.instrument(name)
        parts.append(by_key.get(name) or instruments.Instrument(name, None, read.transpose if read else 0))
    return plan, parts


def _margin_labels(level, lay, system, groups) -> list[tuple[list[int], list]]:
    """The labels printed left of a system: for each, the staff groups it
    names (one, or the two it stands between) and a crop of each of its
    lines, cut close as the text reader's lines are. Some editions set a
    label hard against the bracket; brackets and braces are told from
    letters by height and taken out."""
    import cv2
    import numpy as np
    first, last = lay.staves[system.staves[0]], lay.staves[system.staves[-1]]
    space = first.space
    left = min(lay.staves[st].x0 for st in system.staves)
    top, bottom = max(0, int(first.top - space)), int(last.bottom + space)
    # A first system names its staves in full ("Clarinetti in B."), reaching some 20 spaces left.
    x0, x1 = max(0, int(left - 30 * space)), int(left - 0.3 * space)
    if x1 - x0 < 2 * space:
        return []
    strip = level[top:bottom, x0:x1].copy()
    n, lab, stats, _ = cv2.connectedComponentsWithStats((strip < 128).astype(np.uint8), connectivity=8)
    # Capitals and tall letters stand up to about 2.9 spaces in the
    # editions measured, so height alone is no test. A bracket or brace is
    # taller than any letter, or a tall thin stroke, or stands against
    # the staves at the strip's right edge.
    for k in range(1, n):
        h, w = stats[k, cv2.CC_STAT_HEIGHT], stats[k, cv2.CC_STAT_WIDTH]
        right = strip.shape[1] - (stats[k, cv2.CC_STAT_LEFT] + w)
        if h > 5 * space or (h > 3 * space and (w < 0.6 * space or right < 1.5 * space)):
            strip[lab == k] = 255
    rows = np.flatnonzero((strip < 128).any(axis=1))
    if not len(rows):
        return []
    # Lines of text: runs of inked rows, a gap of a quarter space joining
    # what a line's dots and accents leave apart.
    lines, start = [], rows[0]
    for a, b in zip(rows, rows[1:]):
        if b - a > 0.25 * space:
            lines.append((start, a))
            start = b
    lines.append((start, rows[-1]))
    lines = [(a, b) for a, b in lines if b - a >= 0.4 * space]
    # Two lines whose parentheses or descenders touch read as one tall line:
    # split it at its thinnest row in the middle.
    ink = (strip < 128).sum(axis=1)
    tall = 1.6 * float(np.median([b - a for a, b in lines])) if lines else 0
    split = []
    for a, b in lines:
        if b - a > tall:
            mid = a + (b - a) // 4 + int(np.argmin(ink[a + (b - a) // 4:b - (b - a) // 4 + 1]))
            split += [(a, mid), (mid + 1, b)]
        else:
            split.append((a, b))
    lines = [(a, b) for a, b in split if b - a >= 0.4 * space]
    # Lines close together are one label.
    labels, current = [], []
    for line in lines:
        if current and line[0] - current[-1][1] > 0.8 * space:
            labels.append(current)
            current = []
        current.append(line)
    if current:
        labels.append(current)
    spans = [(lay.staves[g[0]].top, lay.staves[g[-1]].bottom) for g in groups]
    out = []
    for label in labels:
        y = top + (label[0][0] + label[-1][1]) / 2
        inside = [k for k, (t, b) in enumerate(spans) if t - 0.5 * space <= y <= b + 0.5 * space]
        between = [k for k in range(len(spans) - 1) if spans[k][1] < y < spans[k + 1][0]]
        nearest = min(range(len(spans)), key=lambda k: abs((spans[k][0] + spans[k][1]) / 2 - y))
        covered = inside[:1] or ([between[0], between[0] + 1] if between else [nearest])
        pad = int(round(0.4 * space))
        crops = []
        for a, b in label:
            cols = np.flatnonzero((strip[a:b + 1] < 128).any(axis=0))
            crops.append(strip[max(0, a - pad):b + pad + 1, max(0, cols[0] - pad):cols[-1] + pad + 1])
        out.append((covered, crops))
    return out
