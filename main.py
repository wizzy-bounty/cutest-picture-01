#!/usr/bin/env python3
"""cutest-picture-01 entry: "Mochi & Piko".

A deterministic, standard-library-only Python 3 program that draws one
picture and writes it to /output/entry.png (MAKEFUN_OUTPUT_DIR overrides
the directory outside the grader's container).

The picture: Mochi, a chubby cream kitten with a tilted head, a coral bow
and very big sparkly eyes, cuddling Piko, a round baby chick, on a soft
pastel morning background with tiny hearts and sparkles.

Everything is drawn with signed distance fields (SDFs): every shape is a
distance function, edges are anti-aliased from the distance itself, and
the whole scene is painted layer by layer. No randomness, no clock, no
network, no files read: two runs write exactly the same bytes.
"""

import math
import os
import struct
import zlib
from array import array

W = 880
H = 880


# --------------------------------------------------------------------------
# small math helpers
# --------------------------------------------------------------------------

def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def mix(a, b, t):
    return a + (b - a) * t


def mix3(c1, c2, c3, t):
    """Three-stop gradient: c1 at t=0, c2 at t=0.5, c3 at t=1."""
    if t < 0.5:
        u = t * 2.0
        return (mix(c1[0], c2[0], u), mix(c1[1], c2[1], u), mix(c1[2], c2[2], u))
    u = (t - 0.5) * 2.0
    return (mix(c2[0], c3[0], u), mix(c2[1], c3[1], u), mix(c2[2], c3[2], u))


def smin(a, b, k):
    """Polynomial smooth minimum: melts two shapes into one blobby shape."""
    h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return mix(b, a, h) - k * h * (1.0 - h)


def rgb(hexcode):
    return (int(hexcode[1:3], 16) / 255.0,
            int(hexcode[3:5], 16) / 255.0,
            int(hexcode[5:7], 16) / 255.0)


# --------------------------------------------------------------------------
# signed distance fields (distance in pixels, negative inside)
# --------------------------------------------------------------------------

def sd_circle(x, y, cx, cy, r):
    dx = x - cx
    dy = y - cy
    return math.sqrt(dx * dx + dy * dy) - r


def sd_ellipse(x, y, cx, cy, rx, ry):
    # iq's gradient-corrected approximation
    px = abs(x - cx)
    py = abs(y - cy)
    if px < 0.001:
        px = 0.001
    if py < 0.001:
        py = 0.001
    t = math.atan2(py * rx, px * ry)
    ex = rx * math.cos(t)
    ey = ry * math.sin(t)
    vx = px - ex
    vy = py - ey
    d = math.sqrt(vx * vx + vy * vy)
    inside = (px / rx) ** 2 + (py / ry) ** 2 < 1.0
    return -d if inside else d


def sd_ellipse_rot(x, y, cx, cy, rx, ry, ang):
    ca = math.cos(ang)
    sa = math.sin(ang)
    dx = x - cx
    dy = y - cy
    lx = dx * ca + dy * sa
    ly = -dx * sa + dy * ca
    return sd_ellipse(lx, ly, 0.0, 0.0, rx, ry)


def sd_capsule(x, y, ax, ay, bx, by, r):
    pax = x - ax
    pay = y - ay
    bax = bx - ax
    bay = by - ay
    h = clamp((pax * bax + pay * bay) / (bax * bax + bay * bay), 0.0, 1.0)
    dx = pax - bax * h
    dy = pay - bay * h
    return math.sqrt(dx * dx + dy * dy) - r


def sd_triangle(x, y, ax, ay, bx, by, cx, cy):
    def edge(px, py, x1, y1, x2, y2):
        ex = x2 - x1
        ey = y2 - y1
        wx = px - x1
        wy = py - y1
        t = clamp((wx * ex + wy * ey) / (ex * ex + ey * ey), 0.0, 1.0)
        dx = wx - ex * t
        dy = wy - ey * t
        return dx * dx + dy * dy, (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)

    d1, s1 = edge(x, y, ax, ay, bx, by)
    d2, s2 = edge(x, y, bx, by, cx, cy)
    d3, s3 = edge(x, y, cx, cy, ax, ay)
    d = math.sqrt(min(d1, d2, d3))
    if (s1 > 0 and s2 > 0 and s3 > 0) or (s1 < 0 and s2 < 0 and s3 < 0):
        return -d
    return d


def sd_arc(x, y, cx, cy, r, a0, a1):
    """Distance to a circular arc; angles in radians, image coords (y down)."""
    dx = x - cx
    dy = y - cy
    ang = math.atan2(dy, dx)
    if a0 <= ang <= a1:
        return abs(math.sqrt(dx * dx + dy * dy) - r)
    d0 = math.hypot(x - (cx + r * math.cos(a0)), y - (cy + r * math.sin(a0)))
    d1 = math.hypot(x - (cx + r * math.cos(a1)), y - (cy + r * math.sin(a1)))
    return min(d0, d1)


def sd_heart(x, y, cx, cy, s):
    """Classic heart SDF (iq), local units scaled by s; picture y is down,
    so the local y is flipped to keep the heart upright."""
    px = (x - cx) / s
    py = -(y - cy) / s
    px = abs(px)
    if py + px > 1.0:
        d = math.hypot(px - 0.25, py - 0.75) - math.sqrt(2.0) / 4.0
        return d * s
    mx = px + py
    if mx < 0.0:
        mx = 0.0
    d1 = (px - 0.0) ** 2 + (py - 1.0) ** 2
    qx = px - 0.5 * mx
    qy = py - 0.5 * mx
    d2 = qx * qx + qy * qy
    sign = 1.0 if px - py >= 0 else -1.0
    return math.sqrt(min(d1, d2)) * sign * s


def edge_alpha(d, soft=1.6):
    """Anti-aliased coverage from a signed distance."""
    return clamp(0.5 - d / soft, 0.0, 1.0)


# --------------------------------------------------------------------------
# canvas: three float channels, layers painted back to front
# --------------------------------------------------------------------------

class Canvas:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        n = w * h
        self.r = array('f', [0.0]) * n
        self.g = array('f', [0.0]) * n
        self.b = array('f', [0.0]) * n

    def layer(self, x0, y0, x1, y1, fn):
        """fn(x, y) -> (r, g, b, a) or None; composited source-over."""
        x0 = max(0, int(x0))
        y0 = max(0, int(y0))
        x1 = min(self.w, int(x1) + 1)
        y1 = min(self.h, int(y1) + 1)
        R, G, B, w = self.r, self.g, self.b, self.w
        for yy in range(y0, y1):
            base = yy * w
            py = yy + 0.5
            for xx in range(x0, x1):
                res = fn(xx + 0.5, py)
                if res is None:
                    continue
                cr, cg, cb, a = res
                if a <= 0.0:
                    continue
                if a > 1.0:
                    a = 1.0
                i = base + xx
                ia = 1.0 - a
                R[i] = cr * a + R[i] * ia
                G[i] = cg * a + G[i] * ia
                B[i] = cb * a + B[i] * ia

    def fill(self, fn):
        self.layer(0, 0, self.w - 1, self.h - 1, fn)


# --------------------------------------------------------------------------
# the scene
# --------------------------------------------------------------------------

HC = (445.0, 418.0)              # kitten head centre
TILT = -0.10                     # head tilt, radians (her charm point)
_TC = math.cos(TILT)
_TS = math.sin(TILT)


def head_pt(x, y):
    """Map a canvas point into the kitten's tilted head space."""
    dx = x - HC[0]
    dy = y - HC[1]
    return (HC[0] + dx * _TC - dy * _TS,
            HC[1] + dx * _TS + dy * _TC)


def kitten_sdf(x, y):
    """Silhouette of the kitten (head + ears + body), one blobby union."""
    hx, hy = head_pt(x, y)
    d = sd_circle(hx, hy, HC[0], HC[1], 170.0)
    d_body = sd_ellipse(x, y, 450.0, 698.0, 190.0, 148.0)
    d = smin(d, d_body, 55.0)
    ear_l = sd_triangle(hx, hy, 296.0, 336.0, 350.0, 160.0, 434.0, 258.0)
    ear_r = sd_triangle(hx, hy, 456.0, 258.0, 540.0, 160.0, 594.0, 336.0)
    d = smin(d, ear_l, 20.0)
    d = smin(d, ear_r, 20.0)
    return d


def tail_sdf(x, y):
    d = sd_circle(x, y, 240.0, 748.0, 36.0)
    d = smin(d, sd_circle(x, y, 214.0, 692.0, 33.0), 16.0)
    d = smin(d, sd_circle(x, y, 226.0, 634.0, 30.0), 16.0)
    return d


def chick_sdf(x, y):
    d = sd_circle(x, y, 678.0, 662.0, 94.0)
    d = smin(d, sd_capsule(x, y, 660.0, 580.0, 650.0, 546.0, 7.0), 8.0)
    d = smin(d, sd_capsule(x, y, 684.0, 578.0, 692.0, 544.0, 7.0), 8.0)
    return d


FUR_TOP = rgb("#FFFBF1")
FUR_BOT = rgb("#FFE7CB")
FUR_LINE = rgb("#F3CFAC")
CHICK_TOP = rgb("#FFE873")
CHICK_BOT = rgb("#FFC93C")
CHICK_LINE = rgb("#EFAF3E")


def fur_color(x, y):
    t = clamp((y - 240.0) / 580.0, 0.0, 1.0)
    return (mix(FUR_TOP[0], FUR_BOT[0], t),
            mix(FUR_TOP[1], FUR_BOT[1], t),
            mix(FUR_TOP[2], FUR_BOT[2], t))


def shape_layer(cv, sdf, color, bbox, soft=1.6, alpha=1.0, dilate=0.0):
    def fn(x, y):
        a = edge_alpha(sdf(x, y) - dilate, soft) * alpha
        if a <= 0.0:
            return None
        c = color(x, y) if callable(color) else color
        return (c[0], c[1], c[2], a)
    cv.layer(bbox[0], bbox[1], bbox[2], bbox[3], fn)


def soft_disc(cv, cx, cy, r, color, peak, soft, bbox=None):
    if bbox is None:
        bbox = (cx - r - soft, cy - r - soft, cx + r + soft, cy + r + soft)

    def fn(x, y):
        d = sd_circle(x, y, cx, cy, r)
        a = clamp(0.5 - d / soft, 0.0, 1.0) * peak
        if a <= 0.0:
            return None
        return (color[0], color[1], color[2], a)
    cv.layer(bbox[0], bbox[1], bbox[2], bbox[3], fn)


def soft_ellipse(cv, cx, cy, rx, ry, color, peak, soft):
    def fn(x, y):
        d = sd_ellipse(x, y, cx, cy, rx, ry)
        a = clamp(0.5 - d / soft, 0.0, 1.0) * peak
        if a <= 0.0:
            return None
        return (color[0], color[1], color[2], a)
    cv.layer(cx - rx - soft, cy - ry - soft, cx + rx + soft, cy + ry + soft, fn)


def sparkle(cv, cx, cy, rx, ry, color, peak=0.9):
    def fn(x, y):
        q = (abs(x - cx) / rx) ** 0.55 + (abs(y - cy) / ry) ** 0.55
        a = clamp((1.0 - q) / 0.22, 0.0, 1.0) * peak
        if a <= 0.0:
            return None
        return (color[0], color[1], color[2], a)
    cv.layer(cx - rx * 1.6, cy - ry * 1.6, cx + rx * 1.6, cy + ry * 1.6, fn)


def head_layer(cv, fn, bbox):
    def wrapped(x, y):
        hx, hy = head_pt(x, y)
        return fn(hx, hy)
    cv.layer(bbox[0], bbox[1], bbox[2], bbox[3], wrapped)


def paint_eye(cv, ecx):
    def iris(hx, hy, c=ecx):
        d = sd_ellipse(hx, hy, c, 434.0, 27.0, 36.0)
        a = edge_alpha(d)
        if a <= 0.0:
            return None
        col = rgb("#7A4A2B")
        return (col[0], col[1], col[2], a)
    head_layer(cv, iris, (ecx - 36, 390, ecx + 36, 480))

    def pupil(hx, hy, c=ecx):
        d = sd_ellipse(hx, hy, c, 431.0, 20.0, 29.0)
        a = edge_alpha(d)
        if a <= 0.0:
            return None
        col = rgb("#38241C")
        return (col[0], col[1], col[2], a)
    head_layer(cv, pupil, (ecx - 30, 394, ecx + 30, 470))

    def glow(hx, hy, c=ecx):
        d = sd_ellipse(hx, hy, c, 452.0, 15.0, 10.0)
        a = clamp(0.5 - d / 7.0, 0.0, 1.0) * 0.5
        if a <= 0.0:
            return None
        col = rgb("#E8A33D")
        return (col[0], col[1], col[2], a)
    head_layer(cv, glow, (ecx - 26, 430, ecx + 26, 472))

    def spark1(hx, hy, c=ecx):
        d = sd_circle(hx, hy, c - 8.0, 420.0, 10.0)
        a = edge_alpha(d, 2.0) * 0.96
        if a <= 0.0:
            return None
        return (1.0, 1.0, 1.0, a)
    head_layer(cv, spark1, (ecx - 26, 402, ecx + 10, 438))

    def spark2(hx, hy, c=ecx):
        d = sd_circle(hx, hy, c + 8.0, 446.0, 5.0)
        a = edge_alpha(d, 2.0) * 0.85
        if a <= 0.0:
            return None
        return (1.0, 1.0, 1.0, a)
    head_layer(cv, spark2, (ecx - 2, 434, ecx + 20, 458))

    # one tiny eyelash flick at the outer top corner
    side = -1.0 if ecx < 445.0 else 1.0

    def lash(hx, hy, c=ecx, s=side):
        d = sd_capsule(hx, hy, c + s * 22.0, 408.0, c + s * 34.0, 396.0, 3.0)
        a = edge_alpha(d)
        if a <= 0.0:
            return None
        col = rgb("#5A3A2A")
        return (col[0], col[1], col[2], a)
    head_layer(cv, lash, (ecx - 44, 386, ecx + 44, 420))


def draw_scene():
    cv = Canvas(W, H)

    # --- sky: three-stop pastel gradient + a warm glow up top
    top = rgb("#FFD9E8")
    mid = rgb("#FFEAD9")
    bot = rgb("#FFF3DE")

    def bg(x, y):
        c = mix3(top, mid, bot, y / H)
        return (c[0], c[1], c[2], 1.0)
    cv.fill(bg)
    soft_disc(cv, 440.0, 300.0, 300.0, rgb("#FFFDF6"), 0.55, 260.0)
    # halo behind the kitten's head
    soft_disc(cv, HC[0], HC[1] - 10.0, 225.0, rgb("#FFFDF4"), 0.65, 90.0)

    # --- soft clouds low in the frame
    soft_ellipse(cv, 110.0, 800.0, 165.0, 48.0, rgb("#FFFFFF"), 0.50, 46.0)
    soft_ellipse(cv, 800.0, 816.0, 185.0, 52.0, rgb("#FFFFFF"), 0.45, 50.0)
    soft_ellipse(cv, 84.0, 168.0, 74.0, 26.0, rgb("#FFFFFF"), 0.35, 34.0)

    # --- ground shadows
    soft_ellipse(cv, 452.0, 812.0, 298.0, 34.0, rgb("#DE9AA0"), 0.30, 30.0)
    soft_ellipse(cv, 678.0, 776.0, 108.0, 18.0, rgb("#DE9AA0"), 0.25, 20.0)

    # --- kitten tail, curling in on the left
    tb = (168, 590, 292, 800)
    shape_layer(cv, tail_sdf, FUR_LINE, tb, dilate=3.5)
    shape_layer(cv, tail_sdf, fur_color, tb)

    # --- kitten silhouette: outline pass, then fur
    kb = (96, 130, 800, 856)
    shape_layer(cv, kitten_sdf, FUR_LINE, kb, dilate=3.5)
    shape_layer(cv, kitten_sdf, fur_color, kb)

    # inner ears (pink), in tilted head space
    def inner_ear(sign):
        def fn(x, y):
            hx, hy = head_pt(x, y)
            if sign < 0:
                d = sd_triangle(hx, hy, 322.0, 308.0, 352.0, 208.0, 404.0, 262.0)
            else:
                d = sd_triangle(hx, hy, 486.0, 262.0, 538.0, 208.0, 568.0, 308.0)
            a = edge_alpha(d)
            if a <= 0.0:
                return None
            c = rgb("#FFB9CB")
            return (c[0], c[1], c[2], a)
        return fn
    cv.layer(270, 130, 470, 360, inner_ear(-1))
    cv.layer(440, 130, 630, 360, inner_ear(1))

    # belly patch
    soft_ellipse(cv, 450.0, 726.0, 106.0, 86.0, rgb("#FFFEFA"), 0.75, 26.0)

    # --- chick feet (tucked under the body drawn next)
    feet = rgb("#FF9F43")
    shape_layer(cv, lambda x, y: sd_ellipse(x, y, 636.0, 770.0, 17.0, 8.0),
                feet, (612, 756, 660, 784))
    shape_layer(cv, lambda x, y: sd_ellipse(x, y, 706.0, 770.0, 17.0, 8.0),
                feet, (682, 756, 730, 784))

    # --- chick body: outline, sunny gradient, wing
    cb = (566, 524, 792, 770)
    shape_layer(cv, chick_sdf, CHICK_LINE, cb, dilate=3.5)

    def chick_color(x, y):
        t = clamp((y - 560.0) / 210.0, 0.0, 1.0)
        return (mix(CHICK_TOP[0], CHICK_BOT[0], t),
                mix(CHICK_TOP[1], CHICK_BOT[1], t),
                mix(CHICK_TOP[2], CHICK_BOT[2], t))
    shape_layer(cv, chick_sdf, chick_color, cb)
    shape_layer(cv, lambda x, y: sd_ellipse_rot(x, y, 726.0, 692.0, 35.0, 20.0, -0.5),
                rgb("#F6B93B"), (678, 650, 776, 736))
    shape_layer(cv, lambda x, y: sd_arc(x, y, 726.0, 692.0, 20.0, -0.6, 1.9) - 1.6,
                rgb("#E8A92F"), (700, 664, 756, 724))

    # chick face: one bright eye, a beak turned toward the kitten, blush
    shape_layer(cv, lambda x, y: sd_ellipse(x, y, 646.0, 648.0, 11.0, 16.0),
                rgb("#43302A"), (628, 626, 664, 672))
    soft_disc(cv, 642.0, 640.0, 4.2, rgb("#FFFFFF"), 0.95, 2.4)
    shape_layer(cv, lambda x, y: sd_triangle(x, y, 596.0, 678.0, 636.0, 660.0, 636.0, 698.0),
                rgb("#FF9F43"), (588, 652, 644, 706))
    shape_layer(cv, lambda x, y: sd_capsule(x, y, 606.0, 678.0, 634.0, 678.0, 1.6),
                rgb("#E07F2E"), (598, 670, 642, 686))
    soft_ellipse(cv, 708.0, 676.0, 17.0, 10.0, rgb("#FFAB91"), 0.60, 9.0)

    # --- kitten paws, with toe beans lines
    paw = rgb("#FFEDD8")
    for pcx in (352.0, 548.0):
        shape_layer(cv, lambda x, y, c=pcx: sd_ellipse(x, y, c, 798.0, 55.0, 30.0),
                    paw, (pcx - 62, 760, pcx + 62, 834))
        for off in (-17.0, 17.0):
            shape_layer(cv,
                        lambda x, y, c=pcx, o=off: sd_capsule(x, y, c + o, 784.0, c + o, 802.0, 2.4),
                        rgb("#E7BBA3"), (pcx + off - 8, 778, pcx + off + 8, 808))

    # --- kitten face, all in tilted head space

    # muzzle patch
    def muzzle(hx, hy):
        d = sd_ellipse(hx, hy, 445.0, 512.0, 82.0, 52.0)
        a = clamp(0.5 - d / 26.0, 0.0, 1.0) * 0.55
        if a <= 0.0:
            return None
        c = rgb("#FFFEFB")
        return (c[0], c[1], c[2], a)
    head_layer(cv, muzzle, (330, 420, 560, 590))

    # blush
    for bcx in (320.0, 570.0):
        def blush(hx, hy, c=bcx):
            d = sd_ellipse(hx, hy, c, 500.0, 33.0, 17.0)
            a = clamp(0.5 - d / 10.0, 0.0, 1.0) * 0.55
            if a <= 0.0:
                return None
            col = rgb("#FFAEBC")
            return (col[0], col[1], col[2], a)
        head_layer(cv, blush, (bcx - 60, 460, bcx + 60, 540))

    # whiskers: two a side, whisper-light
    whisk = rgb("#EAC9B8")
    for (ax, ay, bx, by) in ((332.0, 476.0, 250.0, 458.0),
                             (334.0, 500.0, 256.0, 502.0),
                             (558.0, 476.0, 640.0, 458.0),
                             (556.0, 500.0, 634.0, 502.0)):
        def wh(hx, hy, a=ax, b=ay, c=bx, d=by):
            dist = sd_capsule(hx, hy, a, b, c, d, 2.1)
            al = edge_alpha(dist, 1.4) * 0.75
            if al <= 0.0:
                return None
            return (whisk[0], whisk[1], whisk[2], al)
        head_layer(cv, wh, (min(ax, bx) - 14, min(ay, by) - 14,
                        max(ax, bx) + 14, max(ay, by) + 14))

    # eyes: warm iris, deep pupil, two highlights, a honey glow at the base
    for ecx in (378.0, 512.0):
        paint_eye(cv, ecx)

    # nose: a tiny rounded heart of a nose
    def nose(hx, hy):
        d = sd_heart(hx, hy + 6.0, 445.0, 492.0, 13.0)
        a = edge_alpha(d)
        if a <= 0.0:
            return None
        col = rgb("#FF8FA3")
        return (col[0], col[1], col[2], a)
    head_layer(cv, nose, (420, 472, 470, 512))

    def nose_shine(hx, hy):
        d = sd_ellipse(hx, hy, 440.0, 489.0, 4.5, 2.8)
        a = edge_alpha(d, 1.6) * 0.8
        if a <= 0.0:
            return None
        return (1.0, 1.0, 1.0, a)
    head_layer(cv, nose_shine, (430, 480, 450, 498))

    # mouth: a little "w" smile under the nose
    mouth_c = rgb("#C97B84")

    def mouth(hx, hy):
        d1 = sd_arc(hx, hy, 429.0, 503.0, 14.5, 0.25, 2.75)
        d2 = sd_arc(hx, hy, 461.0, 503.0, 14.5, 0.40, 2.90)
        d3 = sd_capsule(hx, hy, 445.0, 501.0, 445.0, 510.0, 0.5)
        d = min(d1, d2, d3) - 1.7
        a = edge_alpha(d, 1.5)
        if a <= 0.0:
            return None
        return (mouth_c[0], mouth_c[1], mouth_c[2], a)
    head_layer(cv, mouth, (404, 492, 486, 528))

    # --- the coral bow, riding her left ear
    def bow(x, y):
        hx, hy = head_pt(x, y)
        d1 = sd_ellipse_rot(hx, hy, 300.0, 208.0, 36.0, 21.0, 0.55)
        d2 = sd_ellipse_rot(hx, hy, 366.0, 200.0, 36.0, 21.0, -0.55)
        d = min(d1, d2)
        a = edge_alpha(d)
        if a <= 0.0:
            return None
        col = rgb("#FF8FAB")
        return (col[0], col[1], col[2], a)
    cv.layer(240, 140, 430, 260, bow)

    def bow_shade(x, y):
        hx, hy = head_pt(x, y)
        d = sd_ellipse_rot(hx, hy, 333.0, 208.0, 15.0, 8.0, 0.0)
        a = edge_alpha(d) * 0.35
        if a <= 0.0:
            return None
        col = rgb("#E56B8C")
        return (col[0], col[1], col[2], a)
    cv.layer(306, 188, 360, 228, bow_shade)

    def bow_knot(x, y):
        hx, hy = head_pt(x, y)
        d = sd_circle(hx, hy, 333.0, 204.0, 14.0)
        a = edge_alpha(d)
        if a <= 0.0:
            return None
        col = rgb("#FF7D9C")
        return (col[0], col[1], col[2], a)
    cv.layer(308, 178, 358, 230, bow_knot)

    # --- floating hearts and sparkles
    for (hx, hy, s, col, al) in ((206.0, 322.0, 17.0, "#FF9FB6", 0.95),
                                 (722.0, 232.0, 13.0, "#FFB9CD", 0.95),
                                 (240.0, 548.0, 11.0, "#FFC7D6", 0.9)):
        c = rgb(col)

        def heart(x, y, ccx=hx, ccy=hy, ss=s, col=c, al=al):
            d = sd_heart(x, y, ccx, ccy, ss)
            a = edge_alpha(d, 1.8) * al
            if a <= 0.0:
                return None
            return (col[0], col[1], col[2], a)
        cv.layer(hx - s * 1.6, hy - s * 1.6, hx + s * 1.6, hy + s * 1.6, heart)

    sparkle(cv, 626.0, 150.0, 10.0, 27.0, rgb("#FFD166"))
    sparkle(cv, 168.0, 140.0, 7.0, 19.0, rgb("#FFD166"), 0.8)
    sparkle(cv, 776.0, 452.0, 6.0, 16.0, rgb("#FFE29A"), 0.8)

    # --- two tiny ground flowers
    for (fx, fy, petal) in ((142.0, 812.0, "#FFB7CD"), (756.0, 826.0, "#C9B8FF")):
        pc = rgb(petal)
        for k in range(5):
            ang = k * 2.0 * math.pi / 5.0 - math.pi / 2.0
            px = fx + 10.0 * math.cos(ang)
            py = fy + 10.0 * math.sin(ang)
            soft_disc(cv, px, py, 7.5, pc, 0.95, 2.2)
        soft_disc(cv, fx, fy, 5.0, rgb("#FFD166"), 1.0, 2.0)

    return cv


# --------------------------------------------------------------------------
# PNG writing (filter 0 rows, zlib level 9: deterministic bytes)
# --------------------------------------------------------------------------

def write_png(cv, path):
    w, h = cv.w, cv.h
    raw = bytearray()
    for yy in range(h):
        raw.append(0)
        base = yy * w
        row = bytearray(w * 3)
        for xx in range(w):
            i = base + xx
            j = xx * 3
            row[j] = int(clamp(cv.r[i], 0.0, 1.0) * 255.0 + 0.5)
            row[j + 1] = int(clamp(cv.g[i], 0.0, 1.0) * 255.0 + 0.5)
            row[j + 2] = int(clamp(cv.b[i], 0.0, 1.0) * 255.0 + 0.5)
        raw += row

    def chunk(tag, data):
        out = struct.pack(">I", len(data)) + tag + data
        out += struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        return out

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", ihdr)
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as fh:
        fh.write(png)
    return len(png)


def main():
    outdir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(outdir, exist_ok=True)
    cv = draw_scene()
    size = write_png(cv, os.path.join(outdir, "entry.png"))
    print("wrote entry.png: %dx%d, %d bytes" % (W, H, size))


if __name__ == "__main__":
    main()
