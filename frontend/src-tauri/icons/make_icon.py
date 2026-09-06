"""Generate the Mirume app icon — a stylised watching eye (見る目).

Draws a single 1024x1024 PNG (`icon-source.png`); `npm run tauri icon` then
derives the full `.icns` / `.png` set from it. Run with the backend venv
(Pillow):  python make_icon.py
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

SS = 4  # supersample factor
SIZE = 1024
S = SIZE * SS

# Palette — deep indigo ground, warm cream eye, amber iris.
BG_TOP = (79, 70, 172)
BG_BOTTOM = (49, 41, 120)
SCLERA = (247, 243, 233)
IRIS_OUTER = (232, 156, 42)
IRIS_INNER = (196, 110, 20)
PUPIL = (32, 26, 60)
HIGHLIGHT = (255, 252, 245)


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def _rounded_rect_mask(size: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return mask


def main() -> None:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Vertical gradient background.
    grad = Image.new("RGB", (1, S))
    for y in range(S):
        grad.putpixel((0, y), _lerp(BG_TOP, BG_BOTTOM, y / S))
    grad = grad.resize((S, S))

    pad = int(S * 0.085)
    radius = int(S * 0.225)
    mask = _rounded_rect_mask(S - 2 * pad, radius)
    img.paste(grad.crop((pad, pad, S - pad, S - pad)), (pad, pad), mask)

    cx, cy = S / 2, S / 2
    eye_w = S * 0.60
    eye_h = S * 0.40

    # Almond eye outline as two circular arcs meeting at the corners.
    corner = eye_w / 2
    r = (corner ** 2 + (eye_h / 2) ** 2) / (eye_h)  # radius of each arc
    # Upper arc centre is below the eye, lower arc centre above.
    up_c = (cx, cy + (r - eye_h / 2))
    lo_c = (cx, cy - (r - eye_h / 2))
    span = math.degrees(math.asin(corner / r))

    lens = Image.new("L", (S, S), 0)
    ld = ImageDraw.Draw(lens)
    ld.pieslice(
        [up_c[0] - r, up_c[1] - r, up_c[0] + r, up_c[1] + r],
        270 - span, 270 + span, fill=255,
    )
    ld.pieslice(
        [lo_c[0] - r, lo_c[1] - r, lo_c[0] + r, lo_c[1] + r],
        90 - span, 90 + span, fill=255,
    )
    sclera = Image.new("RGBA", (S, S), SCLERA + (255,))
    img.paste(sclera, (0, 0), lens)

    # Iris — sized to sit fully within the lid opening (no flat clip).
    iris_r = eye_h * 0.46
    iris = Image.new("RGB", (1, int(iris_r * 2)))
    for i in range(int(iris_r * 2)):
        iris.putpixel((0, i), _lerp(IRIS_OUTER, IRIS_INNER, abs(i - iris_r) / iris_r))
    iris = iris.resize((int(iris_r * 2), int(iris_r * 2)))
    iris_mask = Image.new("L", (int(iris_r * 2), int(iris_r * 2)), 0)
    ImageDraw.Draw(iris_mask).ellipse([0, 0, iris_r * 2 - 1, iris_r * 2 - 1], fill=255)
    # Clip the iris to the eye opening so it never spills past the lids.
    iris_layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    iris_layer.paste(iris, (int(cx - iris_r), int(cy - iris_r)), iris_mask)
    iris_layer.putalpha(Image.composite(iris_layer.getchannel("A"), Image.new("L", (S, S), 0), lens))
    img = Image.alpha_composite(img, iris_layer)
    draw = ImageDraw.Draw(img)

    # Pupil + highlight.
    pupil_r = iris_r * 0.46
    draw.ellipse([cx - pupil_r, cy - pupil_r, cx + pupil_r, cy + pupil_r], fill=PUPIL)
    hi_r = pupil_r * 0.42
    draw.ellipse(
        [cx - iris_r * 0.55 - hi_r, cy - iris_r * 0.55 - hi_r,
         cx - iris_r * 0.55 + hi_r, cy - iris_r * 0.55 + hi_r],
        fill=HIGHLIGHT,
    )

    # Eyelid stroke for definition.
    stroke = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    sd = ImageDraw.Draw(stroke)
    lw = int(S * 0.018)
    sd.arc([up_c[0] - r, up_c[1] - r, up_c[0] + r, up_c[1] + r], 270 - span, 270 + span,
           fill=BG_BOTTOM + (255,), width=lw)
    sd.arc([lo_c[0] - r, lo_c[1] - r, lo_c[0] + r, lo_c[1] + r], 90 - span, 90 + span,
           fill=BG_BOTTOM + (255,), width=lw)
    stroke.putalpha(Image.composite(stroke.getchannel("A"),
                                    Image.new("L", (S, S), 0),
                                    _rounded_rect_full(S, pad, radius)))
    img = Image.alpha_composite(img, stroke)

    out = img.resize((SIZE, SIZE), Image.LANCZOS)
    dest = Path(__file__).with_name("icon-source.png")
    out.save(dest)
    print(f"wrote {dest}")


def _rounded_rect_full(size: int, pad: int, radius: int) -> Image.Image:
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle([pad, pad, size - pad - 1, size - pad - 1],
                                        radius=radius, fill=255)
    return m


if __name__ == "__main__":
    main()
