"""Record the reel's product footage from the docs' recorded chat demo into assets/clips/.

Run: python capture.py
Needs `pip install playwright`, `playwright install chromium` and ffmpeg on PATH.
Screens are taken at 540x960 with device scale 3, so crops stay sharp on a 1080x1920 frame.
"""

import subprocess
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
DEMO = (HERE.parents[1] / "docs" / "assets" / "demo" / "chat-demo.html").as_uri()
CLIPS = HERE / "assets" / "clips"


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


def grab(pg, out: Path, name: str, seconds: float) -> Path:
    """Screenshot as fast as the page allows, then stitch the frames at their real times."""
    d = out / name
    d.mkdir()
    stamps, t0 = [], time.time()
    while time.time() - t0 < seconds:
        pg.evaluate("window.scrollTo(0, 0)")
        f = d / f"{len(stamps):04d}.png"
        pg.screenshot(path=str(f))
        stamps.append((f, time.time() - t0))
    lines = []
    for k, (f, t) in enumerate(stamps):
        lines += [
            f"file '{f.as_posix()}'",
            f"duration {(stamps[k + 1][1] - t) if k + 1 < len(stamps) else 0.5:.3f}",
        ]
    (d / "list.txt").write_text("\n".join([*lines, f"file '{stamps[-1][0].as_posix()}'"]))
    mp4 = out / f"{name}.mp4"
    ffmpeg(
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(d / "list.txt"),
        "-vf",
        "fps=30,format=yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "16",
        str(mp4),
    )
    return mp4


def main() -> None:
    CLIPS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
        out = Path(tmp)
        browser = p.chromium.launch()
        pg = browser.new_page(
            viewport={"width": 540, "height": 960}, device_scale_factor=3, color_scheme="dark"
        )
        pg.goto(DEMO)
        pg.wait_for_timeout(1000)
        pg.get_by_text("Plan a rainy day for a family").click()
        pg.wait_for_timeout(8000)
        pg.get_by_text("Graph", exact=True).first.scroll_into_view_if_needed()
        pg.get_by_text("Cost", exact=True).first.click()
        pg.wait_for_timeout(700)
        pg.screenshot(path=str(out / "cost.png"))
        pg.get_by_text("Timeline", exact=True).first.click()
        pg.wait_for_timeout(700)
        pg.screenshot(path=str(out / "timeline.png"))
        pg.close()
        pg = browser.new_page(
            viewport={"width": 540, "height": 960}, device_scale_factor=3, color_scheme="dark"
        )
        pg.goto(DEMO)
        pg.wait_for_timeout(1000)
        pg.get_by_text("Book two tickets for Saturday").click()
        approval = grab(pg, out, "chat_approval", 9)
        browser.close()
        # Crops in source pixels (1620x2880 frames), sized for the reel's cards.
        ffmpeg(
            "-i",
            str(approval),
            "-vf",
            "crop=1560:1085:20:900,scale=920:640:flags=lanczos",
            "-c:v",
            "libx264",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            str(CLIPS / "approval.mp4"),
        )
        ffmpeg(
            "-i",
            str(out / "cost.png"),
            "-vf",
            "crop=1240:340:40:2480,scale=920:-1:flags=lanczos",
            str(CLIPS / "cost.png"),
        )
        ffmpeg(
            "-i",
            str(out / "timeline.png"),
            "-vf",
            "crop=1620:520:0:2360,scale=920:-1:flags=lanczos",
            str(CLIPS / "timeline.png"),
        )
    print("wrote", CLIPS)


if __name__ == "__main__":
    main()
