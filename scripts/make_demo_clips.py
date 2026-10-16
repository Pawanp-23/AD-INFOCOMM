"""Make degraded copies of one recording for the robustness part of the demo.

    python scripts/make_demo_clips.py my_clip.mp4 --out demo_clips/

Writes <name>_clean.mp4, _noisy.mp4 (white noise at roughly 0 dB SNR), _dark.mp4, _blur.mp4 and
_no_video.wav. Upload them one after another: the segment inspector shows reliability
and fusion weight moving away from the degraded signal. Uses the ffmpeg bundled with
imageio-ffmpeg (installed by `pip install -e "backend[inference]"`).
"""

import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg

VARIANTS = {
    "clean": [],
    # Noise mixed at about the speech level; amix normalizes so the result does not clip.
    "noisy": [
        "-filter_complex",
        "anoisesrc=color=white:amplitude=0.25[n];[0:a][n]amix=inputs=2:duration=first[a]",
        "-map",
        "0:v?",
        "-map",
        "[a]",
    ],
    "dark": ["-vf", "eq=brightness=-0.45:contrast=0.6"],
    "blur": ["-vf", "boxblur=12:2"],
}


def run(ffmpeg: str, src: Path, args: list[str], dst: Path) -> None:
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src), *args, str(dst)]
    subprocess.run(cmd, check=True)
    print(f"wrote {dst}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("source", type=Path, help="a short video with a face and speech (5-60 s)")
    p.add_argument("--out", type=Path, default=Path("demo_clips"))
    args = p.parse_args()
    if not args.source.is_file():
        raise SystemExit(f"Not a file: {args.source}")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.source.stem
    for name, extra in VARIANTS.items():
        codec = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"]
        run(ffmpeg, args.source, [*extra, *codec], args.out / f"{stem}_{name}.mp4")
    run(ffmpeg, args.source, ["-vn", "-ac", "1", "-ar", "16000"], args.out / f"{stem}_no_video.wav")


if __name__ == "__main__":
    main()
