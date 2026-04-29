from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch PNG simulation frames into an animated GIF.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("sim"),
        help="Directory containing frame_XXXX.png files (default: sim).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="frame_*.png",
        help="Glob pattern for frame files (default: frame_*.png).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sim/simulation.gif"),
        help="Output GIF path (default: sim/simulation.gif).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=8.0,
        help="Frames per second for the GIF (default: 8).",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="GIF loop count; 0 means infinite looping (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    frame_paths = sorted(args.frames_dir.glob(args.pattern))
    if not frame_paths:
        raise FileNotFoundError(
            f"No frames found in '{args.frames_dir}' matching '{args.pattern}'"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame_duration_ms = int(round(1000.0 / args.fps))

    frames = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
    try:
        frames[0].save(
            args.output,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=args.loop,
            optimize=False,
        )
    finally:
        for frame in frames:
            frame.close()

    print(
        f"Wrote GIF: {args.output} "
        f"({len(frame_paths)} frames, {args.fps:.2f} fps, {frame_duration_ms} ms/frame)"
    )


if __name__ == "__main__":
    main()
