"""
Kalaye — Desktop Client Entry Point
======================================
Launch the Kalaye desktop application.

Usage:
    python -m client.main           # Launch the full GUI
    python -m client.main --demo    # Demo mode (overlay only)
    python -m client.main --srt movie.srt  # Play an SRT file directly
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Kalaye — AI-Powered Subtitle Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--demo", action="store_true",
        help="Run in demo mode with sample subtitle text"
    )
    parser.add_argument(
        "--srt", type=str, default=None,
        help="Path to an SRT file to play on the overlay"
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Subtitle offset in milliseconds"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed multiplier (default: 1.0)"
    )

    args = parser.parse_args()

    if args.demo:
        # Demo mode — just show overlay with sample text
        from client.overlay import main as overlay_demo
        overlay_demo()

    elif args.srt:
        # SRT mode — play a file directly on the overlay
        import os
        from PyQt5.QtWidgets import QApplication
        from client.overlay import SubtitleOverlay, SubtitlePlayer

        if not os.path.exists(args.srt):
            print(f"Error: SRT file not found: {args.srt}")
            sys.exit(1)

        app = QApplication(sys.argv)
        overlay = SubtitleOverlay()
        overlay.show()

        player = SubtitlePlayer(overlay, speed=args.speed)
        player.load_srt(args.srt, offset_ms=args.offset)
        overlay.player = player
        player.start()

        print(f"Playing: {os.path.basename(args.srt)}")
        print(f"Press ] to speed up, [ to slow down")
        sys.exit(app.exec_())

    else:
        # Full GUI mode
        from client.app import launch
        launch()


if __name__ == "__main__":
    main()
