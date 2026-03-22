"""
Ghost Sync — Desktop Client
==============================
Main entry point for the Ghost Sync desktop application.

Usage:
    # Demo mode — test the overlay with sample text
    python -m client.main --demo

    # Play an SRT file over any video (manual mode)
    python -m client.main --srt movie.srt

    # Full auto mode — detect player, identify movie, fetch + sync subtitles
    python -m client.main
"""

import argparse
import os
import sys
import time

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

from client.overlay import SubtitleOverlay, SubtitlePlayer
from client.player_detect import get_running_players, get_player_video_file
from client.api_client import GhostSyncClient


def run_demo(app, overlay):
    """Demo mode: show the overlay with cycling sample text."""
    print("[Ghost Sync] Running in demo mode...")
    print("             The overlay should appear at the bottom of your screen.")
    print("             Press Ctrl+C in this terminal to quit.")

    demo_texts = [
        "👻 Ghost Sync — Subtitle Overlay Demo",
        "This floating text appears over any application.",
        "It's transparent and click-through!",
        "Right-click and drag to reposition the overlay.",
        "",
        "Waiting for real subtitles...",
    ]

    current = [0]

    def next_sub():
        overlay.set_subtitle(demo_texts[current[0]])
        current[0] = (current[0] + 1) % len(demo_texts)

    timer = QTimer()
    timer.timeout.connect(next_sub)
    timer.start(2500)
    next_sub()

    return app.exec_()


def run_srt_mode(app, overlay, srt_path, offset_ms=0):
    """Play an SRT file on the overlay (manual mode)."""
    print(f"[Ghost Sync] Playing SRT: {os.path.basename(srt_path)}")
    if offset_ms != 0:
        print(f"             Offset: {offset_ms:+}ms")

    player = SubtitlePlayer(overlay)
    player.load_srt(srt_path, offset_ms=offset_ms)
    player.start()

    return app.exec_()


def run_auto_mode(app, overlay):
    """
    Full auto mode:
    1. Detect running video player
    2. Get the video file path
    3. Send filename to server for identification
    4. Get subtitles
    5. Sync and display
    """
    print("[Ghost Sync] Auto mode — looking for video players...")

    client = GhostSyncClient()

    # Check server connection
    if not client.health_check():
        print("[ERROR] Cannot connect to Ghost Sync server!")
        print("        Start the server first: uvicorn server.main:app --reload")
        return 1

    # Detect video players
    players = get_running_players()
    if not players:
        print("[INFO] No video players detected.")
        print("       Open a movie in VLC, mpv, or another player, then try again.")
        return 1

    # Use first detected player
    player_info = players[0]
    print(f"[INFO] Found: {player_info['name']} (PID: {player_info['pid']})")

    # Try to get the video file path
    video_file = get_player_video_file(player_info["pid"])
    if video_file:
        print(f"[INFO] Playing: {os.path.basename(video_file)}")

        # Identify the movie
        filename = os.path.basename(video_file)
        movie_info = client.identify(filename)
        if movie_info:
            print(f"[INFO] Identified: {movie_info['title']} ({movie_info.get('year', '?')})")
        else:
            print(f"[INFO] Could not identify movie from filename.")
    else:
        print("[INFO] Could not determine which file is playing.")
        print("       You can use manual mode: python -m client.main --srt your_subtitle.srt")
        return 1

    print("[INFO] Auto mode pipeline is ready. Subtitle fetching coming soon!")
    return app.exec_()


def main():
    parser = argparse.ArgumentParser(
        description="Ghost Sync — Desktop Subtitle Overlay",
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
        help="Subtitle offset in milliseconds (for manual SRT mode)"
    )

    args = parser.parse_args()

    # Create Qt application and overlay
    app = QApplication(sys.argv)
    overlay = SubtitleOverlay()
    overlay.show()

    if args.demo:
        sys.exit(run_demo(app, overlay))
    elif args.srt:
        if not os.path.exists(args.srt):
            print(f"[ERROR] SRT file not found: {args.srt}")
            sys.exit(1)
        sys.exit(run_srt_mode(app, overlay, args.srt, args.offset))
    else:
        sys.exit(run_auto_mode(app, overlay))


if __name__ == "__main__":
    main()
