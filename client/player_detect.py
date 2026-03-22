"""
Ghost Sync — Player Detection
================================
Detects running video players on Windows using psutil.
Auto-triggers subtitle fetch when a player starts.
"""

import psutil


# Known video player process names (lowercase)
KNOWN_PLAYERS = {
    "vlc.exe": "VLC Media Player",
    "mpv.exe": "mpv",
    "mpc-hc.exe": "Media Player Classic",
    "mpc-hc64.exe": "Media Player Classic (64-bit)",
    "mpc-be.exe": "MPC-BE",
    "mpc-be64.exe": "MPC-BE (64-bit)",
    "wmplayer.exe": "Windows Media Player",
    "potplayer.exe": "PotPlayer",
    "potplayer64.exe": "PotPlayer (64-bit)",
    "potplayermini.exe": "PotPlayer Mini",
    "potplayermini64.exe": "PotPlayer Mini (64-bit)",
    "kmplayer.exe": "KMPlayer",
    "smplayer.exe": "SMPlayer",
    "Video.UI.exe": "Windows Films & TV",
    "msedge.exe": "Microsoft Edge",
    "chrome.exe": "Google Chrome",
}


def get_running_players():
    """
    Find all running video player processes.

    Returns list of dicts:
        [{"name": "VLC Media Player", "pid": 1234, "process_name": "vlc.exe"}, ...]
    """
    players = []
    seen_pids = set()

    for proc in psutil.process_iter(["pid", "name"]):
        try:
            proc_name = proc.info["name"].lower()
            if proc_name in KNOWN_PLAYERS and proc.info["pid"] not in seen_pids:
                players.append({
                    "name": KNOWN_PLAYERS[proc_name],
                    "pid": proc.info["pid"],
                    "process_name": proc_name,
                })
                seen_pids.add(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return players


def get_player_video_file(pid):
    """
    Try to find what video file a player has open.
    Works by checking command-line arguments (most players pass the file path).

    Returns the file path or None.
    """
    try:
        proc = psutil.Process(pid)
        cmdline = proc.cmdline()

        # Look for file paths in command line args
        video_extensions = {
            ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv",
            ".webm", ".m4v", ".mpg", ".mpeg", ".ts", ".vob",
        }

        for arg in cmdline[1:]:  # skip the executable itself
            lower = arg.lower()
            for ext in video_extensions:
                if lower.endswith(ext):
                    return arg

    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    return None


def detect_and_report():
    """Print all detected video players and their files."""
    players = get_running_players()

    if not players:
        print("No video players detected.")
        return []

    print(f"Found {len(players)} video player(s):")
    for p in players:
        video_file = get_player_video_file(p["pid"])
        file_info = f" → {video_file}" if video_file else ""
        print(f"  • {p['name']} (PID: {p['pid']}){file_info}")

    return players


if __name__ == "__main__":
    detect_and_report()
