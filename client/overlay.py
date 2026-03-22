"""
Ghost Sync — Floating Subtitle Overlay
========================================
A transparent, click-through, always-on-top window that displays
synced subtitles over any video player.
"""

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QFont, QColor, QPainter, QPainterPath, QFontMetrics


class SubtitleOverlay(QWidget):
    """
    Floating subtitle display window.
    - Transparent background
    - Always on top of other windows
    - Click-through (mouse events pass to apps behind)
    - Draggable with right-click
    """

    def __init__(self):
        super().__init__()

        # Window properties
        self.setWindowTitle("Ghost Sync")
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool  # hides from taskbar
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        # Position: bottom center of screen
        screen = QApplication.primaryScreen().geometry()
        self.overlay_width = int(screen.width() * 0.7)
        self.overlay_height = 120
        x = (screen.width() - self.overlay_width) // 2
        y = screen.height() - self.overlay_height - 80  # 80px from bottom
        self.setGeometry(x, y, self.overlay_width, self.overlay_height)

        # Subtitle state
        self.current_text = ""
        self.font_size = 28
        self.font_family = "Segoe UI"
        self.text_color = QColor(255, 255, 255)       # white
        self.outline_color = QColor(0, 0, 0)           # black outline
        self.outline_width = 3
        self.bg_opacity = 100  # 0-255, semi-transparent background

        # Dragging state
        self._drag_pos = None

    def set_subtitle(self, text):
        """Update the displayed subtitle text."""
        self.current_text = text
        self.update()  # trigger repaint

    def clear_subtitle(self):
        """Clear the subtitle display."""
        self.current_text = ""
        self.update()

    def paintEvent(self, event):
        """Custom paint: subtitle text with outline for readability."""
        if not self.current_text:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Semi-transparent background bar
        bg_color = QColor(0, 0, 0, self.bg_opacity)
        painter.fillRect(self.rect(), bg_color)

        # Font setup
        font = QFont(self.font_family, self.font_size, QFont.Bold)
        painter.setFont(font)

        # Calculate text position (centered)
        metrics = QFontMetrics(font)
        text_rect = metrics.boundingRect(
            self.rect(), Qt.AlignCenter | Qt.TextWordWrap, self.current_text
        )

        # Draw text outline (draw text in black, offset in 8 directions)
        painter.setPen(self.outline_color)
        for dx in range(-self.outline_width, self.outline_width + 1):
            for dy in range(-self.outline_width, self.outline_width + 1):
                if dx == 0 and dy == 0:
                    continue
                offset_rect = self.rect().translated(dx, dy)
                painter.drawText(
                    offset_rect, Qt.AlignCenter | Qt.TextWordWrap,
                    self.current_text
                )

        # Draw main text in white
        painter.setPen(self.text_color)
        painter.drawText(
            self.rect(), Qt.AlignCenter | Qt.TextWordWrap,
            self.current_text
        )

        painter.end()

    # =========================================================================
    # Dragging support (right-click to drag, left-click passes through)
    # =========================================================================

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self._drag_pos = None
            self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            event.accept()


class SubtitlePlayer:
    """
    Plays back an SRT file in sync, displaying subtitles at the right time.
    Reads a list of (start_ms, end_ms, text) tuples and shows them on the overlay.
    """

    def __init__(self, overlay):
        self.overlay = overlay
        self.subtitles = []  # list of (start_ms, end_ms, text)
        self.offset_ms = 0   # sync offset from the sync engine
        self.current_index = 0
        self.elapsed_ms = 0

        # Timer fires every 50ms to check if we need to show/hide subtitles
        self.timer = QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._tick)

    def load_srt(self, srt_path, offset_ms=0):
        """Load subtitles from an SRT file."""
        import pysrt
        subs = pysrt.open(srt_path)
        self.subtitles = [
            (sub.start.ordinal, sub.end.ordinal, sub.text.replace("\n", " "))
            for sub in subs
        ]
        self.offset_ms = offset_ms
        self.current_index = 0
        self.elapsed_ms = 0

    def start(self, start_time_ms=0):
        """Start subtitle playback from a given timestamp."""
        self.elapsed_ms = start_time_ms
        self.timer.start()

    def stop(self):
        """Stop subtitle playback."""
        self.timer.stop()
        self.overlay.clear_subtitle()

    def _tick(self):
        """Check if current time matches any subtitle."""
        self.elapsed_ms += self.timer.interval()
        current_time = self.elapsed_ms + self.offset_ms

        # Find the subtitle that should be displayed now
        displayed = False
        for start_ms, end_ms, text in self.subtitles:
            if start_ms <= current_time <= end_ms:
                self.overlay.set_subtitle(text)
                displayed = True
                break

        if not displayed:
            self.overlay.clear_subtitle()


# =============================================================================
# Quick test — run standalone to see the overlay
# =============================================================================

def main():
    """Quick demo: shows the overlay with sample text."""
    app = QApplication(sys.argv)

    overlay = SubtitleOverlay()
    overlay.show()

    # Demo: cycle through sample subtitles
    demo_texts = [
        "Ghost Sync — Floating Subtitle Overlay",
        "This text appears over any video player.",
        "Click-through: your mouse passes right through.",
        "Right-click and drag to reposition.",
        "",  # clear
        "Ready for real subtitles!",
    ]

    current = [0]

    def next_subtitle():
        overlay.set_subtitle(demo_texts[current[0]])
        current[0] = (current[0] + 1) % len(demo_texts)

    timer = QTimer()
    timer.timeout.connect(next_subtitle)
    timer.start(2000)
    next_subtitle()  # show first one immediately

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
