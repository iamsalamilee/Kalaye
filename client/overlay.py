"""
Ghost Sync — Floating Subtitle Overlay
========================================
A transparent, click-through, always-on-top window that displays
synced subtitles over any video player.
"""

import sys
import time
import bisect
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QComboBox
from PyQt5.QtCore import Qt, QTimer, QPoint, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPainter, QPainterPath, QFontMetrics


class OverlayControls(QWidget):
    """
    A small ⚙️ gear button + language dropdown that lives ON the overlay.
    It's a child widget of the overlay, so it moves with it automatically.
    """
    translation_requested = pyqtSignal(str)
    
    def __init__(self, overlay):
        super().__init__(overlay)  # Parent is the overlay — moves with it!
        self.overlay = overlay
        self._expanded = False

        # Supported languages
        self.ALL_LANGUAGES = [
            "Original (no translation)", 
            "Yoruba", "Hausa", "Igbo", "English", "French", 
            "Spanish", "Arabic", "Japanese", "Chinese (Simplified)", 
            "Portuguese", "German", "Hindi"
        ]

        # --- Gear Button ---
        from PyQt5.QtWidgets import QPushButton
        self.gear_btn = QPushButton("⚙️", self)
        self.gear_btn.setFixedSize(36, 36)
        self.gear_btn.setCursor(Qt.PointingHandCursor)
        self.gear_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(15, 15, 20, 180);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 18px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: rgba(40, 40, 50, 220);
                border: 1px solid rgba(255, 255, 255, 0.4);
            }
        """)
        self.gear_btn.clicked.connect(self._toggle_panel)

        # --- Language Combo (hidden by default) ---
        self.combo = QComboBox(self)
        self.combo.setCursor(Qt.PointingHandCursor)
        self.combo.setStyleSheet("""
            QComboBox {
                background-color: rgba(15, 15, 20, 230);
                color: #F3F4F6;
                border: 1px solid rgba(255, 255, 255, 0.25);
                border-radius: 6px;
                padding: 6px 12px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
                font-weight: bold;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: rgba(15, 15, 20, 255);
                color: white;
                selection-background-color: #1D4D9A;
            }
        """)
        self.combo.addItems(self.ALL_LANGUAGES)
        self.combo.currentTextChanged.connect(self._on_combo_changed)
        self.combo.setFixedWidth(180)
        self.combo.hide()  # Hidden until gear is clicked

        # Layout: position at top-right of overlay
        self._reposition()
        self.show()

    def _reposition(self):
        """Position the controls at the top-right corner of the overlay."""
        parent_w = self.overlay.width()
        # Gear button at top-right
        self.gear_btn.move(parent_w - 46, 4)
        # Combo next to gear (to the left)
        self.combo.move(parent_w - 46 - 185, 6)

    def _toggle_panel(self):
        """Toggle showing/hiding the language dropdown."""
        self._expanded = not self._expanded
        if self._expanded:
            self.combo.show()
        else:
            self.combo.hide()

    def resizeEvent(self, event):
        """Reposition when overlay resizes."""
        self._reposition()
        super().resizeEvent(event)

    def _on_combo_changed(self, target):
        if not target or not self.overlay.player:
            return

        cache = self.overlay.player.subtitles_cache
        # Only treat as cached when the list is non-empty.
        # An empty list means translation is still in progress.
        if target in cache and len(cache[target]) > 0:
            # Already translated and cached! Instant switch.
            self.overlay.player.active_language = target
        else:
            # Brand new language (or still streaming) — request full translation.
            self.translation_requested.emit(target)


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
        # NOTE: We do NOT set WA_TransparentForMouseEvents globally
        # because child widgets (gear button, combo) need to be clickable.
        # Left-click pass-through is handled in mousePressEvent instead.

        # Position: bottom center of screen
        screen = QApplication.primaryScreen().geometry()
        self.overlay_width = int(screen.width() * 0.7)
        self.overlay_height = 120
        x = (screen.width() - self.overlay_width) // 2
        y = screen.height() - self.overlay_height - 80  # 80px from bottom
        self.setGeometry(x, y, self.overlay_width, self.overlay_height)

        # Subtitle state
        self.current_text = ""
        self.font_size = 14
        # Nirmala UI is built into Windows 10/11 and supports Hindi, Arabic, Bengali, etc.
        self.font_family = "Nirmala UI"
        self._font_fallbacks = ["Segoe UI", "Arial Unicode MS", "Microsoft YaHei", "sans-serif"]
        self.text_color = QColor(255, 255, 255)
        self.outline_color = QColor(12, 12, 12)
        self.outline_width = 2
        self.bg_opacity = 30

        # Speed display (shown briefly when speed changes)
        self._speed_display = ""
        self._speed_timer = QTimer()
        self._speed_timer.setSingleShot(True)
        self._speed_timer.timeout.connect(self._clear_speed_display)

        # Reference to the player (set after player is created)
        self.player = None

        # Dragging state
        self._drag_pos = None

    def set_subtitle(self, text):
        """Update the displayed subtitle text."""
        self.current_text = text
        self.update()

    def clear_subtitle(self):
        """Clear the subtitle display."""
        self.current_text = ""
        self.update()

    def show_speed(self, speed):
        """Briefly flash the current speed on screen."""
        self._speed_display = f"Speed: {speed:.1f}x"
        self._speed_timer.start(1500)  # hide after 1.5s
        self.update()

    def _clear_speed_display(self):
        self._speed_display = ""
        self.update()

    def paintEvent(self, event):
        """Custom paint: subtitle text with outline for readability."""
        if not self.current_text and not self._speed_display:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Semi-transparent background bar
        bg_color = QColor(0, 0, 0, self.bg_opacity)
        painter.fillRect(self.rect(), bg_color)

        # Font setup — use fallback chain for multilingual support
        font = QFont(self.font_family, self.font_size, QFont.Bold)
        try:
            font.setFamilies([self.font_family] + self._font_fallbacks)
        except AttributeError:
            pass  # setFamilies not available in older PyQt5 — primary font still set
        painter.setFont(font)

        # Draw subtitle text
        if self.current_text:
            painter.setPen(self.outline_color)
            for dx in range(-self.outline_width, self.outline_width + 1):
                for dy in range(-self.outline_width, self.outline_width + 1):
                    if dx == 0 and dy == 0:
                        continue
                    painter.drawText(
                        self.rect().translated(dx, dy),
                        Qt.AlignCenter | Qt.TextWordWrap,
                        self.current_text
                    )
            painter.setPen(self.text_color)
            painter.drawText(
                self.rect(), Qt.AlignCenter | Qt.TextWordWrap,
                self.current_text
            )

        # Draw speed indicator (top-right, smaller font)
        if self._speed_display:
            small_font = QFont(self.font_family, 11)
            painter.setFont(small_font)
            painter.setPen(QColor(100, 180, 255))  # soft blue
            painter.drawText(
                self.rect().adjusted(0, 6, -10, 0),
                Qt.AlignTop | Qt.AlignRight,
                self._speed_display
            )

        painter.end()

    # =========================================================================
    # Dragging support (right-click to drag, left-click passes through)
    # Speed hotkeys: ] = speed up 0.1x,  [ = slow down 0.1x
    # =========================================================================

    def keyPressEvent(self, event):
        """Hotkeys for live speed and sync control."""
        if self.player is None:
            return
        
        # Speed Controls: [ ]
        speed = self.player.speed
        if event.key() == Qt.Key_BracketRight:  # ] = faster
            speed = min(round(speed + 0.1, 1), 3.0)
            self.player.set_speed(speed)
            self.show_speed(speed)
        elif event.key() == Qt.Key_BracketLeft:  # [ = slower
            speed = max(round(speed - 0.1, 1), 0.5)
            self.player.set_speed(speed)
            self.show_speed(speed)
            
        # Nudge Sync (Off-set): Left/Right Arrows
        # Moves subtitles back or forward by 200ms
        elif event.key() == Qt.Key_Right:  # Shift forward 0.2s
            self.player.offset_ms += 200
            self.show_speed_msg(f"Offset: {self.player.offset_ms}ms")
        elif event.key() == Qt.Key_Left:   # Shift back 0.2s
            self.player.offset_ms -= 200
            self.show_speed_msg(f"Offset: {self.player.offset_ms}ms")

    def show_speed_msg(self, msg):
        """Show arbitrary message in the speed display area."""
        self._speed_display = msg
        self._speed_timer.start(1500)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self._drag_pos = None
            event.accept()


class SubtitlePlayer:
    """
    Plays back an SRT file in sync using a Master Reference Clock.
    This prevents 'timer drift' even over extremely long movies.
    """

    def __init__(self, overlay, speed=1.0):
        import time
        self.overlay = overlay
        self.subtitles_cache = {}    # dict of language_name -> list of (start_ms, end_ms, text)
        self.active_language = "Original (no translation)"
        self.previous_language = None  # last fully-loaded language for smart fallback
        self.offset_ms = 0   # manual adjustment/sync offset
        
        self.speed = speed
        self.is_playing = False
        
        # Internal timing state
        self.start_clock = 0   # time.perf_counter() when play/resume started
        self.base_time_ms = 0  # the movie timestamp where we last started/resumed
        
        # Timer fires every 30ms for smooth 33fps visual updates
        self.timer = QTimer()
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._tick)

    def load_srt(self, srt_path, offset_ms=0):
        """Load subtitles from an SRT file."""
        import pysrt
        try:
            subs = pysrt.open(srt_path, encoding='utf-8')
        except UnicodeDecodeError:
            subs = pysrt.open(srt_path, encoding='latin-1')
        except Exception as e:
            print(f"[Overlay] Error loading SRT: {e}")
            subs = []
            
        parsed = [
            (sub.start.ordinal, sub.end.ordinal, sub.text.replace("\n", " "))
            for sub in subs
        ]
        print(f"[Overlay] Loaded {len(parsed)} subtitle entries from SRT")
        if parsed:
            print(f"[Overlay] First entry: {parsed[0][2][:60]}...")
        
        self.subtitles_cache = {
            "Original (no translation)": parsed
        }
        self.active_language = "Original (no translation)"
        self.offset_ms = offset_ms
        self.base_time_ms = 0
        self.is_playing = False

    def start(self, start_time_ms=0):
        """Start or Resume subtitle playback."""
        import time
        self.base_time_ms = start_time_ms
        self.start_clock = time.perf_counter()
        self.is_playing = True
        self.timer.start()

    def pause(self):
        """Pause playback and save current position."""
        if self.is_playing:
            self.base_time_ms = self.get_current_movie_time()
            self.is_playing = False
            self.timer.stop()

    def stop(self):
        """Stop and reset."""
        self.timer.stop()
        self.is_playing = False
        self.base_time_ms = 0
        self.overlay.clear_subtitle()

    def sync_to_position(self, position_ms):
        """Hard-sync the clock to a detected movie position.

        Called by AudioSyncEngine when it finds the exact playback position
        via audio fingerprint cross-correlation. This corrects any drift.
        """
        import time
        drift = abs(position_ms - self.get_current_movie_time())

        self.base_time_ms = position_ms
        self.start_clock = time.perf_counter()

        # Auto-start if not already playing
        if not self.is_playing:
            self.is_playing = True
            self.timer.start()

        # Only log if the correction was noticeable (>200ms)
        if drift > 200:
            print(f"[Overlay] Clock corrected by {drift:.0f}ms → "
                  f"now at {position_ms / 1000:.1f}s")

    def set_speed(self, speed):
        """Live speed adjustment (Master clock handles the math)."""
        # Save progress at current speed before switching
        if self.is_playing:
            self.base_time_ms = self.get_current_movie_time()
            self.start_clock = time.perf_counter()  # 'time' imported at module level
        self.speed = speed

    def get_current_movie_time(self):
        """Calculate exact movie timestamp using Master Reference Clock."""
        if not self.is_playing:
            return self.base_time_ms
        
        import time
        # elapsed_real_seconds * speed = elapsed_movie_seconds
        real_diff = time.perf_counter() - self.start_clock
        movie_diff_ms = int(real_diff * 1000 * self.speed)
        return self.base_time_ms + movie_diff_ms

    def _find_text(self, current_time, sub_list):
        """Find the subtitle text for a given millisecond timestamp.

        Uses bisect on end_ms values to skip past expired entries in O(log n)
        instead of scanning from the beginning every 30 ms tick.
        """
        if not sub_list:
            return ""

        # Build an end_ms list for bisect (sub_list is always sorted by start_ms,
        # and since subtitles don't overlap, end_ms is also monotonically increasing).
        # We search for the first entry whose end_ms >= current_time.
        end_times = [entry[1] for entry in sub_list]
        idx = bisect.bisect_left(end_times, current_time)

        if idx < len(sub_list):
            start_ms, end_ms, text = sub_list[idx]
            if start_ms <= current_time <= end_ms:
                return text
        return ""

    def _tick(self):
        """Frame update: find and show the correct subtitle for this exact millisecond."""
        current_time = self.get_current_movie_time() + self.offset_ms

        current_text = ""
        
        # 1. Try to find subtitle in the currently active language
        if self.active_language in self.subtitles_cache:
            current_text = self._find_text(current_time, self.subtitles_cache[self.active_language])
            
        # 2. Fall back to PREVIOUS completed language (e.g. Yoruba while Hausa streams)
        if not current_text and self.previous_language:
            if (self.previous_language != self.active_language and 
                self.previous_language in self.subtitles_cache):
                current_text = self._find_text(current_time, self.subtitles_cache[self.previous_language])

        # 3. Last resort: fall back to original English
        if not current_text and self.active_language != "Original (no translation)":
            if "Original (no translation)" in self.subtitles_cache:
                current_text = self._find_text(current_time, self.subtitles_cache["Original (no translation)"])

        if current_text:
            self.overlay.set_subtitle(current_text)
        else:
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
