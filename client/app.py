"""
Kalaye — Desktop Application
==============================
Beautiful full-window GUI for AI-powered subtitle generation.
Drop a movie file → get subtitles in seconds → overlay on any player.
"""

import sys
import os
import tempfile

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QTextEdit, QComboBox,
    QProgressBar, QFrame, QSizePolicy, QGraphicsDropShadowEffect,
    QSpacerItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation, QSize, QMimeData
from PyQt5.QtGui import (
    QFont, QColor, QIcon, QPalette, QDragEnterEvent, QDropEvent,
    QPainter, QLinearGradient, QBrush, QPen, QFontDatabase
)

from client.overlay import SubtitleOverlay, SubtitlePlayer
from client.audio_sync import AudioSyncEngine


# ============================================================================
# Color Palette & Presentation Styling
# ============================================================================

COLORS = {
    "bg":           "#F0F4F8",    # Very light cool blue/slate background
    "surface":      "#FFFFFF",    # Clean white for cards and inputs
    "border":       "#CBD5E1",    # Soft slate borders
    "text_main":    "#0F172A",    # Midnight blue text to tie into the theme
    "text_muted":   "#475569",    # Slate gray secondary text
    "primary":      "#1D4D9A",    # Kalaye Logo exact blue!
    "primary_glow": "#2C66C9",    # Lighter logo blue for hover
    "success":      "#10B981",    # Emerald
}

STYLESHEET = f"""
QMainWindow {{
    background-color: {COLORS['bg']};
}}

QWidget {{
    color: {COLORS['text_main']};
    font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
    font-size: 18px;
}}

/* Beautiful Group Boxes */
QGroupBox {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    margin-top: 36px;
    padding-top: 20px;
    font-weight: bold;
    font-size: 19px;
    color: {COLORS['text_muted']};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 24px;
    top: 0px;
    background-color: {COLORS['primary']};
    color: white;
    padding: 8px 18px;
    border-radius: 8px;
}}

/* Modern Inputs */
QLineEdit {{
    background-color: {COLORS['bg']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 10px 14px;
    color: {COLORS['text_main']};
}}
QLineEdit:focus {{
    border: 1px solid {COLORS['primary']};
}}

/* Modern Combo Box */
QComboBox {{
    background-color: {COLORS['bg']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 10px 14px;
}}
QComboBox::drop-down {{
    border: none;
}}

/* Premium Buttons */
QPushButton {{
    background-color: {COLORS['surface']};
    color: {COLORS['text_main']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: bold;
}}
QPushButton:hover {{
    background-color: {COLORS['border']};
    color: {COLORS['text_main']};
}}
QPushButton:disabled {{
    color: #94A3B8;
    background-color: {COLORS['surface']};
}}

/* Magic 'Action' Buttons */
QPushButton#ActionBtn {{
    background-color: {COLORS['primary']};
    color: white;
    border: none;
    font-size: 18px;
    padding: 12px 24px;
    border-radius: 8px;
}}
QPushButton#ActionBtn:hover {{
    background-color: {COLORS['primary_glow']};
}}
QPushButton#ActionBtn:disabled {{
    background-color: #C4C4C4;
    color: #6B7280;
}}

/* Progress Bar */
QProgressBar {{
    background-color: {COLORS['bg']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    text-align: center;
    color: transparent;
    height: 12px;
}}
QProgressBar::chunk {{
    background-color: {COLORS['success']};
    border-radius: 5px;
}}

/* Text Editors */
QTextEdit {{
    background-color: {COLORS['bg']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 16px;
    font-family: 'Consolas', monospace;
    font-size: 16px;
    color: {COLORS['text_muted']};
}}
"""


# ============================================================================
# Pipeline Worker Thread
# ============================================================================

class PipelineWorker(QThread):
    """Runs the AI pipeline in a background thread."""

    progress = pyqtSignal(str, int)   # (message, percent)
    log = pyqtSignal(str)             # log line
    finished = pyqtSignal(dict)       # result dict
    error = pyqtSignal(str)           # error message

    def __init__(self, file_path, language=None):
        super().__init__()
        self.file_path = file_path
        self.language = language

    def run(self):
        try:
            import subprocess
            import json
            import tempfile
            import os

            self.progress.emit("Connecting to cloud...", 10)
            self.log.emit("☁️ Connecting to cloud servers...")

            # Use a temp json file to pass the result back
            json_out = os.path.join(tempfile.gettempdir(), "kalaye_pipeline_out.json")

            cmd = [
                "modal", "run", "server/pipeline.py",
                "--filepath", self.file_path,
                "--json-out", json_out
            ]
            if self.language:
                cmd.extend(["--language", self.language])

            # Force UTF-8 encoding in the child python process so modal doesn't crash 
            # Force unbuffered output so logs appear instantly instead of hanging
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUNBUFFERED"] = "1"

            # Run pipeline as a separate subprocess
            # This completely avoids the PyQT <-> Asyncio heartbeat bugs
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )

            for line in iter(process.stdout.readline, ""):
                line = line.strip()
                if not line:
                    continue
                
                # Full verbose output → developer's terminal only
                print(f"[Pipeline] {line}", flush=True)

                # Only emit user-friendly stage updates to the GUI
                if "[Pre-step]" in line:
                    self.progress.emit("Extracting audio from video...", 15)
                    self.log.emit(" Extracting audio track from your media file...")
                elif "[Step 1/3]" in line:
                    self.progress.emit("Transcribing with Whisper AI...", 30)
                    self.log.emit(" AI is listening and transcribing speech...")
                elif "[Step 2/3]" in line:
                    self.progress.emit("Detecting environmental sounds...", 70)
                    self.log.emit(" Detecting background sounds & effects...")
                elif "[Step 3/3]" in line:
                    self.progress.emit("Merging subtitles...", 90)
                    self.log.emit(" Assembling your final subtitles...")
                elif "PIPELINE COMPLETE" in line:
                    self.progress.emit("Done!", 100)
                    self.log.emit(" All done! Your subtitles are ready.")

            process.wait()

            if process.returncode != 0:
                print(f"[Pipeline] FAILED with exit code {process.returncode}", flush=True)
                self.error.emit("Pipeline encountered an error. Check terminal for details.")
                return

            # Read the JSON result back
            if os.path.exists(json_out):
                with open(json_out, "r", encoding="utf-8") as f:
                    result = json.load(f)
                self.finished.emit(result)
            else:
                self.error.emit("Pipeline did not generate output.")

        except Exception as e:
            self.error.emit(str(e))


# ============================================================================
# Translation Worker Thread (Streaming v2)
# ============================================================================

class TranslateWorker(QThread):
    """Translates SRT content using Gemini in streaming chunks.

    Instead of waiting 3-5 minutes for the full translation,
    this emits translated SRT chunks every ~10 seconds so the
    overlay can start showing subtitles immediately.
    """

    progress = pyqtSignal(str)          # status message
    chunk_ready = pyqtSignal(str, int, int, str)  # (srt_chunk, chunk_idx, total_chunks, target_lang)
    finished = pyqtSignal(str)          # full translated SRT
    error = pyqtSignal(str)

    def __init__(self, srt_content, target_language):
        super().__init__()
        self.srt_content = srt_content
        self.target_language = target_language

    def run(self):
        try:
            from dotenv import load_dotenv
            load_dotenv()

            from server.translator import translate_srt_chunked

            self.progress.emit(f"Translating to {self.target_language} (streaming)...")

            all_srt_chunks = []

            for chunk_data in translate_srt_chunked(self.srt_content, self.target_language):
                # Check for errors in individual chunks
                if "error" in chunk_data:
                    self.progress.emit(
                        f"⚠️ Chunk {chunk_data['chunk_index'] + 1} failed: {chunk_data['error']}"
                    )
                    continue

                srt_chunk = chunk_data.get("srt_chunk", "")
                chunk_idx = chunk_data["chunk_index"]
                total = chunk_data["total_chunks"]

                if srt_chunk:
                    all_srt_chunks.append(srt_chunk)
                    # Emit each chunk with explicit target_language so if combobox changes mid-streaming, 
                    # it still appends to the correct dictionary list!
                    self.chunk_ready.emit(srt_chunk, chunk_idx, total, self.target_language)

                self.progress.emit(
                    f"🌍 Translated chunk {chunk_idx + 1}/{total} "
                    f"({chunk_data.get('blocks_out', '?')} blocks, "
                    f"{chunk_data.get('elapsed_seconds', '?')}s)"
                )

            # Stitch all chunks into full SRT
            full_srt = "\n\n".join(all_srt_chunks)
            self.finished.emit(full_srt)

        except Exception as e:
            self.error.emit(str(e))



# SyncHeartbeat REMOVED — replaced by AudioSyncEngine (audio_sync.py).
# The old CPU-based approach used process CPU% to guess play/pause,
# which caused false pauses during dark scenes and accumulated drift.
# AudioSyncEngine uses audio fingerprint cross-correlation instead,
# giving ±100ms accuracy with any video player.




# ============================================================================
# Main Window (Classic 2006 Desktop Style)
# ============================================================================

class KalayeWindow(QMainWindow):
    """Main Kalaye desktop application window."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Kalaye AI — Premium Subtitles")
        self.setMinimumSize(900, 700)
        self.resize(1000, 800)
        self.setStyleSheet(STYLESHEET)

        # State
        self.srt_content = None
        self.translated_srt = None  # stores the translated version separately
        self.current_file = None
        self.overlay = None
        self.controls = None
        self.player = None
        self.worker = None
        self.translate_worker = None
        self.audio_sync = None
        self.log_visible = False

        self._build_ui()

    def _build_ui(self):
        from PyQt5.QtWidgets import QGroupBox, QLineEdit, QFormLayout, QHBoxLayout, QLabel
        from PyQt5.QtGui import QPixmap
        from PyQt5.QtCore import Qt
        import os
    
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(24)

        # ---------------------------------------------------------
        # Header Area
        # ---------------------------------------------------------
        header_layout = QHBoxLayout()
        header_label = QLabel()
        
        # Safely resolve path to web/logo.png from the root directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logo_path = os.path.join(base_dir, "web", "logo.png")
        
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            # Scale logo beautifully to match header height, smoothing edges
            scaled_pixmap = pixmap.scaledToHeight(60, Qt.SmoothTransformation)
            header_label.setPixmap(scaled_pixmap)
        else:
            # Fallback if image isn't found
            header_label.setText(" KALAYE AI")
            header_label.setStyleSheet("font-size: 35px; font-weight: 800; color: white;")
            
        header_layout.addWidget(header_label)
        header_layout.addStretch() # Push logo to the left
        main_layout.addLayout(header_layout)

        # ---------------------------------------------------------
        # Group Box: 1. Input & Generation
        # ---------------------------------------------------------
        group_input = QGroupBox("1. Video Input & Generation")
        layout_input = QFormLayout()
        
        # File selector row
        file_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        
        self.browse_btn = QPushButton(" Browse Media")
        self.browse_btn.clicked.connect(self._browse)
        
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(self.browse_btn)
        
        # Start button
        self.start_btn = QPushButton(" Generate Original Subtitles")
        self.start_btn.setObjectName("ActionBtn")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._on_start)
        
        layout_input.addRow("Select Media:", file_layout)
        layout_input.addRow("", self.start_btn)
        group_input.setLayout(layout_input)
        main_layout.addWidget(group_input)

        # ---------------------------------------------------------
        # Group Box: 2. Status & Preview
        # ---------------------------------------------------------
        group_status = QGroupBox("2. Status & Preview")
        layout_status = QVBoxLayout()
        
        # Progress
        progress_layout = QHBoxLayout()
        self.progress_label = QLabel("Ready.")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        layout_status.addLayout(progress_layout)
        
        # Logs / Preview Tabs (Just stacked for classic feel)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setPlaceholderText("Pipeline logs will appear here...")
        
        self.srt_preview = QTextEdit()
        self.srt_preview.setReadOnly(True)
        self.srt_preview.setPlaceholderText("Generated Subtitles Preview...")
        
        splitter = QHBoxLayout()
        splitter.addWidget(self.log_area)
        splitter.addWidget(self.srt_preview)
        layout_status.addLayout(splitter)
        
        group_status.setLayout(layout_status)
        main_layout.addWidget(group_status)

        # ---------------------------------------------------------
        # Group Box: 3. Translation & Overlay
        # ---------------------------------------------------------
        group_output = QGroupBox("3. Translation & Native Overlay")
        layout_output = QVBoxLayout()
        
        ctrl_layout = QHBoxLayout()
        
        self.lang_combo = QComboBox()
        self.lang_combo.addItems([
            "Original (no translation)",
            "Yoruba", "Hausa", "Igbo", "English", "French", "Spanish", 
            "Arabic", "Japanese", "Chinese (Simplified)", 
            "Portuguese", "German", "Hindi"
        ])
        self.lang_combo.currentTextChanged.connect(self._on_lang_changed)
        
        self.translate_btn = QPushButton("↹ Translate Pipeline")
        self.translate_btn.setObjectName("ActionBtn")
        self.translate_btn.clicked.connect(self._on_translate)
        self.translate_btn.setEnabled(False)
        
        self.overlay_btn = QPushButton("❐ Launch Magic Overlay")
        self.overlay_btn.setObjectName("ActionBtn")
        self.overlay_btn.clicked.connect(self._on_launch_overlay)
        self.overlay_btn.setEnabled(False)
        
        ctrl_layout.addWidget(QLabel("Target Language:"))
        ctrl_layout.addWidget(self.lang_combo)
        ctrl_layout.addWidget(self.translate_btn)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.overlay_btn)
        
        save_layout = QHBoxLayout()
        self.save_btn = QPushButton("⎘ Save Source SRT")
        self.save_btn.clicked.connect(self._on_save_srt)
        self.save_btn.setEnabled(False)
        
        self.download_translated_btn = QPushButton("⭳ Export Translated SRT")
        self.download_translated_btn.clicked.connect(self._on_download_translated)
        self.download_translated_btn.setEnabled(False)
        
        save_layout.addWidget(self.save_btn)
        save_layout.addWidget(self.download_translated_btn)
        save_layout.addStretch()
        
        layout_output.addLayout(ctrl_layout)
        layout_output.addLayout(save_layout)
        group_output.setLayout(layout_output)
        main_layout.addWidget(group_output)

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _browse(self):
        """User browsed for a movie file."""
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Movie File", "",
            "Media Files (*.mp4 *.mkv *.avi *.mov *.wmv *.flv *.webm *.mp3 *.wav *.m4a)"
        )
        if path:
            self._on_file_selected(path)

    def _on_file_selected(self, path):
        """Load file logic."""
        self.current_file = path
        self.file_path_edit.setText(path)
        
        self.start_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Ready — click Generate to start")
        self.log_area.clear()
        self.srt_preview.clear()
        self.srt_content = None
        
        self.overlay_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.translate_btn.setEnabled(False)

    def _on_start(self):
        """Start the pipeline."""
        if not self.current_file:
            return

        self.start_btn.setEnabled(False)
        self.start_btn.setText("Processing...")
        self.log_area.clear()

        self.worker = PipelineWorker(self.current_file)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._on_log)
        self.worker.finished.connect(self._on_pipeline_done)
        self.worker.error.connect(self._on_pipeline_error)
        self.worker.start()

    def _on_progress(self, message, percent):
        """Pipeline progress update."""
        self.progress_bar.setValue(percent)
        self.progress_label.setText(message)

    def _on_log(self, line):
        """Pipeline log line."""
        self.log_area.append(line)
        # Auto-scroll to bottom
        scrollbar = self.log_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_translate_log(self, line):
        """Translation log — verbose to terminal, friendly to GUI."""
        print(f"[Translate] {line}", flush=True)
        # Only show chunk progress in GUI, not raw details
        if "Translated chunk" in line:
            # Extract just the chunk count part
            self.log_area.append(f" {line}")
            scrollbar = self.log_area.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def _on_pipeline_done(self, result):
        """Pipeline completed successfully."""
        self.srt_content = result["srt_content"]

        # -------------------------------------------------------------
        # Pre-initialize the Overlay and Player SILENTLY.
        # This allows the player to start caching translated chunks
        # immediately, even before the user clicks "Launch Overlay".
        # -------------------------------------------------------------
        if not self.overlay:
            from client.overlay import SubtitleOverlay, OverlayControls
            self.overlay = SubtitleOverlay()
            self.controls = OverlayControls(self.overlay)
            self.controls.translation_requested.connect(self._on_overlay_translation_request)
            
        import tempfile
        srt_path = os.path.join(tempfile.gettempdir(), "kalaye_overlay.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(self.srt_content)
            
        self.player = SubtitlePlayer(self.overlay)
        self.player.load_srt(srt_path)
        self.overlay.player = self.player

        # Update UI
        self.start_btn.setText("Generate English Subtitles")
        self.start_btn.setEnabled(True)

        self.overlay_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.translate_btn.setEnabled(True)

        # Show SRT preview
        preview = self.srt_content[:2000]
        if len(self.srt_content) > 2000:
            preview += "\n\n... (truncated)"
        self.srt_preview.setText(preview)

        self.progress_label.setText(
            f" Done — {result.get('dialogue_count', '?')} subtitles, "
            f"{len(result.get('sound_tags', []))} sound tags"
        )

        # Friendly summary in GUI
        self._on_log(f"\n Pipeline complete!")
        self._on_log(f"   {result.get('dialogue_count', '?')} subtitles generated")
        # Full details to terminal
        print(f"[Pipeline] Complete! Language: {result.get('language', 'auto')}", flush=True)
        print(f"[Pipeline] Subtitles: {result.get('dialogue_count', '?')}", flush=True)
        print(f"[Pipeline] Sound tags: {', '.join(result.get('sound_tags', [])) or '(none)'}", flush=True)

    def _on_pipeline_error(self, error_msg):
        """Pipeline failed."""
        self.start_btn.setText("Generate English Subtitles")
        self.start_btn.setEnabled(True)
        self.progress_label.setText("Error")
        self.progress_bar.setValue(0)
        self._on_log("\n Something went wrong. Please check terminal for details.")
        print(f"[Pipeline] ERROR: {error_msg}", flush=True)

    def _on_lang_changed(self, target):
        """Instantly switch subtitle language in the overlay when dropdown changes."""
        if self.player:
            self.player.active_language = target

    def _on_overlay_translation_request(self, target):
        """Triggered when user picks an uncached language directly from the floating overlay."""
        # Sync the main UI dropdown
        self.lang_combo.blockSignals(True)
        self.lang_combo.setCurrentText(target)
        self.lang_combo.blockSignals(False)
        
        # Start streaming!
        self._on_translate()

    def _on_translate(self):
        """Translate subtitles to selected language (streaming)."""
        if not self.srt_content:
            return

        target = self.lang_combo.currentText()
        if target == "Original (no translation)":
            return

        self.translate_btn.setEnabled(False)
        self.translate_btn.setText("Translating...")
        self.download_translated_btn.setEnabled(False)
        
        # We don't clear translated_srt here if we want to store multiple, but for UI
        # let's just keep track of the *most recent* full translation for downloading.
        self.translated_srt = None

        # Stop any previous translation still running so its stale chunk_ready
        # signals can't bleed into this new session.
        if self.translate_worker and self.translate_worker.isRunning():
            self.translate_worker.chunk_ready.disconnect()
            self.translate_worker.finished.disconnect()
            self.translate_worker.error.disconnect()
            self.translate_worker.quit()
            self.translate_worker.wait(500)
        
        if self.player:
            # Save the current language as fallback while new one streams
            if (self.player.active_language != "Original (no translation)" and
                self.player.active_language in self.player.subtitles_cache and
                len(self.player.subtitles_cache[self.player.active_language]) > 0):
                self.player.previous_language = self.player.active_language
            
            self.player.active_language = target
            if target not in self.player.subtitles_cache:
                self.player.subtitles_cache[target] = []

        self.translate_worker = TranslateWorker(self.srt_content, target)
        self.translate_worker.progress.connect(self._on_translate_log)
        self.translate_worker.chunk_ready.connect(self._on_translate_chunk)
        self.translate_worker.finished.connect(self._on_translate_done)
        self.translate_worker.error.connect(self._on_translate_error)
        self.translate_worker.start()

    def _on_translate_chunk(self, srt_chunk, chunk_idx, total_chunks, target_lang):
        """A translated chunk arrived — update preview and overlay live."""
        # Update progress label
        self.progress_label.setText(
            f"Translating {target_lang}... chunk {chunk_idx + 1}/{total_chunks}"
        )
        # Update progress bar
        percent = int(((chunk_idx + 1) / total_chunks) * 100)
        self.progress_bar.setValue(percent)

        # Append to live SRT preview (show latest chunk at the bottom)
        self.srt_preview.append(f"\n--- Chunk {chunk_idx + 1} ({target_lang}) ---\n")
        # Show first 500 chars of this chunk
        preview = srt_chunk[:500]
        if len(srt_chunk) > 500:
            preview += "\n..."
        self.srt_preview.append(preview)
        # Auto-scroll
        scrollbar = self.srt_preview.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

        # If the overlay exists, append this chunk's subtitles live 
        # (even if hidden or paused)
        if self.player:
            self._append_srt_to_player(srt_chunk, target_lang)

    def _parse_time_ms(self, t_str):
        """Convert '00:01:03,600' to total milliseconds."""
        import traceback
        try:
            parts = t_str.replace(',', ':').split(':')
            if len(parts) == 4:
                h, m, s, ms = map(int, parts)
                return (h * 3600000) + (m * 60000) + (s * 1000) + ms
        except:
            pass
        return 0

    def _append_srt_to_player(self, srt_chunk, target_lang):
        """Parse an SRT chunk and append it directly to the active language queue.
        Uses a robust manual parser so we don't rely on strict pysrt formatting.
        """
        try:
            if target_lang not in self.player.subtitles_cache:
                self.player.subtitles_cache[target_lang] = []
                
            blocks = srt_chunk.strip().split("\n\n")
            for block in blocks:
                lines = [line.strip() for line in block.split("\n") if line.strip()]
                if len(lines) < 3:
                     continue
                
                # Check line 2 for timestamps (e.g., '00:00:10,000 --> 00:00:12,000')
                ts_line = lines[1]
                if '-->' not in ts_line:
                     # Sequence number might be missing, check line 0
                     if '-->' in lines[0]:
                          ts_line = lines[0]
                          text = " ".join(lines[1:])
                     else:
                          continue
                else:
                     text = " ".join(lines[2:])
                     
                start_str, end_str = [x.strip() for x in ts_line.split("-->")]
                start_ms = self._parse_time_ms(start_str)
                end_ms = self._parse_time_ms(end_str)
                
                if end_ms > start_ms:
                    entry = (start_ms, end_ms, text)
                    self.player.subtitles_cache[target_lang].append(entry)
                
            # Keep sorted by start time
            self.player.subtitles_cache[target_lang].sort(key=lambda x: x[0])
            
        except Exception as e:
            self._on_log(f" Failed to parse chunk for overlay: {e}")

    def _on_translate_done(self, translated_srt):
        """Translation completed — all chunks received."""
        self.translated_srt = translated_srt
        self.translate_btn.setText("Translate (Streaming)")
        self.translate_btn.setEnabled(True)

        # Show full translated SRT in preview
        preview = translated_srt[:2000]
        if len(translated_srt) > 2000:
            preview += "\n\n... (truncated)"
        self.srt_preview.setText(preview)

        self.download_translated_btn.setEnabled(True)

        self.progress_label.setText("Translation complete!")
        self.progress_bar.setValue(100)
        self._on_log(f"Translation complete! {len(translated_srt):,} chars")
        
        # Mark this language as the latest completed translation for fallback
        if self.player:
            target = self.lang_combo.currentText()
            if target != "Original (no translation)":
                self.player.previous_language = target

    def _on_translate_error(self, error_msg):
        """Translation failed."""
        self.translate_btn.setText("Translate (Streaming)")
        self.translate_btn.setEnabled(True)
        self.progress_label.setText("Translation error")
        self._on_log(f"Translation error: {error_msg}")

    def _on_download_translated(self):
        """Save the translated SRT to a user-chosen location."""
        if not self.translated_srt:
            return

        # Build default filename
        default_name = ""
        if self.current_file:
            base = os.path.splitext(os.path.basename(self.current_file))[0]
            lang = self.lang_combo.currentText().replace(" ", "_")
            default_name = f"{base}_{lang}.srt"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Translated Subtitle", default_name,
            "SubRip Files (*.srt);;All Files (*)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.translated_srt)
            self._on_log(f"💾 Translated SRT saved to: {path}")

    def _on_launch_overlay(self):
        """Launch the floating subtitle overlay with VLC HTTP Sync."""
        if not self.player:
            return

        # Show the previously hidden overlay and controls
        self.overlay.show()
        if self.controls:
            self.controls.show()

        self._on_log(" Starting VLC Sync...")
        self._on_log("   Make sure VLC's Web interface is enabled (see docs)")

        # Stop any previous sync engine
        if self.audio_sync and self.audio_sync.isRunning():
            self.audio_sync.stop()
            self.audio_sync.wait(2000)

        self.audio_sync = AudioSyncEngine(self.current_file)
        self.audio_sync.position_found.connect(self._on_audio_sync_position)
        self.audio_sync.playback_state.connect(self._on_audio_sync_playback)
        self.audio_sync.status.connect(self._on_audio_sync_status)
        self.audio_sync.start()

        self.overlay_btn.setText("Overlay Running...")
        self.overlay_btn.setEnabled(False)

    def _on_audio_sync_position(self, position_ms, confidence):
        """VLC sync found the exact movie position."""
        if not self.player:
            return
        self.player.sync_to_position(position_ms)

    def _on_audio_sync_playback(self, is_playing):
        """VLC reported play/pause state change."""
        if not self.player:
            return
        if not is_playing and self.player.is_playing:
            self.player.pause()

    def _on_audio_sync_status(self, message):
        """Show sync status in the log panel."""
        self._on_log(f"   [Sync] {message}")

    def _on_save_srt(self):
        """Save SRT to file."""
        if not self.srt_content:
            return

        # Default filename from movie
        default_name = ""
        if self.current_file:
            base = os.path.splitext(os.path.basename(self.current_file))[0]
            default_name = f"{base}.srt"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Subtitle File", default_name,
            "SubRip Files (*.srt);;All Files (*)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.srt_content)
            self._on_log(f"💾 Saved to: {path}")

    def closeEvent(self, event):
        """Clean up on window close."""
        if self.audio_sync and self.audio_sync.isRunning():
            self.audio_sync.stop()
            self.audio_sync.wait(2000)
        if self.overlay:
            self.overlay.close()
        if self.player:
            self.player.stop()
        event.accept()


# ============================================================================
# Entry Point
# ============================================================================

def launch():
    """Launch the Kalaye desktop application."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = KalayeWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    launch()
