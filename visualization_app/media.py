"""Shared in-memory tactical video excerpt encoding."""
import base64
import subprocess
import streamlit as st


@st.cache_data(show_spinner="Preparing video clip...")
def video_window(path, modified, fps, start, end):
    """Decode only the configured window when the user runs the application.

    Resize and encode a browser-compatible clip in memory; do not persist match media.
    modified is a cache key so replacing the source video invalidates the cached clip.
    """
    import imageio_ffmpeg
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-ss", str(start / fps),
               "-i", path, "-t", str((end - start) / fps), "-an",
               "-vf", "scale=1280:-2", "-r", str(fps), "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "25", "-pix_fmt", "yuv420p",
               "-movflags", "frag_keyframe+empty_moov", "-f", "mp4", "pipe:1"]
    result = subprocess.run(command, capture_output=True, check=True)
    if not result.stdout:
        raise ValueError("No video frames decoded. Check the video path and time window.")
    return base64.b64encode(result.stdout).decode("ascii")

