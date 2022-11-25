import pathlib
import warnings
from typing import Optional, Tuple

import ffmpeg
import numpy as np


class Recorder:
    """Handles the recording of a video from frames using ffmpeg"""

    def __init__(
            self, framerate: int, input_wh: Tuple[int, int], output_path,
            input_pixel_format: str = "rgb24", output_vcodec: str = "libx264",
            ffmpeg_input_kwargs: Optional[dict] = None,
            ffmpeg_output_kwargs: Optional[dict] = None,
            supress_stdout: bool = True,
    ):
        """

        :param framerate: Output ramerate to pass to ffmpeg
        :param input_wh: Input width and height.
        :param output_path: With the extension
        :param input_pixel_format: For example, "rgba" or "bgr24"
        :param output_vcodec:
        :param ffmpeg_input_kwargs:
        :param ffmpeg_output_kwargs:
        :param supress_stdout: Whether the stdout of ffmpeg should be suppressed instead of being
            output into the console.
        """
        # TODO: put some links to ffmpeg docs here
        if ffmpeg_input_kwargs is None:
            ffmpeg_input_kwargs = {}
        if ffmpeg_output_kwargs is None:
            ffmpeg_output_kwargs = {}
        if supress_stdout:
            ffmpeg_output_kwargs["loglevel"] = "quiet"
        process = (
            ffmpeg
            .input(
                'pipe:', framerate=str(int(framerate)), format='rawvideo',
                pix_fmt=input_pixel_format, s='{}x{}'.format(*input_wh),
                **ffmpeg_input_kwargs
            )
            .output(
                str(output_path), vcodec=output_vcodec, **ffmpeg_output_kwargs
            )
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        self.process = process
        self.wh = input_wh
        self.running = True
        self.output = pathlib.Path(output_path)

    def send_frame(self, frame: np.ndarray):
        """Sends a frame to ffmpeg to be encoded."""
        if tuple(frame.shape[:2][::-1]) != self.wh:
            raise ValueError("The frame provided has wrong size!")
        self.process.stdin.write(frame.tobytes())

    def close(self):
        """Close the recorder and wait for the ffmpeg process to finish."""
        if not self.running:
            return
        self.running = False
        self.process.stdin.close()
        self.process.wait()

    def __del__(self):
        if self.running:
            warnings.warn(
                "[Recorder] Warning: a running instance got dropped without being closed. "
                "Will attempt to close it.")
        self.close()
