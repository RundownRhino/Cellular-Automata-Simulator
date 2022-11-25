import os
from pathlib import Path
from tempfile import mkstemp

import gameoflife_ndimage.simulation as sim
from gameoflife_ndimage.video import Recorder


def test_recording_basic():
    rules = sim.Rules2D.classic()
    size = (256, 256)
    draw_params = sim.DrawParams(dead_color=[0, 0, 0], alive_color=[255, 255, 255], resize_factor=4)

    state = sim.State2D.random(rules, size)
    input_wh = tuple(a * draw_params.resize_factor for a in state.wh)

    fd, path_str = mkstemp(suffix=".mp4")
    path = Path(path_str)
    assert path.exists()
    recorder = None
    try:
        recorder = Recorder(
            framerate=5,
            input_wh=input_wh,
            output_path=path,
        )
        state.run_and_record(100, draw_params, recorder)
        recorder.close()

        # Investigate the results.
        assert path.exists()
        stats = path.stat()
        # for me it's about 1M, so let's just draw some rough boundaries
        assert 500 * 10**3 < stats.st_size < 2 * 10**6
    finally:
        # tempfile cleanup.
        try:
            if recorder:
                recorder.close()
        finally:
            try:
                os.close(fd)  # just in case
            finally:
                os.unlink(path)
    assert not path.exists()
