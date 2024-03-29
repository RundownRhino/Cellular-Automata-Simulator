# Cellular Automata Simulator

Uses `numpy` and `scipy.ndimage` to quickly simulate arbitrary rulesets for nearest-neighbours cellular automata. Can
generate videos and images of the results via `ffmpeg-python` and `pillow`.

### Usage example:

```python
import gameoflife_ndimage.simulation as sim
from gameoflife_ndimage.video import Recorder

if __name__ == '__main__':
    rules = sim.Rules2D.classic()
    size = (256, 256)
    draw_params = sim.DrawParams(dead_color=[0, 0, 0], alive_color=[255, 255, 255], resize_factor=4)

    state = sim.State2D.random(rules, size)
    input_wh = tuple(a * draw_params.resize_factor for a in state.wh)
    recorder = Recorder(
        framerate=5,
        input_wh=input_wh,
        output_path="output/gol_classic_{}x{}_from_random.mp4".format(*size),
    )
    state.run_and_record(100, draw_params, recorder)
    recorder.close()
```