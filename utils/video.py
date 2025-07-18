import imageio
import os

def record_video(frames, filename, fps=30):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with imageio.get_writer(filename, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
