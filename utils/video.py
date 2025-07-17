import imageio

def record_video(frames, filename, fps=30):
    with imageio.get_writer(filename, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
