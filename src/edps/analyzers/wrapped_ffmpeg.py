import static_ffmpeg
from ffmpeg import input as input
from ffmpeg import probe as probe

# Add ffmpeg binaries to environment if they do not already exist.
static_ffmpeg.add_paths(weak=True)
