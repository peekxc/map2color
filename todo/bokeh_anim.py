import os
import tempfile
import subprocess
from typing import Union, Optional, Sequence


def animate_seq(
	figures: Union[str, Sequence],
	output_fn: str,
	fps: int = 10,
	scale: float = 1.0,
	format: str = "mp4",
	output_dn: str = None,
	width: int = 250,
	height: int = 250,
) -> str:
	from bokeh.io import export_png, export_svg
	from tqdm import tqdm
	from selenium import webdriver
	from selenium.webdriver.chrome.options import Options

	## Initialize headless browser session
	chrome_options = Options()
	chrome_options.add_argument("--headless")
	driver = webdriver.Chrome(options=chrome_options)

	tmpdir = tempfile.TemporaryDirectory().name if output_dn is None else output_dn
	if not os.path.isdir(tmpdir):
		os.makedirs(tmpdir)

	file_paths = []
	w, h = int(width), int(height)
	print(f"Saving outputs to: {tmpdir}")
	tmp_dir_name = str(tmpdir)
	for i, fig in tqdm(enumerate(figures)):
		file_path = os.path.join(tmp_dir_name, f"frame_{i:05d}.png")
		export_png(fig, filename=file_path, webdriver=driver, scale_factor=scale)
		file_paths.append(file_path)
	driver.quit()

	# Convert PNGs to MP4 using ffmpeg
	output_file = f"{output_fn}.{format}"
	ffmpeg_cmd = ["ffmpeg", "-y"]
	ffmpeg_cmd += ["-framerate", str(fps)]
	ffmpeg_cmd += ["-i", os.path.join(tmp_dir_name, "frame_%05d.png")]
	ffmpeg_cmd += ["-vf", f"scale={w}:{h}"]
	ffmpeg_cmd += ["-c:v", "libx264"]
	# ffmpeg_cmd += ["-r", str(fps)]
	ffmpeg_cmd += ["-pix_fmt", "yuv420p"]
	ffmpeg_cmd += ["-loop", "0", output_file]

	# subprocess.run(ffmpeg_cmd, check=True)
	cmd = " ".join(ffmpeg_cmd)
	status = subprocess.call(f"source ~/.bash_profile && {cmd}", shell=True)
	return status
