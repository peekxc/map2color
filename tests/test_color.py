from difflib import get_close_matches

from map2color import colors_html_box
from IPython.display import display, HTML


def test_colorbar():
	HTML(colors_html_box(["#ff0000", "#00ff00", "#0000ff"], interpolate=True))


get_close_matches("TurBo", ["Turbo256", "Tubero_400"], n=1, cutoff=0.5)
