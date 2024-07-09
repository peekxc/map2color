import importlib.metadata
__version__ = importlib.metadata.version("map2color")


from .color import map2hex, map2rgb, hex2rgb, rgb2hex, color2hex