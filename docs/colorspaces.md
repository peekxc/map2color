Color space | Year / Originator | Original domain | Coordinate meaning | Device dependence | Perceptual uniformity | Typical gamut | Pros | Cons | Still used for
------------|------------------|-----------------|--------------------|-------------------|-----------------------|---------------|------|------|--------------
RGB         | 1861 Maxwell; formalized 1930s | Display devices, additive light | (R,G,B) primaries | Strong | Very poor | Device-limited | Simple, linear hardware mapping | Not perceptual, mixing unintuitive | Displays, image storage, shaders
sRGB        | 1996 HP/Microsoft | Consumer displays, web | Gamma-corrected RGB | Strong | Poor | Moderate | Standardized, ubiquitous | Clips colors, non-uniform | Web, images, UI
Adobe RGB   | 1998 Adobe | Printing, photography | Wide-gamut RGB | Strong | Poor | Larger than sRGB | Better CMYK mapping | Still non-uniform | Photography, print workflows
CMY/CMYK    | 19th c. printing | Subtractive printing | Ink absorption | Strong | Very poor | Printer-limited | Matches ink physics | Nonlinear, device-specific | Printing
HSV         | 1978 Smith | UI, color picking | Hue, Saturation, Value | Strong | Very poor | RGB-derived | Intuitive controls | Discontinuous, non-metric | Color pickers
HSL         | 1978 Smith | UI | Hue, Saturation, Lightness | Strong | Very poor | RGB-derived | Symmetric lightness | Still non-uniform | UI tools
HSI         | 1970s | Image analysis | Hue, Saturation, Intensity | Strong | Very poor | RGB-derived | Separates intensity | Rarely standardized | Vision research
YIQ         | 1953 NTSC | Analog TV | Luma + chroma | Strong | Poor | Broadcast-limited | Backward compatibility | Obsolete | Legacy TV
YCbCr       | 1982 JPEG | Video compression | Luma + chroma diffs | Strong | Poor | Video-limited | Efficient compression | Not perceptual | Video codecs
CIEXYZ      | 1931 CIE | Color science | Imaginary primaries | Weak | Poor | Encloses all visible | Linear, foundational | Not intuitive | Reference space
xyY         | 1931 CIE | Colorimetry | Chromaticity + luminance | Weak | Poor | Same as XYZ | Separates brightness | Still non-uniform | Chromaticity diagrams
CIELAB      | 1976 CIE | Perceptual color difference | L*, a*, b* | Weak | Approx. uniform | Large | ΔE meaningful | Fails at extremes | Color difference, QC
CIELUV      | 1976 CIE | Lighting, displays | L*, u*, v* | Weak | Approx. uniform | Large | Better for additive light | Less common | Lighting
Hunter Lab  | 1948 Hunter | Industrial QC | Nonlinear XYZ | Weak | Moderate | Medium | Early perceptual model | Superseded by CIELAB | Legacy QC
IPT         | 1998 Fairchild | HDR imaging | Intensity, Protan, Tritan | Weak | Better than Lab | HDR | Better hue linearity | Complex | HDR research
ICtCp       | 2015 Dolby | HDR video | Intensity + chroma | Strong | Moderate | HDR | Good for compression | Not intuitive | HDR video
Oklab       | 2020 Björn Ottosson | Graphics, UI | Perceptual Lab-like | Weak | High | Large | Smooth gradients | New, less standardized | Modern graphics
Oklch       | 2020 | Graphics | Polar Oklab | Weak | High | Large | Intuitive hue control | Same limits as Oklab | CSS, design
Munsell     | 1905 Munsell | Painting, education | Hue, Value, Chroma | Weak | High | Reflective | Artist-friendly | Hard to compute | Art, education
NCS         | 1966 Hård & Sivik | Design | Perceptual oppositions | Weak | High | Reflective | Human-centric | Not device-ready | Architecture
PCCS        | 1964 Japan | Design education | Hue, Tone | Weak | High | Reflective | Pedagogical | Regional | Education
