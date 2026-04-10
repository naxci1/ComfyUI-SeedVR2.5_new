# Assets

Place your application icons here before building the EXE:

| File       | Usage                                                |
|------------|------------------------------------------------------|
| `icon.png` | Window icon shown in the title bar at runtime        |
| `icon.ico` | Windows taskbar / Explorer icon embedded in the EXE  |

## Creating a placeholder icon (Python 3 + Pillow)

```python
from PIL import Image, ImageDraw
img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)
draw.ellipse((16, 16, 240, 240), fill=(0, 180, 216, 255))
img.save("icon.png")
img.save("icon.ico", sizes=[(16,16),(32,32),(48,48),(256,256)])
```

If Pillow is not available the application runs without an icon; the build
will simply omit the `--icon` flag.
