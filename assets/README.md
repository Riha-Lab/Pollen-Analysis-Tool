# assets/

Place your icon files here before building:

| File | Platform | Description |
|------|----------|-------------|
| `icon.ico` | Windows | Multi-resolution ICO (256×256 recommended) |
| `icon.icns` | macOS | ICNS bundle (use `iconutil` or [img2icns](https://img2icns.com/)) |
| `icon.png` | Linux / Docker | PNG, 512×512 |
| `dmg_background.png` | macOS DMG | Background image, 660×400 px |

If you already have JPG/PNG/ICO/ICNS files, just rename/copy them here.

## Converting PNG → ICO (Windows)
Install ImageMagick, then:
```
magick icon.png -define icon:auto-resize="256,128,64,48,32,16" icon.ico
```

## Converting PNG → ICNS (macOS)
```bash
mkdir icon.iconset
sips -z 16 16     icon.png --out icon.iconset/icon_16x16.png
sips -z 32 32     icon.png --out icon.iconset/icon_16x16@2x.png
sips -z 32 32     icon.png --out icon.iconset/icon_32x32.png
sips -z 64 64     icon.png --out icon.iconset/icon_32x32@2x.png
sips -z 128 128   icon.png --out icon.iconset/icon_128x128.png
sips -z 256 256   icon.png --out icon.iconset/icon_128x128@2x.png
sips -z 256 256   icon.png --out icon.iconset/icon_256x256.png
sips -z 512 512   icon.png --out icon.iconset/icon_256x256@2x.png
sips -z 512 512   icon.png --out icon.iconset/icon_512x512.png
iconutil -c icns icon.iconset
```
