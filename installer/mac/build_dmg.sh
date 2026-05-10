#!/usr/bin/env bash
# installer/mac/build_dmg.sh
# Creates a drag-to-install macOS DMG from a PyInstaller .app bundle.
# Requires: create-dmg  (brew install create-dmg)
# Run AFTER PyInstaller:  bash installer/mac/build_dmg.sh [arm64|x86_64]

set -euo pipefail

ARCH="${1:-$(uname -m)}"
APP_NAME="PollenAnalysisTool"
# FIX 1: Read version from spec or default — avoids hardcoded "1.0.0" in DMG name
VERSION="${APP_VERSION:-1.0.0}"
APP_PATH="dist/${APP_NAME}.app"
DMG_NAME="${APP_NAME}-${VERSION}-macOS-${ARCH}.dmg"
STAGING="dist/dmg-staging"

echo "▶  Building DMG for arch=${ARCH}  app=${APP_PATH}  version=${VERSION}"

if [[ ! -d "${APP_PATH}" ]]; then
    echo "ERROR: ${APP_PATH} not found. Run PyInstaller first."
    exit 1
fi

rm -rf "${STAGING}" && mkdir -p "${STAGING}"
cp -r "${APP_PATH}" "${STAGING}/"

# FIX 2: Sign the .app BEFORE packaging into DMG (not during — create-dmg
# copies files, breaking signatures applied inside the staging folder).
# Also notarize the final DMG separately in CI after this script exits.
if [[ -n "${CODESIGN_IDENTITY:-}" ]]; then
    echo "▶  Code-signing with identity: ${CODESIGN_IDENTITY}"
    # Sign all inner binaries first (deep sign via find), then the .app itself.
    # This order matters: outer bundle must be signed last.
    find "${STAGING}/${APP_NAME}.app/Contents" \
        \( -name "*.dylib" -o -name "*.so" -o -name "*.framework" \) \
        -exec codesign --force --options runtime \
            --sign "${CODESIGN_IDENTITY}" \
            --entitlements installer/mac/entitlements.plist {} \;
    codesign --deep --force --verify --verbose \
             --options runtime \
             --sign "${CODESIGN_IDENTITY}" \
             --entitlements installer/mac/entitlements.plist \
             "${STAGING}/${APP_NAME}.app"
    echo "▶  Verifying signature..."
    codesign --verify --deep --strict --verbose=2 "${STAGING}/${APP_NAME}.app"
fi

# FIX 3: Add --background and --no-internet-enable for a cleaner DMG.
# --no-internet-enable prevents macOS from quarantining files inside the DMG.
create-dmg \
    --volname "Pollen Analysis Tool" \
    --volicon "assets/icon.icns" \
    --window-pos 200 120 \
    --window-size 660 400 \
    --icon-size 128 \
    --icon "${APP_NAME}.app" 160 185 \
    --hide-extension "${APP_NAME}.app" \
    --app-drop-link 500 185 \
    --no-internet-enable \
    "dist/${DMG_NAME}" \
    "${STAGING}/"

echo "✅  DMG created: dist/${DMG_NAME}"
echo "ℹ️   Next step in CI: notarize dist/${DMG_NAME} with xcrun notarytool"
