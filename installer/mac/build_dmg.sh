#!/usr/bin/env bash
# installer/mac/build_dmg.sh
# Creates a drag-to-install macOS DMG from a PyInstaller .app bundle.
# Requires: create-dmg  (brew install create-dmg)
# Run AFTER PyInstaller:  bash installer/mac/build_dmg.sh [arm64|x86_64]

set -euo pipefail

ARCH="${1:-$(uname -m)}"
APP_NAME="PollenAnalysisTool"
VERSION="1.0.0"
APP_PATH="dist/${APP_NAME}.app"
DMG_NAME="${APP_NAME}-${VERSION}-macOS-${ARCH}.dmg"
STAGING="dist/dmg-staging"

echo "▶  Building DMG for arch=${ARCH}  app=${APP_PATH}"

if [[ ! -d "${APP_PATH}" ]]; then
    echo "ERROR: ${APP_PATH} not found. Run PyInstaller first."
    exit 1
fi

rm -rf "${STAGING}" && mkdir -p "${STAGING}"
cp -r "${APP_PATH}" "${STAGING}/"

# Sign the .app if CODESIGN_IDENTITY is set in the environment
if [[ -n "${CODESIGN_IDENTITY:-}" ]]; then
    echo "▶  Code-signing with identity: ${CODESIGN_IDENTITY}"
    codesign --deep --force --options runtime \
             --sign "${CODESIGN_IDENTITY}" \
             --entitlements installer/mac/entitlements.plist \
             "${STAGING}/${APP_NAME}.app"
fi

create-dmg \
    --volname "Pollen Analysis Tool" \
    --volicon "assets/icon.icns" \
    --window-pos 200 120 \
    --window-size 660 400 \
    --icon-size 128 \
    --icon "${APP_NAME}.app" 160 185 \
    --hide-extension "${APP_NAME}.app" \
    --app-drop-link 500 185 \
    --background "assets/dmg_background.png" \
    "dist/${DMG_NAME}" \
    "${STAGING}/"

echo "✅  DMG created: dist/${DMG_NAME}"
