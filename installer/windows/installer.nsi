; ─────────────────────────────────────────────────────────────────────────────
; Pollen Analysis Tool — NSIS Windows Installer Script
; Build:  makensis installer\windows\installer.nsi
; Requires NSIS ≥ 3.08  +  the 'dist\PollenAnalysisTool' folder from PyInstaller
; ─────────────────────────────────────────────────────────────────────────────

!define APP_NAME      "Pollen Analysis Tool"
!define APP_VERSION   "1.0.0"
!define APP_PUBLISHER "Riha Lab"
!define APP_URL       "https://github.com/Riha-Lab/Pollen-Analysis-Tool"
!define APP_EXE       "PollenAnalysisTool.exe"
!define INSTALL_DIR   "$PROGRAMFILES64\PollenAnalysisTool"
!define REG_KEY       "Software\Microsoft\Windows\CurrentVersion\Uninstall\PollenAnalysisTool"

Name           "${APP_NAME} ${APP_VERSION}"
OutFile        "PollenAnalysisTool-Setup-Windows-x64.exe"
InstallDir     "${INSTALL_DIR}"
RequestExecutionLevel admin
SetCompressor  /SOLID lzma

; FIX 1: Set estimated install size so Add/Remove Programs shows it correctly.
; Update this number (in KB) if the installed size changes significantly.
!define ESTIMATED_SIZE 2500000   ; ~2.5 GB for PyTorch + models

; Modern UI
!include "MUI2.nsh"
!define MUI_ABORTWARNING
!define MUI_ICON   "..\..\assets\icon.ico"
!define MUI_UNICON "..\..\assets\icon.ico"

; FIX 2: Add a "first launch" info page so users know models will download.
; This is the plain-text version — replace with a bitmap if you have one.
!define MUI_WELCOMEPAGE_TEXT "Welcome to the Pollen Analysis Tool installer.$\r$\n$\r$\nOn first launch the app will download AI model weights (~800 MB total). This only happens once — subsequent starts are instant.$\r$\n$\r$\nPlease ensure you have an internet connection ready."

; Pages
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "..\..\LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

; ── Install ──────────────────────────────────────────────────────────────────
Section "Install"
  SetOutPath "$INSTDIR"
  File /r "..\..\dist\PollenAnalysisTool\*"

  ; Desktop shortcut
  CreateShortcut "$DESKTOP\Pollen Analysis Tool.lnk" \
                 "$INSTDIR\${APP_EXE}" "" \
                 "$INSTDIR\${APP_EXE}" 0

  ; Start-menu shortcut
  CreateDirectory "$SMPROGRAMS\Riha Lab"
  CreateShortcut  "$SMPROGRAMS\Riha Lab\Pollen Analysis Tool.lnk" \
                  "$INSTDIR\${APP_EXE}" "" \
                  "$INSTDIR\${APP_EXE}" 0

  ; Registry — Add/Remove Programs
  WriteRegStr   HKLM "${REG_KEY}" "DisplayName"      "${APP_NAME}"
  WriteRegStr   HKLM "${REG_KEY}" "DisplayVersion"   "${APP_VERSION}"
  WriteRegStr   HKLM "${REG_KEY}" "Publisher"        "${APP_PUBLISHER}"
  WriteRegStr   HKLM "${REG_KEY}" "URLInfoAbout"     "${APP_URL}"
  WriteRegStr   HKLM "${REG_KEY}" "InstallLocation"  "$INSTDIR"
  WriteRegStr   HKLM "${REG_KEY}" "UninstallString"  "$INSTDIR\Uninstall.exe"
  WriteRegDWORD HKLM "${REG_KEY}" "NoModify"         1
  WriteRegDWORD HKLM "${REG_KEY}" "NoRepair"         1
  ; FIX 3: Write estimated size so Windows shows it in Add/Remove Programs
  WriteRegDWORD HKLM "${REG_KEY}" "EstimatedSize"    "${ESTIMATED_SIZE}"

  WriteUninstaller "$INSTDIR\Uninstall.exe"
SectionEnd

; ── Uninstall ────────────────────────────────────────────────────────────────
Section "Uninstall"
  ; FIX 4: Also clean up the model weight cache the app writes on first run.
  ; This is optional — comment out if you want to preserve cached models.
  RMDir /r "$LOCALAPPDATA\.cache\pollen_analysis_tool"

  RMDir /r "$INSTDIR"
  Delete   "$DESKTOP\Pollen Analysis Tool.lnk"
  RMDir /r "$SMPROGRAMS\Riha Lab"
  DeleteRegKey HKLM "${REG_KEY}"
SectionEnd
