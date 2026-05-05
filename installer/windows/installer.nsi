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

; Modern UI
!include "MUI2.nsh"
!define MUI_ABORTWARNING
!define MUI_ICON   "..\..\assets\icon.ico"
!define MUI_UNICON "..\..\assets\icon.ico"

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

  WriteUninstaller "$INSTDIR\Uninstall.exe"
SectionEnd

; ── Uninstall ────────────────────────────────────────────────────────────────
Section "Uninstall"
  RMDir /r "$INSTDIR"
  Delete   "$DESKTOP\Pollen Analysis Tool.lnk"
  RMDir /r "$SMPROGRAMS\Riha Lab"
  DeleteRegKey HKLM "${REG_KEY}"
SectionEnd
