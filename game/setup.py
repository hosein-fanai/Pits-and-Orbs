from cx_Freeze import setup, Executable

import sys


build_options = {"packages": ["pygame"], "excludes": ["tkinter"], "include_files": ["./game/pao.ico"]}

base = "Win32GUI" if sys.platform == "win32" else None

executables = [
    Executable("pits_and_orbs.py", base=base, target_name="Pits and Orbs", icon="pao.ico")
]

msi_data = {
    "Icon": [
        ("IconId", "pao.ico"),
    ],
    "Shortcut": [
        ("DesktopShortcut", "DesktopFolder", "Pits and Orbs", "TARGETDIR", "[TARGETDIR]Pits and Orbs.exe", None, None, None, None, None, None, "TARGETDIR"),
        ("StartMenuShortcut", "StartMenuFolder", "Pits and Orbs", "TARGETDIR", "[TARGETDIR]Pits and Orbs.exe", None, None, None, None, None, None, "TARGETDIR")
    ],
}

bdist_msi_options = {
    "data": msi_data,
    "initial_target_dir": r"[ProgramFilesFolder]\Pits and Orbs",
}

setup(
    name="Pits and Orbs",
    version="0.1.0",
    description="A simple game (environment) written in PyGame from scratch by EchineF to demonstrate a multi-agent system.",
    author_email="hosein.fanai@gmail.com",
    download_url="https://github.com/hosein-fanai/Pits-and-Orbs/game/dist/Pits and Orbs-0.1.0-win64.msi",
    options = {
        "build_exe": build_options,
        "bdist_msi": bdist_msi_options,
    },
    executables=executables
)