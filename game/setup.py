from cx_Freeze import setup, Executable

import sys


build_options = {
    "packages": ["pygame"], 
    "excludes": ["tkinter"], 
    "include_files": ["./game/pao.ico"],
    "build_exe": "./game/build",
}

base = "Win32GUI" if sys.platform == "win32" else None

executables = [
    Executable("./game/pits_and_orbs.py", base=base, target_name="Pits and Orbs", icon="./game/pao.ico")
]

msi_data = {
    "Icon": [
        ("IconId", "./game/pao.ico"),
    ],
    "Shortcut": [
        ("DesktopShortcut", "DesktopFolder", "Pits and Orbs", "TARGETDIR", "[TARGETDIR]Pits and Orbs.exe", None, None, None, None, None, None, "TARGETDIR"),
        ("StartMenuShortcut", "StartMenuFolder", "Pits and Orbs", "TARGETDIR", "[TARGETDIR]Pits and Orbs.exe", None, None, None, None, None, None, "TARGETDIR")
    ],
}

bdist_msi_options = {
    "data": msi_data,
    "initial_target_dir": r"[ProgramFilesFolder]\Pits and Orbs",
    "install_icon": "./game/pao.ico",
}

setup(
    name="Pits and Orbs",
    version="0.2.0",
    description="A simple game (environment) written in PyGame from scratch by EchineF to demonstrate a multi-agent system.",
    author_email="hosein.fanai@gmail.com",
    download_url="https://github.com/hosein-fanai/Pits-and-Orbs/game/Pits%20and%20Orbs-0.2.0-win64.msi",
    options = {
        "build_exe": build_options,
        "bdist_msi": bdist_msi_options,
    },
    executables=executables
)