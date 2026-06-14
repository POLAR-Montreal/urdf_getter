"""
List the robot arms available in `robot_descriptions`, ask which one you want,
then save that robot's URDF + meshes into a self-contained folder:

    Robots/<description_name>/
        <description_name>.urdf   (or the original .urdf for non-xacro robots)
        meshes/ ...

Some robots (xArm, most URs, Franka, ...) ship only Xacro source and have their
URDF generated on the fly. For those you also need xacrodoc:

    pip install robot_descriptions xacrodoc

Run:  python download_robot.py
"""

import importlib
import os
import re
import shutil
from pathlib import Path

from robot_descriptions._descriptions import DESCRIPTIONS
from robot_descriptions._xacro import get_urdf_path  # returns URDF, rendering Xacro if needed

PROJECT_ROOT = Path(__file__).resolve().parent
ROBOTS_DIR = PROJECT_ROOT / "Robots"


def get_arms():
    """Return {description_name: Description} for every arm that has a URDF."""
    return {
        name: d
        for name, d in sorted(DESCRIPTIONS.items())
        if "arm" in d.tags and d.has_urdf
    }


def print_arms(arms):
    print(f"\nAvailable arm descriptions ({len(arms)}):\n")
    for name, d in arms.items():
        print(f"  {name:<32} {d.robot}  ({d.maker})")
    print()


def _make_relative_to_package(urdf_text, package_path):
    """Rewrite absolute mesh/material paths (that live under the package) into
    paths relative to the package root, so the URDF works next to copied meshes."""
    package_path = os.path.abspath(package_path)

    def repl(match):
        raw = match.group(1)
        path = raw[len("file://"):] if raw.startswith("file://") else raw
        if os.path.isabs(path):
            abs_path = os.path.abspath(path)
            if abs_path.startswith(package_path + os.sep):
                rel = os.path.relpath(abs_path, package_path)
                return f'filename="{Path(rel).as_posix()}"'
        return match.group(0)  # leave package:// and already-relative paths untouched

    return re.sub(r'filename="([^"]+)"', repl, urdf_text)


def save_robot(name):
    """Download (if needed) and save the robot into Robots/<name>/ as a
    self-contained URDF + meshes. Returns (dest_dir, urdf_file)."""
    module = importlib.import_module(f"robot_descriptions.{name}")
    package_path = Path(module.PACKAGE_PATH)
    urdf_src = Path(get_urdf_path(module))  # static URDF, or freshly rendered from xacro

    dest = ROBOTS_DIR / name
    shutil.copytree(package_path, dest, dirs_exist_ok=True)  # brings the meshes along

    if str(urdf_src).startswith(str(package_path)):
        # Robot ships a real .urdf inside its package: it's already copied,
        # with mesh references intact. Just point at it.
        urdf_file = dest / os.path.relpath(urdf_src, package_path)
    else:
        # URDF was rendered from xacro into a separate cache. Copy it in and
        # make its (absolute) mesh paths relative to the copied package.
        urdf_text = _make_relative_to_package(urdf_src.read_text(), package_path)
        urdf_file = dest / f"{name}.urdf"
        urdf_file.write_text(urdf_text)

    return dest, urdf_file


def main():
    arms = get_arms()
    print_arms(arms)

    while True:
        name = input("Enter the exact description name (or 'q' to quit): ").strip()
        if name.lower() in ("q", "quit", "exit"):
            print("Aborted.")
            return
        if name in arms:
            break
        print(f"  '{name}' is not in the list. Copy a name exactly as shown above.\n")

    print(f"\nFetching '{name}' (downloads/renders on first use, then cached)...")
    dest, urdf = save_robot(name)
    print(f"\nSaved to:  {dest}")
    print(f"URDF file: {urdf}")


if __name__ == "__main__":
    main()