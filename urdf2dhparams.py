"""
urdf2dhparams.py

Extract Denavit-Hartenberg parameters (both classic/distal and modified/Craig
conventions) from a robot URDF saved by download_robot.py, and write them to a
CSV in the same folder as the URDF.

Behaviour (as configured):
  * Auto-detects the main actuated kinematic chain (root -> deepest link with the
    most actuated joints).
  * Folds the fixed base mount and end-effector offset into a base transform and a
    tool transform, so the DH model reproduces the URDF forward kinematics exactly.
  * Angles are written in DEGREES, lengths in METRES (URDF native units).
  * Self-checks the result by reconstructing forward kinematics from the DH table
    at many random joint configurations and comparing against the URDF.

Usage:
    python urdf2dhparams.py                 # prompts for the description name
    python urdf2dhparams.py xarm7_description
    python urdf2dhparams.py path/to/robot.urdf

Requires: numpy
"""

import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ACTUATED = {"revolute", "continuous", "prismatic"}
PROJECT_ROOT = Path(__file__).resolve().parent
ROBOTS_DIR = PROJECT_ROOT / "Robots"


# --------------------------------------------------------------------------- #
# Small SE(3) helpers
# --------------------------------------------------------------------------- #
def rpy_to_R(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def origin_T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = rpy_to_R(*rpy)
    T[:3, 3] = xyz
    return T


def inv(T):
    R = T[:3, :3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ T[:3, 3]
    return Ti


def frame_T(o, x, z):
    x = x / np.linalg.norm(x)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, o
    return T


# --------------------------------------------------------------------------- #
# URDF parsing and chain handling
# --------------------------------------------------------------------------- #
def parse_urdf(path):
    root = ET.parse(path).getroot()
    joints, children = {}, set()
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz, rpy = np.zeros(3), np.zeros(3)
        if o is not None:
            if o.get("xyz"):
                xyz = np.array([float(v) for v in o.get("xyz").split()])
            if o.get("rpy"):
                rpy = np.array([float(v) for v in o.get("rpy").split()])
        ax = j.find("axis")
        axis = (np.array([float(v) for v in ax.get("xyz").split()])
                if ax is not None else np.array([1.0, 0.0, 0.0]))
        joints[j.get("name")] = dict(
            name=j.get("name"), type=j.get("type"),
            parent=j.find("parent").get("link"), child=j.find("child").get("link"),
            xyz=xyz, rpy=rpy, axis=axis)
        children.add(j.find("child").get("link"))
    links = [l.get("name") for l in root.findall("link")]
    base = next(l for l in links if l not in children)
    return joints, base


def auto_chain(joints, base):
    """Return the ordered joint list of the chain with the most actuated joints."""
    by_parent = {}
    for j in joints.values():
        by_parent.setdefault(j["parent"], []).append(j)
    best = [None]

    def walk(link, path):
        kids = by_parent.get(link, [])
        if not kids:
            key = (sum(1 for j in path if j["type"] in ACTUATED), len(path))
            if best[0] is None or key > best[0][0]:
                best[0] = (key, list(path))
            return
        for j in kids:
            walk(j["child"], path + [j])

    walk(base, [])
    return best[0][1]


def chain_axes_and_tip(chain):
    """Actuated joint axis lines (point, dir) in the base frame at q=0, and base->tip."""
    T = np.eye(4)
    axes = []
    for j in chain:
        T = T @ origin_T(j["xyz"], j["rpy"])
        if j["type"] in ACTUATED:
            zdir = T[:3, :3] @ (j["axis"] / np.linalg.norm(j["axis"]))
            axes.append(dict(point=T[:3, 3].copy(), dir=zdir.copy(), joint=j))
    return axes, T


# --------------------------------------------------------------------------- #
# DH frame assignment (common normal between consecutive joint axes)
# --------------------------------------------------------------------------- #
def common_normal(p1, z1, p2, z2, eps=1e-9):
    """Feet of the common perpendicular between two lines, plus its direction/length."""
    z1 = z1 / np.linalg.norm(z1)
    z2 = z2 / np.linalg.norm(z2)
    cx = np.cross(z1, z2)
    ncx = np.linalg.norm(cx)
    if ncx < eps:  # parallel axes
        w = p2 - p1
        perp = w - np.dot(w, z1) * z1
        a = np.linalg.norm(perp)
        xdir = perp / a if a > eps else np.zeros(3)
        return p1.copy(), p1 + perp, xdir, a, True
    w0 = p1 - p2
    b, d, e = np.dot(z1, z2), np.dot(z1, w0), np.dot(z2, w0)
    den = 1 - b * b
    foot1 = p1 + ((b * e - d) / den) * z1
    foot2 = p2 + ((e - b * d) / den) * z2
    sep = foot2 - foot1
    a = np.linalg.norm(sep)
    xdir = cx / ncx
    if a > eps and np.dot(sep, xdir) < 0:
        xdir = -xdir
    return foot1, foot2, xdir, a, False


def _fix_degenerate(x, z, neighbour):
    """Pick an x perpendicular to z when the common normal direction is undefined."""
    ref = neighbour if neighbour is not None else np.array([1.0, 0.0, 0.0])
    xd = ref - np.dot(ref, z) * z
    if np.linalg.norm(xd) < 1e-9:
        ref = np.array([0.0, 1.0, 0.0])
        xd = ref - np.dot(ref, z) * z
    return xd / np.linalg.norm(xd)


def assign_standard(axes):
    """Standard / distal DH frames 0..n: z_{i-1} is the axis of joint i."""
    n = len(axes)
    Z = [a["dir"] for a in axes]
    P = [a["point"] for a in axes]
    z = [Z[i] for i in range(n)] + [Z[n - 1]]
    o = [None] * (n + 1)
    x = [None] * (n + 1)
    for i in range(1, n):
        _, f2, xd, _, _ = common_normal(P[i - 1], Z[i - 1], P[i], Z[i])
        o[i] = f2
        x[i] = xd if np.linalg.norm(xd) > 1e-9 else None
    for i in range(1, n):
        if x[i] is None:
            x[i] = _fix_degenerate(None, z[i], x[i - 1])
    f1, _, _, _, _ = common_normal(P[0], Z[0], P[1], Z[1])
    o[0], x[0] = f1, x[1].copy()
    o[n], x[n] = P[n - 1].copy(), x[n - 1].copy()
    return [frame_T(o[i], x[i], z[i]) for i in range(n + 1)]


def assign_modified(axes):
    """Modified / Craig frames 0..n: z_i is the axis of joint i."""
    n = len(axes)
    Z = [a["dir"] for a in axes]
    P = [a["point"] for a in axes]
    z = [Z[0]] + [Z[i - 1] for i in range(1, n + 1)]
    o = [None] * (n + 1)
    x = [None] * (n + 1)
    for i in range(1, n):
        f1, _, xd, _, _ = common_normal(P[i - 1], Z[i - 1], P[i], Z[i])
        o[i] = f1
        x[i] = xd if np.linalg.norm(xd) > 1e-9 else None
    for i in range(1, n):
        if x[i] is None:
            nb = x[i + 1] if i + 1 < n else x[i - 1]
            x[i] = _fix_degenerate(None, z[i], nb)
    o[0] = o[1].copy() if o[1] is not None else P[0].copy()
    x[0] = x[1].copy()
    o[n], x[n] = P[n - 1].copy(), x[n - 1].copy()
    return [frame_T(o[i], x[i], z[i]) for i in range(n + 1)]


# --------------------------------------------------------------------------- #
# Parameter extraction + reconstruction
# --------------------------------------------------------------------------- #
def extract_standard(A):  # A = Rz(th) Tz(d) Tx(a) Rx(al)
    th = np.arctan2(A[1, 0], A[0, 0])
    al = np.arctan2(A[2, 1], A[2, 2])
    d = A[2, 3]
    a = A[0, 3] * np.cos(th) + A[1, 3] * np.sin(th)
    return th, d, a, al


def recon_standard(th, d, a, al):
    cz, sz, cx, sx = np.cos(th), np.sin(th), np.cos(al), np.sin(al)
    return np.array([[cz, -sz * cx, sz * sx, a * cz],
                     [sz, cz * cx, -cz * sx, a * sz],
                     [0, sx, cx, d], [0, 0, 0, 1]])


def extract_modified(M):  # M = Rx(al) Tx(a) Rz(th) Tz(d)
    th = np.arctan2(-M[0, 1], M[0, 0])
    al = np.arctan2(-M[1, 2], M[2, 2])
    a = M[0, 3]
    d = M[2, 3] * np.cos(al) - M[1, 3] * np.sin(al)
    return al, a, th, d


def recon_modified(al, a, th, d):
    cz, sz, cx, sx = np.cos(th), np.sin(th), np.cos(al), np.sin(al)
    return np.array([[cz, -sz, 0, a],
                     [sz * cx, cz * cx, -sx, -d * sx],
                     [sz * sx, cz * sx, cx, d * cx], [0, 0, 0, 1]])


# --------------------------------------------------------------------------- #
# Forward kinematics (for validation)
# --------------------------------------------------------------------------- #
def urdf_fk(chain, q):
    T = np.eye(4)
    for j in chain:
        T = T @ origin_T(j["xyz"], j["rpy"])
        if j["type"] in ACTUATED:
            qi = q.get(j["name"], 0.0)
            ax = j["axis"] / np.linalg.norm(j["axis"])
            M = np.eye(4)
            if j["type"] == "prismatic":
                M[:3, 3] = ax * qi
            else:
                K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
                M[:3, :3] = np.eye(3) + np.sin(qi) * K + (1 - np.cos(qi)) * K @ K
            T = T @ M
    return T


def dh_fk(Tbase, rows, Ttool, kinds, names, q, modified):
    T = Tbase.copy()
    for row, kind, nm in zip(rows, kinds, names):
        qi = q.get(nm, 0.0)
        if modified:
            al, a, th, d = row
        else:
            th, d, a, al = row
        if kind == "prismatic":
            d = d + qi
        else:
            th = th + qi
        T = T @ (recon_modified(al, a, th, d) if modified else recon_standard(th, d, a, al))
    return T @ Ttool


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def resolve_urdf(arg):
    """Accept either a .urdf path or a description name (looked up in Robots/<name>/)."""
    p = Path(arg)
    if p.suffix == ".urdf" and p.exists():
        return p
    folder = ROBOTS_DIR / arg
    if not folder.is_dir():
        raise FileNotFoundError(f"No folder {folder}. Run download_robot.py for '{arg}' first.")
    urdfs = sorted(folder.rglob("*.urdf"))
    if not urdfs:
        raise FileNotFoundError(f"No .urdf file found under {folder}.")
    preferred = folder / f"{arg}.urdf"
    return preferred if preferred in urdfs else urdfs[0]


def compute_dh(urdf_path):
    joints, base = parse_urdf(urdf_path)
    chain = auto_chain(joints, base)
    axes, T_tip = chain_axes_and_tip(chain)
    n = len(axes)
    if n == 0:
        raise ValueError("No actuated joints found in the detected chain.")
    names = [a["joint"]["name"] for a in axes]
    kinds = [a["joint"]["type"] for a in axes]

    Fs = assign_standard(axes)
    std = [extract_standard(inv(Fs[i - 1]) @ Fs[i]) for i in range(1, n + 1)]
    base_std, tool_std = Fs[0], inv(Fs[n]) @ T_tip

    Fm = assign_modified(axes)
    mod = [extract_modified(inv(Fm[i - 1]) @ Fm[i]) for i in range(1, n + 1)]
    base_mod, tool_mod = Fm[0], inv(Fm[n]) @ T_tip

    # Validate both conventions against URDF FK at random configurations.
    rng = np.random.default_rng(0)
    err_std = err_mod = 0.0
    for _ in range(300):
        q = {nm: rng.uniform(-3.0, 3.0) for nm in names}
        Tu = urdf_fk(chain, q)
        err_std = max(err_std, np.abs(dh_fk(base_std, std, tool_std, kinds, names, q, False) - Tu).max())
        err_mod = max(err_mod, np.abs(dh_fk(base_mod, mod, tool_mod, kinds, names, q, True) - Tu).max())

    return dict(names=names, kinds=kinds, std=std, mod=mod,
                base_std=base_std, tool_std=tool_std,
                base_mod=base_mod, tool_mod=tool_mod,
                err_std=err_std, err_mod=err_mod, base_link=base)


def write_csv(out_path, robot_name, urdf_path, dh):
    deg = np.degrees

    def flat(T):
        return ",".join(f"{v:.9g}" for v in T.reshape(-1))

    with open(out_path, "w", newline="") as f:
        f.write(f"# DH parameters for {robot_name}\n")
        f.write(f"# source URDF: {urdf_path}\n")
        f.write("# lengths in metres, angles in degrees\n")
        f.write("# 'variable' = parameter driven by the joint (offset value shown at q=0)\n")
        f.write("# classic (distal):  A_i = Rz(theta) Tz(d) Tx(a) Rx(alpha)\n")
        f.write("# modified (Craig):  A_i = Rx(alpha) Tx(a) Rz(theta) Tz(d)\n")
        f.write(f"# FK self-check vs URDF (max abs error over 300 random configs): "
                f"classic={dh['err_std']:.2e}, modified={dh['err_mod']:.2e}\n")
        f.write(f"# base_transform (4x4 row-major, classic): {flat(dh['base_std'])}\n")
        f.write(f"# tool_transform (4x4 row-major, classic): {flat(dh['tool_std'])}\n")
        f.write(f"# base_transform (4x4 row-major, modified): {flat(dh['base_mod'])}\n")
        f.write(f"# tool_transform (4x4 row-major, modified): {flat(dh['tool_mod'])}\n")

        w = csv.writer(f)
        w.writerow(["joint", "type", "variable",
                    "classic_theta_deg", "classic_d_m", "classic_a_m", "classic_alpha_deg",
                    "modified_alpha_deg", "modified_a_m", "modified_d_m", "modified_theta_deg"])
        for i in range(len(dh["names"])):
            th, d, a, al = dh["std"][i]
            mal, ma, mth, md = dh["mod"][i]
            var = "d" if dh["kinds"][i] == "prismatic" else "theta"
            w.writerow([dh["names"][i], dh["kinds"][i], var,
                        f"{deg(th):.6f}", f"{d:.6f}", f"{a:.6f}", f"{deg(al):.6f}",
                        f"{deg(mal):.6f}", f"{ma:.6f}", f"{md:.6f}", f"{deg(mth):.6f}"])


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else input(
        "Description name (e.g. xarm7_description) or path to a .urdf: ").strip()
    urdf_path = resolve_urdf(arg)
    robot_name = urdf_path.parent.name
    print(f"Reading URDF: {urdf_path}")

    dh = compute_dh(urdf_path)
    print(f"Detected chain: {len(dh['names'])} actuated joints from base '{dh['base_link']}'")
    print(f"  {', '.join(dh['names'])}")
    print(f"FK self-check (max error vs URDF): classic={dh['err_std']:.2e}, "
          f"modified={dh['err_mod']:.2e}")

    out_path = urdf_path.parent / f"{robot_name}_dh_params.csv"
    write_csv(out_path, robot_name, urdf_path, dh)
    print(f"Saved DH parameters to: {out_path}")


if __name__ == "__main__":
    main()
