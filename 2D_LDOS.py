"""

2D-LDOS

LDOS multilayer GUI calculator — v38

Version 38 adds interactive plot-properties dialogs (axis limits/scales and 2D-map color limits/normalization) while retaining the 2D sweep maps introduced in v36.

Standalone Tkinter application for building an arbitrary planar multilayer stack,
visualizing the stack and dipole position, calculating LDOS angular spectra, saving/loading
configurations, running energy-dependent LDOS-integral sweeps, and sweeping a selected layer thickness.

Material files can be CSV/TXT/DAT with flexible columns. Use Browse + map columns to assign energy/wavelength, n, and kappa columns.
"""
from __future__ import annotations

import csv
import cmath
import json
import os
import traceback
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Polygon
from matplotlib.colors import Normalize, LogNorm

HBAR_EV_S = 6.582119569e-16
C0 = 2.99792458e8


def safe_cexp(z: complex, clip: float = 80.0) -> complex:
    """Complex exponential with clipped real exponent to avoid overflow.

    This is a numerical stabilizer for evanescent waves in thick layers.
    It preserves the phase and clips only the magnitude exp(Re(z)).
    clip=80 limits magnitudes to about 5e34, large enough for relative
    amplitudes but small enough to avoid floating-point overflow/NaNs.
    """
    z = complex(z)
    re = max(-clip, min(clip, z.real))
    return complex(np.exp(re) * np.cos(z.imag), np.exp(re) * np.sin(z.imag))




@dataclass
class Material:
    mode: str = "n"  # "n", "eps", or "file"
    n_value: complex = 1.0 + 0.0j
    eps_value: complex = 1.0 + 0.0j
    file_path: str = ""
    file_x_kind: str = "energy_ev"
    file_data_kind: str = "nk"  # "nk" or "epsilon"
    file_x_col: int = 0
    file_n_col: int = 1
    file_k_col: int = 2
    file_eps_re_col: int = 1
    file_eps_im_col: int = 2
    file_has_header: bool = True
    file_delimiter: str = "auto"
    _file_cache: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default=None, init=False, repr=False)

    def epsilon(self, energy: float) -> complex:
        if self.mode == "n":
            return self.n_value ** 2
        if self.mode == "eps":
            return self.eps_value
        if self.mode == "file":
            if not self.file_path:
                raise ValueError("File material selected but no file path provided")
            E, n, k = self._load_file()
            idx = int(np.argmin(np.abs(E - energy)))
            return complex(n[idx], k[idx]) ** 2
        raise ValueError(f"Unknown material mode: {self.mode}")

    def _load_file(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._file_cache is not None:
            return self._file_cache

        def split_line(line: str):
            line = line.strip()
            if not line:
                return []
            delim = self.file_delimiter
            if delim == "comma":
                return [x.strip() for x in line.split(",")]
            if delim == "tab":
                return [x.strip() for x in line.split("\t")]
            if delim == "space":
                return line.split()
            return [x.strip() for x in line.split(",")] if "," in line else line.split()

        rows = []
        data_kind = getattr(self, "file_data_kind", "nk")
        if data_kind == "epsilon":
            max_col = max(int(self.file_x_col), int(self.file_eps_re_col), int(self.file_eps_im_col))
        else:
            max_col = max(int(self.file_x_col), int(self.file_n_col), int(self.file_k_col))

        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = split_line(line)
                if len(parts) <= max_col:
                    continue
                try:
                    x = float(parts[int(self.file_x_col)])
                    if data_kind == "epsilon":
                        eps_re = float(parts[int(self.file_eps_re_col)])
                        eps_im = float(parts[int(self.file_eps_im_col)])
                        n_complex = cmath.sqrt(complex(eps_re, eps_im))
                        n = float(np.real(n_complex))
                        k = float(np.imag(n_complex))
                        if k < 0:
                            n = -n
                            k = -k
                    else:
                        n = float(parts[int(self.file_n_col)])
                        k = float(parts[int(self.file_k_col)])
                except ValueError:
                    continue

                if self.file_x_kind == "energy_ev":
                    energy_ev = x
                elif self.file_x_kind == "wavelength_nm":
                    energy_ev = 1239.8419843320026 / x
                elif self.file_x_kind == "wavelength_um":
                    energy_ev = 1.2398419843320026 / x
                elif self.file_x_kind == "wavelength_m":
                    energy_ev = 1239.8419843320026 / (x * 1e9)
                else:
                    raise ValueError(f"Unknown material x-axis kind: {self.file_x_kind}")
                rows.append((energy_ev, n, k))

        if not rows:
            raise ValueError(f"Could not read mapped material data from {self.file_path}. Check column mapping, delimiter, and units.")
        arr = np.asarray(rows, dtype=float)
        arr = arr[np.argsort(arr[:, 0])]
        self._file_cache = (arr[:, 0], arr[:, 1], arr[:, 2])
        return self._file_cache

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "n_real": float(self.n_value.real),
            "n_imag": float(self.n_value.imag),
            "eps_real": float(self.eps_value.real),
            "eps_imag": float(self.eps_value.imag),
            "file_path": self.file_path,
            "file_x_kind": self.file_x_kind,
            "file_data_kind": self.file_data_kind,
            "file_x_col": int(self.file_x_col),
            "file_n_col": int(self.file_n_col),
            "file_k_col": int(self.file_k_col),
            "file_eps_re_col": int(self.file_eps_re_col),
            "file_eps_im_col": int(self.file_eps_im_col),
            "file_has_header": bool(self.file_has_header),
            "file_delimiter": self.file_delimiter,
        }

    @staticmethod
    def from_dict(d: dict) -> "Material":
        return Material(
            mode=d.get("mode", "n"),
            n_value=complex(d.get("n_real", 1.0), d.get("n_imag", 0.0)),
            eps_value=complex(d.get("eps_real", 1.0), d.get("eps_imag", 0.0)),
            file_path=d.get("file_path", ""),
            file_x_kind=d.get("file_x_kind", "energy_ev"),
            file_data_kind=d.get("file_data_kind", "nk"),
            file_x_col=int(d.get("file_x_col", 0)),
            file_n_col=int(d.get("file_n_col", 1)),
            file_k_col=int(d.get("file_k_col", 2)),
            file_eps_re_col=int(d.get("file_eps_re_col", d.get("file_n_col", 1))),
            file_eps_im_col=int(d.get("file_eps_im_col", d.get("file_k_col", 2))),
            file_has_header=bool(d.get("file_has_header", True)),
            file_delimiter=d.get("file_delimiter", "auto"),
        )


@dataclass
class Layer:
    name: str
    thickness_m: float
    material: Material

    def epsilon(self, energy: float) -> complex:
        return self.material.epsilon(energy)

    def to_dict(self) -> dict:
        return {"name": self.name, "thickness_m": None if not np.isfinite(self.thickness_m) else float(self.thickness_m), "material": self.material.to_dict()}

    @staticmethod
    def from_dict(d: dict) -> "Layer":
        thickness = np.inf if d.get("thickness_m") is None else float(d["thickness_m"])
        return Layer(d.get("name", "layer"), thickness, Material.from_dict(d.get("material", {})))


def kz_for_layer(k0: float, eps: complex, s: float, eps_ref: complex) -> complex:
    val = cmath.sqrt(eps - eps_ref * s * s)
    if val.imag < 0:
        val = -val
    return k0 * val


def fresnel(kza: complex, kzb: complex, eps_a: complex, eps_b: complex) -> tuple[complex, complex]:
    rs = (kza - kzb) / (kza + kzb)
    rp = (eps_b * kza - eps_a * kzb) / (eps_b * kza + eps_a * kzb)
    return rs, rp


def generalized_reflection(eps_layers: Sequence[complex], d_layers: Sequence[float], k0: float, s: float, eps_ref: complex) -> tuple[complex, complex]:
    n = len(eps_layers)
    if n < 2:
        return 0.0 + 0.0j, 0.0 + 0.0j
    kz = [kz_for_layer(k0, eps, s, eps_ref) for eps in eps_layers]
    rs_next, rp_next = fresnel(kz[-2], kz[-1], eps_layers[-2], eps_layers[-1])
    for j in range(n - 3, -1, -1):
        rs_j, rp_j = fresnel(kz[j], kz[j + 1], eps_layers[j], eps_layers[j + 1])
        d = d_layers[j + 1]
        phase = safe_cexp(2j * kz[j + 1] * d) if np.isfinite(d) else 0.0
        rs_next = (rs_j + rs_next * phase) / (1.0 + rs_j * rs_next * phase)
        rp_next = (rp_j + rp_next * phase) / (1.0 + rp_j * rp_next * phase)
    return rs_next, rp_next


def calculate_ldos(layers: Sequence[Layer], dipole_layer_index: int, dipole_fraction_from_bottom: float, energy_ev: float, s_values: np.ndarray) -> dict[str, np.ndarray | float]:
    if len(layers) < 3:
        raise ValueError("Use at least three layers: top half-space, dipole layer/finite layers, bottom half-space.")
    if not 0 <= dipole_layer_index < len(layers):
        raise ValueError("Dipole layer index out of range")
    dipole_layer = layers[dipole_layer_index]
    d1 = dipole_layer.thickness_m
    if not np.isfinite(d1) or d1 <= 0:
        raise ValueError("Dipole layer must have a positive finite thickness")
    if not 0.0 <= dipole_fraction_from_bottom <= 1.0:
        raise ValueError("Dipole fraction must be between 0 and 1")

    eps_all = [layer.epsilon(energy_ev) for layer in layers]
    d_all = [layer.thickness_m for layer in layers]
    eps1 = eps_all[dipole_layer_index]
    k0 = energy_ev / (HBAR_EV_S * C0)
    k1 = k0 * cmath.sqrt(eps1)

    z_from_bottom = dipole_fraction_from_bottom * d1
    z0_d = -z_from_bottom
    z0_u = d1 - z_from_bottom

    eps_down = eps_all[dipole_layer_index:]
    d_down = d_all[dipole_layer_index:]
    eps_up = [eps_all[dipole_layer_index]] + list(reversed(eps_all[:dipole_layer_index]))
    d_up = [d_all[dipole_layer_index]] + list(reversed(d_all[:dipole_layer_index]))

    dpds_hor, dpds_perp = [], []
    Pfree = (energy_ev / HBAR_EV_S) ** 4 / (3 * C0 ** 3)

    
    # User-facing s_values are vacuum-normalized s_vac = k_parallel/k0.
    # The original formulas in this code used an internal variable normalized
    # to the dipole medium, s_layer = k_parallel/k1 = s_vac/sqrt(eps1).
    # Use s_layer internally so the LDOS formula is unchanged, but return and
    # plot the user-facing vacuum s. Then the vacuum light cone is at s=1.
    
    n1_eff = cmath.sqrt(eps1)
    for s_vac in s_values:
        s_vac = float(s_vac)
        s = s_vac / n1_eff
        q1 = kz_for_layer(k0, eps1, s, eps1)
        p = s_vac * k0
        rs_d, rp_d = generalized_reflection(eps_down, d_down, k0, s, eps1)
        rs_u, rp_u = generalized_reflection(eps_up, d_up, k0, s, eps1)
        exp_d = safe_cexp(-2j * q1 * z0_d)
        exp_u = safe_cexp(2j * q1 * z0_u)
        exp_cavity = safe_cexp(2j * q1 * d1)
        term1 = (1 + rp_d * exp_d) * (1 + rp_u * exp_u) / (1 - rp_u * rp_d * exp_cavity)
        term2 = (1 + rs_d * exp_d) * (1 + rs_u * exp_u) / (1 - rs_u * rs_d * exp_cavity)
        term3 = (1 - rp_d * exp_d) * (1 - rp_u * exp_u) / (1 - rp_u * rp_d * exp_cavity)
        hor = k0 / Pfree * C0 / (2 * eps1) * k0 * np.real(0.5 * p / q1 * k1 ** 2 * term2 + p / q1 * q1 ** 2 * 0.5 * term3)
        perp = k0 / Pfree * C0 / (2 * eps1) * k0 * np.real(p ** 3 / q1 * term1)
        dpds_hor.append(float(np.real(hor)))
        dpds_perp.append(float(np.real(perp)))

    dpds_hor = np.asarray(dpds_hor)
    dpds_perp = np.asarray(dpds_perp)
    return {"s": s_values, "s_definition": "vacuum-normalized k_parallel/k0", "kpar": s_values * k0, "dpds_hor": dpds_hor, "dpds_perp": dpds_perp, "ldos_hor": float(np.trapezoid(dpds_hor, s_values)), "ldos_perp": float(np.trapezoid(dpds_perp, s_values)), "k0": float(k0)}


# -----------------------------------------------------------------------------
# v20 boundary-condition dipole field solver
# -----------------------------------------------------------------------------
# This block replaces the earlier experimental field-profile engine.  It solves
# fields directly from boundary conditions for each s.  LDOS is not used as an
# input to the field distribution.

@dataclass
class _BFieldSegment:
    name: str
    original_layer_index: int
    thickness_m: float
    eps: complex


# -----------------------------------------------------------------------------
# v23 photonic-style amplitude-source field solver
# -----------------------------------------------------------------------------
# This replaces the earlier derivative-jump scalar source with a solver closer to
# the user's photonic_sim logic: build the full interface matrix for the stack,
# inject known right/left source amplitudes in the dipole layer, solve all other
# right/left amplitudes simultaneously, then reconstruct fields.  LDOS is not
# used as an input to the field calculation.



def _v34_add_vacuum_light_cone_guide(ax, x_axis_choice="s"):
    """Add a guide for the vacuum light cone at s=1."""
    try:
        if str(x_axis_choice).lower() == "s":
            ax.axvline(1.0, linestyle=":", linewidth=1.0, alpha=0.7)
            yl = ax.get_ylim()
            ax.text(1.0, yl[1], "s=1 vacuum light cone", rotation=90,
                    va="top", ha="right", fontsize=8, alpha=0.7)
            ax.set_ylim(yl)
    except Exception:
        pass


class MaterialFileMapperDialog(tk.Toplevel):
    def __init__(self, parent, file_path: str, material: Optional[Material] = None):
        super().__init__(parent)
        self.title("Map material file columns")
        self.geometry("900x560")
        self.minsize(760, 460)
        self.result: Optional[dict] = None
        self.file_path = file_path
        material = material or Material(mode="file", file_path=file_path)

        self.delim_var = tk.StringVar(value=getattr(material, "file_delimiter", "auto"))
        self.xkind_var = tk.StringVar(value=getattr(material, "file_x_kind", "energy_ev"))
        self.data_kind_var = tk.StringVar(value=getattr(material, "file_data_kind", "nk"))
        self.xcol_var = tk.StringVar(value=str(getattr(material, "file_x_col", 0)))
        self.ncol_var = tk.StringVar(value=str(getattr(material, "file_n_col", 1)))
        self.kcol_var = tk.StringVar(value=str(getattr(material, "file_k_col", 2)))
        self.eps_re_col_var = tk.StringVar(value=str(getattr(material, "file_eps_re_col", getattr(material, "file_n_col", 1))))
        self.eps_im_col_var = tk.StringVar(value=str(getattr(material, "file_eps_im_col", getattr(material, "file_k_col", 2))))
        self.plot_view_var = tk.StringVar(value="nk")
        self.header_var = tk.BooleanVar(value=bool(getattr(material, "file_has_header", True)))

        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)
        ttk.Label(root, text=f"File: {file_path}", wraplength=850).pack(anchor="w", pady=(0, 6))

        controls = ttk.LabelFrame(root, text="Column mapping")
        controls.pack(fill=tk.X, pady=4)
        ttk.Label(controls, text="Delimiter").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(controls, textvariable=self.delim_var, values=["auto", "comma", "tab", "space"], width=10, state="readonly").grid(row=0, column=1, padx=4)
        ttk.Label(controls, text="Data are").grid(row=0, column=2, sticky="w", padx=4)
        ttk.Combobox(controls, textvariable=self.data_kind_var, values=["nk", "epsilon"], width=10, state="readonly").grid(row=0, column=3, padx=4)
        ttk.Label(controls, text="X column").grid(row=0, column=4, sticky="w", padx=4)
        ttk.Entry(controls, textvariable=self.xcol_var, width=6).grid(row=0, column=5, padx=4)
        ttk.Label(controls, text="X means").grid(row=0, column=6, sticky="w", padx=4)
        ttk.Combobox(controls, textvariable=self.xkind_var, values=["energy_ev", "wavelength_nm", "wavelength_um", "wavelength_m"], width=16, state="readonly").grid(row=0, column=7, padx=4)

        ttk.Label(controls, text="n / eps real col").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(controls, textvariable=self.ncol_var, width=6).grid(row=1, column=1, padx=4)
        ttk.Entry(controls, textvariable=self.eps_re_col_var, width=6).grid(row=1, column=2, padx=4)
        ttk.Label(controls, text="kappa / eps imag col").grid(row=1, column=3, sticky="w", padx=4)
        ttk.Entry(controls, textvariable=self.kcol_var, width=6).grid(row=1, column=4, padx=4)
        ttk.Entry(controls, textvariable=self.eps_im_col_var, width=6).grid(row=1, column=5, padx=4)
        ttk.Checkbutton(controls, text="first row is header/comment", variable=self.header_var).grid(row=1, column=6, columnspan=2, sticky="w", padx=4)

        ttk.Label(controls, text="Plot view").grid(row=2, column=0, sticky="w", padx=4, pady=(4, 4))
        ttk.Combobox(controls, textvariable=self.plot_view_var, values=["nk", "epsilon"], width=10, state="readonly").grid(row=2, column=1, padx=4, pady=(4, 4))

        action_btns = ttk.Frame(controls)
        action_btns.grid(row=3, column=0, columnspan=8, sticky="w", padx=4, pady=(6, 4))
        ttk.Button(action_btns, text="Refresh preview", command=self.refresh_preview).pack(side="left", padx=(0, 6))
        ttk.Button(action_btns, text="Plot", command=self.plot_mapped_data).pack(side="left", padx=(0, 6))
        ttk.Button(action_btns, text="Use this mapping", command=self.accept).pack(side="left", padx=(0, 6))
        ttk.Button(action_btns, text="Cancel", command=self.destroy).pack(side="left")

        preview_frame = ttk.LabelFrame(root, text="Preview: first readable rows")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=6)
        self.tree = ttk.Treeview(preview_frame, show="headings", height=14)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll = ttk.Scrollbar(preview_frame, orient="vertical", command=self.tree.yview)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=yscroll.set)
        self.status_var = tk.StringVar(value="")
        ttk.Label(root, textvariable=self.status_var, foreground="darkblue").pack(anchor="w")
        self.refresh_preview()
        self.grab_set()
        self.transient(parent)

    def split_line(self, line: str):
        line = line.strip()
        if not line:
            return []
        d = self.delim_var.get()
        if d == "comma":
            return [x.strip() for x in line.split(",")]
        if d == "tab":
            return [x.strip() for x in line.split("\t")]
        if d == "space":
            return line.split()
        return [x.strip() for x in line.split(",")] if "," in line else line.split()

    def read_preview_rows(self, max_rows=40):
        rows = []
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = self.split_line(line)
                if not parts:
                    continue
                rows.append(parts)
                if len(rows) >= max_rows:
                    break
        return rows

    def refresh_preview(self):
        try:
            rows = self.read_preview_rows()
            ncols = max((len(r) for r in rows), default=0)
            columns = [f"col {i}" for i in range(ncols)]
            self.tree.delete(*self.tree.get_children())
            self.tree["columns"] = columns
            for c in columns:
                self.tree.heading(c, text=c)
                self.tree.column(c, width=95, stretch=True)
            for r in rows[:25]:
                self.tree.insert("", "end", values=list(r) + [""] * (ncols - len(r)))
            self.status_var.set(self.try_mapped_preview(rows))
        except Exception as e:
            self.status_var.set(f"Preview error: {e}")

    def try_mapped_preview(self, rows):
        try:
            xcol = int(self.xcol_var.get())
            data_kind = self.data_kind_var.get()
            if data_kind == "epsilon":
                c1 = int(self.eps_re_col_var.get()); c2 = int(self.eps_im_col_var.get())
            else:
                c1 = int(self.ncol_var.get()); c2 = int(self.kcol_var.get())
            out = []
            for r in rows:
                try:
                    x = float(r[xcol]); v1 = float(r[c1]); v2 = float(r[c2])
                except Exception:
                    continue
                kind = self.xkind_var.get()
                if kind == "energy_ev":
                    E = x
                elif kind == "wavelength_nm":
                    E = 1239.8419843320026 / x
                elif kind == "wavelength_um":
                    E = 1.2398419843320026 / x
                elif kind == "wavelength_m":
                    E = 1239.8419843320026 / (x * 1e9)
                else:
                    E = np.nan
                out.append((E, v1, v2))
                if len(out) >= 3:
                    break
            if not out:
                return "No numeric mapped rows found yet. Adjust columns/delimiter/units."
            label = "eps1, eps2" if data_kind == "epsilon" else "n, k"
            return f"Mapped sample ({label}): " + "; ".join([f"E={E:.4g} eV, a={v1:.4g}, b={v2:.4g}" for E, v1, v2 in out])
        except Exception as e:
            return f"Mapping not valid yet: {e}"


    def mapped_arrays(self):
        xcol = int(self.xcol_var.get())
        data_kind = self.data_kind_var.get()
        if data_kind == "epsilon":
            c1 = int(self.eps_re_col_var.get())
            c2 = int(self.eps_im_col_var.get())
        else:
            c1 = int(self.ncol_var.get())
            c2 = int(self.kcol_var.get())
        kind = self.xkind_var.get()
        rows = []
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = self.split_line(line)
                if len(parts) <= max(xcol, c1, c2):
                    continue
                try:
                    x = float(parts[xcol])
                    a = float(parts[c1])
                    b = float(parts[c2])
                except Exception:
                    continue

                if kind == "energy_ev":
                    energy_ev = x
                    x_plot = x
                    x_label = "Energy (eV)"
                elif kind == "wavelength_nm":
                    energy_ev = 1239.8419843320026 / x
                    x_plot = x
                    x_label = "Wavelength (nm)"
                elif kind == "wavelength_um":
                    energy_ev = 1.2398419843320026 / x
                    x_plot = x
                    x_label = "Wavelength (µm)"
                elif kind == "wavelength_m":
                    energy_ev = 1239.8419843320026 / (x * 1e9)
                    x_plot = x * 1e9
                    x_label = "Wavelength (nm)"
                else:
                    raise ValueError(f"Unknown X kind: {kind}")

                if data_kind == "epsilon":
                    eps_complex = complex(a, b)
                    n_complex = cmath.sqrt(eps_complex)
                    n = float(np.real(n_complex))
                    k = float(np.imag(n_complex))
                    if k < 0:
                        n = -n
                        k = -k
                    eps_re = a
                    eps_im = b
                else:
                    n = a
                    k = b
                    eps_complex = complex(n, k) ** 2
                    eps_re = float(np.real(eps_complex))
                    eps_im = float(np.imag(eps_complex))

                rows.append((x_plot, energy_ev, n, k, eps_re, eps_im))
        if not rows:
            raise ValueError("No numeric rows found with the current mapping.")
        arr = np.asarray(rows, dtype=float)
        arr = arr[np.argsort(arr[:, 0])]
        return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5], x_label


    def plot_mapped_data(self):
        try:
            x_plot, energy_ev, n, k, eps_re, eps_im, x_label = self.mapped_arrays()
            win = tk.Toplevel(self)
            win.title("Material file plot preview")
            win.geometry("850x560")
            fig = Figure(figsize=(7.5, 4.8), dpi=100)
            ax = fig.add_subplot(111)
            if self.plot_view_var.get() == "epsilon":
                ax.plot(x_plot, eps_re, label="Re(epsilon)")
                ax.plot(x_plot, eps_im, label="Im(epsilon)")
                ax.set_ylabel("Dielectric function")
            else:
                ax.plot(x_plot, n, label="n")
                ax.plot(x_plot, k, label="kappa")
                ax.set_ylabel("Optical constants")
            ax.set_xlabel(x_label)
            ax.set_title("Mapped material data preview")
            ax.grid(True, alpha=0.3)
            ax.legend()

            emin = float(np.nanmin(energy_ev))
            emax = float(np.nanmax(energy_ev))
            ax.text(
                0.02, 0.98,
                f"Energy range: {emin:.4g}–{emax:.4g} eV\nRows: {len(x_plot)}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round", alpha=0.15),
            )

            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            toolbar = NavigationToolbar2Tk(canvas, win)
            toolbar.update()
            toolbar.pack(fill=tk.X)
            fig.tight_layout()
            canvas.draw_idle()
        except Exception as e:
            messagebox.showerror("Plot material data", str(e), parent=self)


    def accept(self):
        try:
            mapping = {
                "file_path": self.file_path,
                "file_delimiter": self.delim_var.get(),
                "file_x_kind": self.xkind_var.get(),
                "file_data_kind": self.data_kind_var.get(),
                "file_x_col": int(self.xcol_var.get()),
                "file_n_col": int(self.ncol_var.get()),
                "file_k_col": int(self.kcol_var.get()),
                "file_eps_re_col": int(self.eps_re_col_var.get()),
                "file_eps_im_col": int(self.eps_im_col_var.get()),
                "file_has_header": bool(self.header_var.get()),
            }
            Material(mode="file", **mapping)._load_file()
            self.result = mapping
            self.destroy()
        except Exception as e:
            messagebox.showerror("Invalid material mapping", str(e))


class LayerDialog(tk.Toplevel):
    def __init__(self, parent, layer: Optional[Layer] = None):
        super().__init__(parent)
        self.title("Layer")
        self.resizable(True, False)
        self.geometry("620x360")
        self.minsize(560, 330)
        self.result: Optional[Layer] = None
        self.name_var = tk.StringVar(value=layer.name if layer else "layer")
        self.thick_var = tk.StringVar(value=("inf" if layer and not np.isfinite(layer.thickness_m) else (str(layer.thickness_m / 1e-9) if layer else "1.0")))
        self.mode_var = tk.StringVar(value=layer.material.mode if layer else "n")
        self.n_re_var = tk.StringVar(value=str(layer.material.n_value.real) if layer else "1.0")
        self.n_im_var = tk.StringVar(value=str(layer.material.n_value.imag) if layer else "0.0")
        self.eps_re_var = tk.StringVar(value=str(layer.material.eps_value.real) if layer else "1.0")
        self.eps_im_var = tk.StringVar(value=str(layer.material.eps_value.imag) if layer else "0.0")
        self.file_var = tk.StringVar(value=layer.material.file_path if layer else "")
        self.file_x_kind_var = tk.StringVar(value=getattr(layer.material, "file_x_kind", "energy_ev") if layer else "energy_ev")
        self.file_data_kind_var = tk.StringVar(value=getattr(layer.material, "file_data_kind", "nk") if layer else "nk")
        self.file_x_col_var = tk.StringVar(value=str(getattr(layer.material, "file_x_col", 0)) if layer else "0")
        self.file_n_col_var = tk.StringVar(value=str(getattr(layer.material, "file_n_col", 1)) if layer else "1")
        self.file_k_col_var = tk.StringVar(value=str(getattr(layer.material, "file_k_col", 2)) if layer else "2")
        self.file_eps_re_col_var = tk.StringVar(value=str(getattr(layer.material, "file_eps_re_col", getattr(layer.material, "file_n_col", 1))) if layer else "1")
        self.file_eps_im_col_var = tk.StringVar(value=str(getattr(layer.material, "file_eps_im_col", getattr(layer.material, "file_k_col", 2))) if layer else "2")
        self.file_delimiter_var = tk.StringVar(value=getattr(layer.material, "file_delimiter", "auto") if layer else "auto")
        frm = ttk.Frame(self, padding=12)
        frm.grid(row=0, column=0, sticky="nsew")
        frm.columnconfigure(1, weight=1)
        frm.columnconfigure(2, weight=0)
        ttk.Label(frm, text="Name").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.name_var, width=24).grid(row=0, column=1, columnspan=2, sticky="ew")
        ttk.Label(frm, text="Thickness (nm, or inf)").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.thick_var, width=24).grid(row=1, column=1, columnspan=2, sticky="ew")
        ttk.Label(frm, text="Material input").grid(row=2, column=0, sticky="w")
        ttk.Radiobutton(frm, text="constant n", variable=self.mode_var, value="n").grid(row=2, column=1, sticky="w")
        ttk.Radiobutton(frm, text="constant epsilon", variable=self.mode_var, value="eps").grid(row=3, column=1, sticky="w")
        ttk.Radiobutton(frm, text="file E,n,k", variable=self.mode_var, value="file").grid(row=4, column=1, sticky="w")
        ttk.Label(frm, text="n real / imag").grid(row=5, column=0, sticky="w")
        n_pair = ttk.Frame(frm)
        n_pair.grid(row=5, column=1, columnspan=2, sticky="w")
        ttk.Entry(n_pair, textvariable=self.n_re_var, width=12).pack(side="left", padx=(0, 8))
        ttk.Entry(n_pair, textvariable=self.n_im_var, width=12).pack(side="left")

        ttk.Label(frm, text="eps real / imag").grid(row=6, column=0, sticky="w")
        eps_pair = ttk.Frame(frm)
        eps_pair.grid(row=6, column=1, columnspan=2, sticky="w")
        ttk.Entry(eps_pair, textvariable=self.eps_re_var, width=12).pack(side="left", padx=(0, 8))
        ttk.Entry(eps_pair, textvariable=self.eps_im_var, width=12).pack(side="left")
        ttk.Label(frm, text="File").grid(row=7, column=0, sticky="w", pady=(8, 2))
        ttk.Entry(frm, textvariable=self.file_var, width=32).grid(row=7, column=1, columnspan=2, sticky="ew", pady=(8, 2))

        file_btns = ttk.Frame(frm)
        file_btns.grid(row=8, column=1, columnspan=2, sticky="w", pady=(2, 8))
        ttk.Button(file_btns, text="Browse/map", command=self.browse, width=14).pack(side="left", padx=(0, 6))
        ttk.Button(file_btns, text="View/plot/remap", command=self.remap_current_file, width=16).pack(side="left")

        b = ttk.Frame(frm)
        b.grid(row=9, column=0, columnspan=3, sticky="e", pady=(10, 0))
        ttk.Button(b, text="OK", command=self.ok, width=10).pack(side="left", padx=(0, 6))
        ttk.Button(b, text="Cancel", command=self.destroy, width=10).pack(side="left")
        self.grab_set()
        self.wait_visibility()
        self.focus()

    def browse(self):
        path = filedialog.askopenfilename(
            filetypes=[("Data files", "*.csv *.txt *.dat"), ("CSV/text", "*.csv *.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        self.file_var.set(path)
        existing = Material(
            mode="file",
            file_path=path,
            file_x_kind=self.file_x_kind_var.get(),
            file_data_kind=self.file_data_kind_var.get(),
            file_x_col=int(self.file_x_col_var.get()),
            file_n_col=int(self.file_n_col_var.get()),
            file_k_col=int(self.file_k_col_var.get()),
            file_eps_re_col=int(self.file_eps_re_col_var.get()),
            file_eps_im_col=int(self.file_eps_im_col_var.get()),
            file_delimiter=self.file_delimiter_var.get(),
        )
        dlg = MaterialFileMapperDialog(self, path, existing)
        self.wait_window(dlg)
        if dlg.result:
            self.file_var.set(dlg.result["file_path"])
            self.file_x_kind_var.set(dlg.result["file_x_kind"])
            self.file_data_kind_var.set(dlg.result["file_data_kind"])
            self.file_x_col_var.set(str(dlg.result["file_x_col"]))
            self.file_n_col_var.set(str(dlg.result["file_n_col"]))
            self.file_k_col_var.set(str(dlg.result["file_k_col"]))
            self.file_eps_re_col_var.set(str(dlg.result["file_eps_re_col"]))
            self.file_eps_im_col_var.set(str(dlg.result["file_eps_im_col"]))
            self.file_delimiter_var.set(dlg.result["file_delimiter"])

    def remap_current_file(self):
        path = self.file_var.get().strip()
        if not path:
            messagebox.showinfo("View/remap file", "No file selected yet.", parent=self)
            return
        existing = Material(
            mode="file",
            file_path=path,
            file_x_kind=self.file_x_kind_var.get(),
            file_data_kind=self.file_data_kind_var.get(),
            file_x_col=int(self.file_x_col_var.get()),
            file_n_col=int(self.file_n_col_var.get()),
            file_k_col=int(self.file_k_col_var.get()),
            file_eps_re_col=int(self.file_eps_re_col_var.get()),
            file_eps_im_col=int(self.file_eps_im_col_var.get()),
            file_delimiter=self.file_delimiter_var.get(),
        )
        dlg = MaterialFileMapperDialog(self, path, existing)
        self.wait_window(dlg)
        if dlg.result:
            self.file_var.set(dlg.result["file_path"])
            self.file_x_kind_var.set(dlg.result["file_x_kind"])
            self.file_data_kind_var.set(dlg.result["file_data_kind"])
            self.file_x_col_var.set(str(dlg.result["file_x_col"]))
            self.file_n_col_var.set(str(dlg.result["file_n_col"]))
            self.file_k_col_var.set(str(dlg.result["file_k_col"]))
            self.file_eps_re_col_var.set(str(dlg.result["file_eps_re_col"]))
            self.file_eps_im_col_var.set(str(dlg.result["file_eps_im_col"]))
            self.file_delimiter_var.set(dlg.result["file_delimiter"])

    def ok(self):
        try:
            thick = self.thick_var.get().strip().lower()
            thickness_m = np.inf if thick in {"inf", "infinite", "semi", "semi-infinite"} else float(thick) * 1e-9
            self.result = Layer(
                self.name_var.get().strip() or "layer",
                thickness_m,
                Material(
                    mode=self.mode_var.get(),
                    n_value=complex(float(self.n_re_var.get()), float(self.n_im_var.get())),
                    eps_value=complex(float(self.eps_re_var.get()), float(self.eps_im_var.get())),
                    file_path=self.file_var.get().strip(),
                    file_x_kind=self.file_x_kind_var.get(),
                    file_data_kind=self.file_data_kind_var.get(),
                    file_x_col=int(self.file_x_col_var.get()),
                    file_n_col=int(self.file_n_col_var.get()),
                    file_k_col=int(self.file_k_col_var.get()),
                    file_eps_re_col=int(self.file_eps_re_col_var.get()),
                    file_eps_im_col=int(self.file_eps_im_col_var.get()),
                    file_delimiter=self.file_delimiter_var.get(),
                ),
            )
            self.destroy()
        except Exception as exc:
            messagebox.showerror("Invalid layer", str(exc), parent=self)


class LDOSApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(" 2D-LDOS calculator")
        self.geometry("1500x980")
        self.minsize(1320, 860)
        self.layers: list[Layer] = []
        self.result: Optional[dict[str, np.ndarray | float]] = None
        self.sweep_result: Optional[dict[str, np.ndarray]] = None
        self.thickness_sweep_result: Optional[dict[str, np.ndarray]] = None
        self.energy_spectra_result: Optional[dict[str, np.ndarray]] = None
        self.thickness_spectra_result: Optional[dict[str, np.ndarray]] = None
        self.energy_map_colorbar = None
        self.thickness_map_colorbar = None
        self.energy_var = tk.StringVar(value="1.651")
        self.s_start_var = tk.StringVar(value="1e-20")
        self.s_stop_var = tk.StringVar(value="0.99999")
        self.s_points_var = tk.StringVar(value="1000")
        self.dipole_idx_var = tk.StringVar(value="2")
        self.dipole_frac_var = tk.StringVar(value="0.5")
        self.plot_kind_var = tk.StringVar(value="horizontal")
        self.angular_x_var = tk.StringVar(value="kparallel")
        self.energy_min_var = tk.StringVar(value="1.5")
        self.energy_max_var = tk.StringVar(value="1.8")
        self.energy_points_var = tk.StringVar(value="31")
        self.int_s_min_var = tk.StringVar(value="1.01")
        self.int_s_max_var = tk.StringVar(value="300")
        self.energy_show_spectra_var = tk.BooleanVar(value=False)
        self.thick_layer_idx_var = tk.StringVar(value="1")
        self.thick_min_nm_var = tk.StringVar(value="0")
        self.thick_max_nm_var = tk.StringVar(value="3.3")
        self.thick_points_var = tk.StringVar(value="11")
        self.thick_int_s_min_var = tk.StringVar(value="1.01")
        self.thick_int_s_max_var = tk.StringVar(value="300")
        self.thick_show_spectra_var = tk.BooleanVar(value=False)
        self._build_ui()
        self._load_example_stack()
        self.refresh_tree()
        self.update_stack_visualization()


    def help_text(self, key: str) -> str:
        texts = {
            "Energy E (eV)": "Photon/emitter energy used to evaluate k0 = E/(hbar*c) and all spectral material files. File materials are interpolated at this energy.",
            "s start": "Lower limit of s = k_parallel/k0 (vacuum) for the angular-spectrum LDOS calculation. s < 1 is inside the light cone; s > 1 is evanescent/near-field.",
            "s stop": "Upper limit of s = k_parallel/k0 (vacuum) for the angular-spectrum LDOS calculation. Large values probe highly confined near-field/FRET components.",
            "s points": "Number of sampled s values between s start and s stop. More points gives smoother integrals but takes longer.",
            "Dipole layer index": "Index of the layer containing the emitter/dipole. Layer indices are shown in the layer table, counted from top to bottom starting at 0.",
            "Dipole fraction from bottom": "Position of the dipole inside its layer: 0 = bottom interface of that layer, 1 = top interface, 0.5 = middle.",
            "Layer table": "Ordered stack from top to bottom. Semi-infinite layers use thickness inf. Finite layers are used for propagation through the multilayer stack.",
            "Dipole orientation": "Global orientation used for LDOS calculations. Horizontal uses the in-plane dipole channel; perpendicular uses the out-of-plane channel.",
            "Energy sweep": "Repeats the LDOS integral for many energies. The integral limits choose which s-range contributes to the plotted LDOS integral.",
            "Thickness sweep": "Repeats the LDOS integral while changing the selected layer thickness. The original thickness is restored after the sweep.",
            "Angular spectra export": "Exports the angular spectra from a sweep. The spectra are sampled on the common s = k_parallel/k0 grid; k_parallel changes with energy in an energy sweep.",
            "2D sweep map": "Shows the angular LDOS as a color map. The vertical axis is the sweep parameter and the horizontal axis follows the angular-spectrum x-axis choice (k_parallel or s). Energy-sweep k_parallel maps use the correct energy-dependent k_parallel grid.",
            "Stack visualization": "Schematic stack drawing. Thickness is not to scale; it is for checking layer order, names, and dipole position.",
            "Save load config": "Saves/loads the layer stack, global LDOS settings, sweep settings, and angular-spectrum display settings to/from JSON.",
            "Angular spectrum x axis": "Choose k_parallel in nm^-1 or dimensionless s = k_parallel/k0 (vacuum) for angular-spectrum plots.",
        }
        return texts.get(key, "No detailed help text has been added for this item yet.")

    def info_button(self, parent, key: str, title: Optional[str] = None):
        return ttk.Button(parent, text="ⓘ", width=2, command=lambda k=key, t=title: messagebox.showinfo(t or k, self.help_text(k)))

    def _build_ui(self):
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)
        ttk.Label(root, text="2D-LDOS: A tool for calculating the optical DOS of dipoles inside layered structures", font=("Arial", 10, "bold")).pack(fill="x", pady=(0,6))
        body = ttk.Frame(root)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="y")
        right = ttk.Frame(body)
        right.pack(side="right", fill="both", expand=True, padx=(10, 0))
        self._build_left(left)
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True)
        self._build_spectrum_tab()
        self._build_stack_tab()
        self._build_sweep_tab()
        self._build_thickness_tab()

    def _build_left(self, left):
        settings = ttk.LabelFrame(left, text="Calculation settings", padding=8)
        settings.pack(fill="x")
        for i, (label, var) in enumerate([
            ("Energy E (eV)", self.energy_var), ("s start", self.s_start_var), ("s stop", self.s_stop_var),
            ("s points", self.s_points_var), ("Dipole layer index", self.dipole_idx_var), ("Dipole fraction from bottom", self.dipole_frac_var),
        ]):
            ttk.Label(settings, text=label).grid(row=i, column=0, sticky="w", pady=2)
            self.info_button(settings, label).grid(row=i, column=1, sticky="w", padx=(3, 6), pady=2)
            ttk.Entry(settings, textvariable=var, width=16).grid(row=i, column=2, sticky="ew", pady=2)
        layers_box = ttk.LabelFrame(left, text="Layers: top to bottom", padding=8)
        layers_box.pack(fill="both", expand=True, pady=(8, 0))
        layer_help_row = ttk.Frame(layers_box)
        layer_help_row.pack(fill="x", pady=(0, 4))
        ttk.Label(layer_help_row, text="Ordered multilayer stack").pack(side="left")
        self.info_button(layer_help_row, "Layer table", "Layer table help").pack(side="left", padx=4)
        self.info_button(layer_help_row, "Save load config", "Save/load configuration help").pack(side="left", padx=4)
        cols = ("idx", "name", "d_nm", "mode", "value")
        self.tree = ttk.Treeview(layers_box, columns=cols, show="headings", height=16)
        for c, w in zip(cols, [45, 120, 80, 55, 180]):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<Double-1>", lambda e: self.edit_layer())
        btns = ttk.Frame(layers_box)
        btns.pack(fill="x", pady=(8, 0))
        for txt, cmd, r, c in [
            ("Add layer", self.add_layer, 0, 0), ("Copy layer", self.copy_layer, 0, 1), ("Edit", self.edit_layer, 0, 2),
            ("Remove", self.remove_layer, 1, 0), ("Clear all", self.clear_layers, 1, 1), ("Example", self._load_example_stack, 1, 2),
            ("Move up", lambda: self.move_layer(-1), 2, 0), ("Move down", lambda: self.move_layer(1), 2, 1),
            ("Save config", self.save_config, 3, 0), ("Load config", self.load_config, 3, 1),
        ]:
            ttk.Button(btns, text=txt, command=cmd).grid(row=r, column=c, padx=2, pady=2, sticky="ew")
        calc_box = ttk.Frame(left)
        calc_box.pack(fill="x", pady=(8, 0))
        ttk.Radiobutton(calc_box, text="Horizontal", variable=self.plot_kind_var, value="horizontal").pack(side="left")
        ttk.Radiobutton(calc_box, text="Perpendicular", variable=self.plot_kind_var, value="perpendicular").pack(side="left")
        self.info_button(calc_box, "Dipole orientation", "Dipole orientation help").pack(side="left", padx=(2, 6))
        ttk.Button(calc_box, text="Calculate LDOS", command=self.calculate).pack(side="left", padx=8)
        ttk.Button(calc_box, text="Export CSV", command=self.export_csv).pack(side="left")

        plot_opts = ttk.LabelFrame(left, text="Angular-spectrum x axis", padding=6)
        plot_opts.pack(fill="x", pady=(8, 0))
        ttk.Label(plot_opts, text="x axis").grid(row=0, column=0, sticky="w", padx=2, pady=2)
        angular_x_combo = ttk.Combobox(
            plot_opts, textvariable=self.angular_x_var,
            values=["kparallel", "s"], width=10, state="readonly"
        )
        angular_x_combo.grid(row=0, column=1, sticky="w", padx=2, pady=2)
        angular_x_combo.bind("<<ComboboxSelected>>", lambda _event: self.redraw_current_angular_plots())
        self.info_button(plot_opts, "Angular spectrum x axis", "Angular-spectrum x-axis help").grid(
            row=0, column=2, sticky="w", padx=2
        )

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(left, textvariable=self.status_var, wraplength=470).pack(fill="x", pady=(8, 0))

    def _build_spectrum_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Angular spectrum")
        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel(r"$k_{//}$ (nm$^{-1}$)")
        self.ax.set_ylabel(r"$d\rho/dk_\parallel\, k_0/\rho_0$")
        self.canvas = FigureCanvasTkAgg(self.fig, master=tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, tab)
        self.toolbar.update()
        self._attach_plot_properties_button(self.toolbar, self.ax, self.canvas, title="Angular spectrum properties")

    def _build_stack_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Stack visualization")
        hdr = ttk.Frame(tab)
        hdr.pack(fill="x", padx=4, pady=(4, 0))
        ttk.Label(hdr, text="Stack schematic").pack(side="left")
        self.info_button(hdr, "Stack visualization", "Stack visualization help").pack(side="left", padx=4)
        self.stack_fig = Figure(figsize=(7, 5), dpi=100)
        self.stack_ax = self.stack_fig.add_subplot(111)
        self.stack_canvas = FigureCanvasTkAgg(self.stack_fig, master=tab)
        self.stack_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.stack_toolbar = NavigationToolbar2Tk(self.stack_canvas, tab)
        self.stack_toolbar.update()
        ttk.Button(tab, text="Refresh stack visualization", command=self.update_stack_visualization).pack(anchor="e", pady=4, padx=4)

    def _build_sweep_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Energy sweep")
        controls = ttk.LabelFrame(tab, text="Energy-dependent integral", padding=8)
        controls.pack(fill="x", padx=4, pady=4)
        self.info_button(controls, "Energy sweep", "Energy sweep help").grid(row=0, column=10, sticky="w", padx=4)
        fields = [
            ("Energy min (eV)", self.energy_min_var), ("Energy max (eV)", self.energy_max_var), ("Energy points", self.energy_points_var),
            ("Integral s min", self.int_s_min_var), ("Integral s max", self.int_s_max_var),
        ]
        for i, (label, var) in enumerate(fields):
            ttk.Label(controls, text=label).grid(row=0, column=2*i, sticky="w", padx=3)
            ttk.Entry(controls, textvariable=var, width=10).grid(row=0, column=2*i+1, sticky="w", padx=3)
        ttk.Checkbutton(controls, text="Also draw angular-spectrum overlay in Curves tab", variable=self.energy_show_spectra_var).grid(row=1, column=0, columnspan=6, pady=4, sticky="w")
        ttk.Button(controls, text="Run energy sweep", command=self.run_energy_sweep).grid(row=2, column=0, columnspan=2, pady=6, sticky="w")
        ttk.Button(controls, text="Export sweep CSV", command=self.export_sweep_csv).grid(row=2, column=2, columnspan=2, pady=6, sticky="w")
        ttk.Button(controls, text="Export angular spectra CSV", command=self.export_energy_spectra_csv).grid(row=2, column=4, columnspan=2, pady=6, sticky="w")
        self.info_button(controls, "Angular spectra export", "Angular spectra export help").grid(row=2, column=6, sticky="w", padx=3)
        self.info_button(controls, "2D sweep map", "2D sweep map help").grid(row=2, column=7, sticky="w", padx=3)

        self.energy_view_notebook = ttk.Notebook(tab)
        self.energy_view_notebook.pack(fill="both", expand=True, padx=4, pady=4)

        curves_tab = ttk.Frame(self.energy_view_notebook)
        map_tab = ttk.Frame(self.energy_view_notebook)
        self.energy_view_notebook.add(curves_tab, text="Curves")
        self.energy_view_notebook.add(map_tab, text="2D map")

        plot_pane = ttk.PanedWindow(curves_tab, orient="vertical")
        plot_pane.pack(fill="both", expand=True)
        top_frame = ttk.LabelFrame(plot_pane, text="Integral plot — toolbar above", padding=2)
        bottom_frame = ttk.LabelFrame(plot_pane, text="Angular spectra plot — toolbar above", padding=2)
        plot_pane.add(top_frame, weight=1)
        plot_pane.add(bottom_frame, weight=1)
        self.sweep_fig = Figure(figsize=(7, 3.2), dpi=100)
        self.sweep_ax = self.sweep_fig.add_subplot(111)
        self.sweep_canvas = FigureCanvasTkAgg(self.sweep_fig, master=top_frame)
        self.sweep_toolbar = NavigationToolbar2Tk(self.sweep_canvas, top_frame)
        self.sweep_toolbar.update()
        self._attach_plot_properties_button(self.sweep_toolbar, self.sweep_ax, self.sweep_canvas, title="Energy integral plot properties")
        self.sweep_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.energy_spectra_fig = Figure(figsize=(7, 3.2), dpi=100)
        self.energy_spectra_ax = self.energy_spectra_fig.add_subplot(111)
        self.energy_spectra_canvas = FigureCanvasTkAgg(self.energy_spectra_fig, master=bottom_frame)
        self.energy_spectra_toolbar = NavigationToolbar2Tk(self.energy_spectra_canvas, bottom_frame)
        self.energy_spectra_toolbar.update()
        self._attach_plot_properties_button(self.energy_spectra_toolbar, self.energy_spectra_ax, self.energy_spectra_canvas, title="Energy spectra plot properties")
        self.energy_spectra_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.energy_map_fig = Figure(figsize=(7, 6.2), dpi=100)
        self.energy_map_ax = self.energy_map_fig.add_subplot(111)
        self.energy_map_canvas = FigureCanvasTkAgg(self.energy_map_fig, master=map_tab)
        self.energy_map_toolbar = NavigationToolbar2Tk(self.energy_map_canvas, map_tab)
        self.energy_map_toolbar.update()
        self._attach_plot_properties_button(
            self.energy_map_toolbar, self.energy_map_ax, self.energy_map_canvas,
            mappable_getter=lambda: getattr(self, "energy_map_mappable", None),
            colorbar_getter=lambda: getattr(self, "energy_map_colorbar", None),
            title="Energy–k map properties",
        )
        self.energy_map_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.energy_map_ax.set_title("Run an energy sweep to generate the 2D LDOS map")
        self.energy_map_canvas.draw()

    def _build_thickness_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Thickness sweep")
        controls = ttk.LabelFrame(tab, text="Thickness-dependent integral", padding=8)
        controls.pack(fill="x", padx=4, pady=4)
        self.info_button(controls, "Thickness sweep", "Thickness sweep help").grid(row=0, column=6, sticky="w", padx=4)
        fields = [
            ("Layer index to sweep", self.thick_layer_idx_var),
            ("Thickness min (nm)", self.thick_min_nm_var),
            ("Thickness max (nm)", self.thick_max_nm_var),
            ("Thickness points", self.thick_points_var),
            ("Integral s min", self.thick_int_s_min_var),
            ("Integral s max", self.thick_int_s_max_var),
        ]
        for i, (label, var) in enumerate(fields):
            row = 0 if i < 3 else 1
            col = 2 * (i % 3)
            ttk.Label(controls, text=label).grid(row=row, column=col, sticky="w", padx=3, pady=2)
            ttk.Entry(controls, textvariable=var, width=12).grid(row=row, column=col + 1, sticky="w", padx=3, pady=2)
        ttk.Checkbutton(controls, text="Also draw angular-spectrum overlay in Curves tab", variable=self.thick_show_spectra_var).grid(row=2, column=0, columnspan=6, pady=4, sticky="w")
        ttk.Button(controls, text="Run thickness sweep", command=self.run_thickness_sweep).grid(row=3, column=0, columnspan=2, pady=6, sticky="w")
        ttk.Button(controls, text="Export thickness sweep CSV", command=self.export_thickness_sweep_csv).grid(row=3, column=2, columnspan=2, pady=6, sticky="w")
        ttk.Button(controls, text="Export angular spectra CSV", command=self.export_thickness_spectra_csv).grid(row=3, column=4, columnspan=2, pady=6, sticky="w")
        self.info_button(controls, "Angular spectra export", "Angular spectra export help").grid(row=3, column=6, sticky="w", padx=3)
        self.info_button(controls, "2D sweep map", "2D sweep map help").grid(row=3, column=7, sticky="w", padx=3)

        self.thickness_view_notebook = ttk.Notebook(tab)
        self.thickness_view_notebook.pack(fill="both", expand=True, padx=4, pady=4)

        curves_tab = ttk.Frame(self.thickness_view_notebook)
        map_tab = ttk.Frame(self.thickness_view_notebook)
        self.thickness_view_notebook.add(curves_tab, text="Curves")
        self.thickness_view_notebook.add(map_tab, text="2D map")

        plot_pane = ttk.PanedWindow(curves_tab, orient="vertical")
        plot_pane.pack(fill="both", expand=True)
        top_frame = ttk.LabelFrame(plot_pane, text="Integral plot — toolbar above", padding=2)
        bottom_frame = ttk.LabelFrame(plot_pane, text="Angular spectra plot — toolbar above", padding=2)
        plot_pane.add(top_frame, weight=1)
        plot_pane.add(bottom_frame, weight=1)
        self.thick_fig = Figure(figsize=(7, 3.2), dpi=100)
        self.thick_ax = self.thick_fig.add_subplot(111)
        self.thick_canvas = FigureCanvasTkAgg(self.thick_fig, master=top_frame)
        self.thick_toolbar = NavigationToolbar2Tk(self.thick_canvas, top_frame)
        self.thick_toolbar.update()
        self._attach_plot_properties_button(self.thick_toolbar, self.thick_ax, self.thick_canvas, title="Thickness integral plot properties")
        self.thick_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.thick_spectra_fig = Figure(figsize=(7, 3.2), dpi=100)
        self.thick_spectra_ax = self.thick_spectra_fig.add_subplot(111)
        self.thick_spectra_canvas = FigureCanvasTkAgg(self.thick_spectra_fig, master=bottom_frame)
        self.thick_spectra_toolbar = NavigationToolbar2Tk(self.thick_spectra_canvas, bottom_frame)
        self.thick_spectra_toolbar.update()
        self._attach_plot_properties_button(self.thick_spectra_toolbar, self.thick_spectra_ax, self.thick_spectra_canvas, title="Thickness spectra plot properties")
        self.thick_spectra_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.thickness_map_fig = Figure(figsize=(7, 6.2), dpi=100)
        self.thickness_map_ax = self.thickness_map_fig.add_subplot(111)
        self.thickness_map_canvas = FigureCanvasTkAgg(self.thickness_map_fig, master=map_tab)
        self.thickness_map_toolbar = NavigationToolbar2Tk(self.thickness_map_canvas, map_tab)
        self.thickness_map_toolbar.update()
        self._attach_plot_properties_button(
            self.thickness_map_toolbar, self.thickness_map_ax, self.thickness_map_canvas,
            mappable_getter=lambda: getattr(self, "thickness_map_mappable", None),
            colorbar_getter=lambda: getattr(self, "thickness_map_colorbar", None),
            title="Thickness–k map properties",
        )
        self.thickness_map_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.thickness_map_ax.set_title("Run a thickness sweep to generate the 2D LDOS map")
        self.thickness_map_canvas.draw()


    def _load_example_stack(self):
        # Generic, lossless demonstration stack for the public release.
        # The dipole sits at the center of a 100 nm n=1.5 dielectric layer
        # between air and a semi-infinite n=2 substrate.
        self.layers = [
            Layer("air", np.inf, Material("n", n_value=1.0 + 0j)),
            Layer("dielectric_dipole", 100e-9, Material("n", n_value=1.5 + 0j)),
            Layer("substrate", np.inf, Material("n", n_value=2.0 + 0j)),
        ]
        self.dipole_idx_var.set("1")
        self.dipole_frac_var.set("0.5")
        self.thick_layer_idx_var.set("1")
        self.refresh_tree()
        self.update_stack_visualization()

    def selected_index(self) -> Optional[int]:
        sel = self.tree.selection()
        return None if not sel else int(self.tree.item(sel[0], "values")[0])

    def refresh_tree(self):
        if not hasattr(self, "tree"):
            return
        for row in self.tree.get_children():
            self.tree.delete(row)
        for i, layer in enumerate(self.layers):
            d = "inf" if not np.isfinite(layer.thickness_m) else f"{layer.thickness_m / 1e-9:.6g}"
            mat = layer.material
            if mat.mode == "n": val = f"n={mat.n_value.real:g}+{mat.n_value.imag:g}i"
            elif mat.mode == "eps": val = f"eps={mat.eps_value.real:g}+{mat.eps_value.imag:g}i"
            else: val = os.path.basename(mat.file_path) or "file not set"
            self.tree.insert("", "end", values=(i, layer.name, d, mat.mode, val))
        self.update_stack_visualization()

    def add_layer(self):
        dlg = LayerDialog(self); self.wait_window(dlg)
        if dlg.result:
            idx = self.selected_index()
            self.layers.append(dlg.result) if idx is None else self.layers.insert(idx + 1, dlg.result)
            self.refresh_tree()


    def copy_layer(self):
        idx = self.selected_index()
        if idx is None:
            messagebox.showinfo("Copy layer", "Select a layer to copy first."); return
        original = self.layers[idx]
        copied = Layer.from_dict(original.to_dict())
        copied.name = original.name + "_copy"
        self.layers.insert(idx + 1, copied)
        self.refresh_tree()
        children = self.tree.get_children()
        if idx + 1 < len(children):
            self.tree.selection_set(children[idx + 1])
            self.tree.focus(children[idx + 1])
        self.status_var.set(f"Copied layer {idx} to position {idx + 1}.")

    def clear_layers(self):
        if not self.layers:
            self.status_var.set("Layer list is already empty."); return
        ok = messagebox.askyesno("Clear all layers", "Delete all layers and start from an empty stack?")
        if not ok:
            return
        self.layers = []
        self.result = None
        self.sweep_result = None
        self.thickness_sweep_result = None
        self.refresh_tree()
        self.ax.clear()
        self.ax.set_xlabel(r"$k_{//}$ (nm$^{-1}$)")
        self.ax.set_ylabel(r"$d\rho/dk_\parallel\, k_0/\rho_0$")
        self.canvas.draw()
        self.status_var.set("All layers deleted. Add layers to build a new stack.")

    def edit_layer(self):
        idx = self.selected_index()
        if idx is None:
            messagebox.showinfo("Edit layer", "Select a layer first."); return
        dlg = LayerDialog(self, self.layers[idx]); self.wait_window(dlg)
        if dlg.result:
            self.layers[idx] = dlg.result; self.refresh_tree()

    def remove_layer(self):
        idx = self.selected_index()
        if idx is not None:
            del self.layers[idx]; self.refresh_tree()

    def move_layer(self, direction: int):
        idx = self.selected_index()
        if idx is None: return
        new = idx + direction
        if 0 <= new < len(self.layers):
            self.layers[idx], self.layers[new] = self.layers[new], self.layers[idx]
            self.refresh_tree(); self.tree.selection_set(self.tree.get_children()[new])

    def calculate(self):
        try:
            energy, s_start, s_stop, npts = float(self.energy_var.get()), float(self.s_start_var.get()), float(self.s_stop_var.get()), int(self.s_points_var.get())
            dip_idx, dip_frac = int(self.dipole_idx_var.get()), float(self.dipole_frac_var.get())
            s = np.linspace(s_start, s_stop, npts)
            self.result = calculate_ldos(self.layers, dip_idx, dip_frac, energy, s)
            x, xlabel = self.get_angular_x(np.asarray(self.result["s"]), np.asarray(self.result["kpar"]) / 1e9)
            y = np.asarray(self.result["dpds_hor"] if self.plot_kind_var.get() == "horizontal" else self.result["dpds_perp"])
            self.ax.clear(); self.plot_one_spectrum(self.ax, x, y, linewidth=2)
            self.apply_angular_axis_options(self.ax, xlabel)
            self.ax.set_ylabel(r"$d\rho/dk_\parallel\, k_0/\rho_0$")
            self.ax.set_title(f"LDOS angular spectrum, E={energy:g} eV ({self.plot_kind_var.get()})"); self.fig.tight_layout(); self.canvas.draw()
            self.status_var.set(f"Done. LDOS horizontal={self.result['ldos_hor']:.6g}, perpendicular={self.result['ldos_perp']:.6g}, k0={self.result['k0']:.6g} m^-1")
            self.notebook.select(0)
        except Exception as exc:
            traceback.print_exc(); messagebox.showerror("Calculation error", str(exc)); self.status_var.set("Calculation failed: " + str(exc))

    def run_energy_sweep(self):
        try:
            e_min, e_max, e_pts = float(self.energy_min_var.get()), float(self.energy_max_var.get()), int(self.energy_points_var.get())
            s_min, s_max, npts = float(self.int_s_min_var.get()), float(self.int_s_max_var.get()), int(self.s_points_var.get())
            dip_idx, dip_frac = int(self.dipole_idx_var.get()), float(self.dipole_frac_var.get())
            energies = np.linspace(e_min, e_max, e_pts)
            s = np.linspace(s_min, s_max, npts)
            hor, perp, spectra_hor, spectra_perp, kpar_rows = [], [], [], [], []
            for E in energies:
                res = calculate_ldos(self.layers, dip_idx, dip_frac, float(E), s)
                hor.append(res["ldos_hor"]); perp.append(res["ldos_perp"])
                # Store spectra for every sweep point. This costs little extra because calculate_ldos
                # already evaluates them, and it makes the 2D map immediately available.
                spectra_hor.append(np.asarray(res["dpds_hor"]))
                spectra_perp.append(np.asarray(res["dpds_perp"]))
                kpar_rows.append(np.asarray(res["kpar"]) / 1e9)
                self.status_var.set(f"Energy sweep running: E={E:.4g} eV"); self.update_idletasks()
            spectra_hor = np.asarray(spectra_hor)
            spectra_perp = np.asarray(spectra_perp)
            kpar_grid = np.asarray(kpar_rows)
            self.sweep_result = {"energy_ev": energies, "ldos_horizontal": np.asarray(hor), "ldos_perpendicular": np.asarray(perp), "s_min": np.array([s_min]), "s_max": np.array([s_max]), "active_orientation": np.array([self.plot_kind_var.get()])}
            self.sweep_ax.clear()
            active = self.plot_kind_var.get()
            active_vals = hor if active == "horizontal" else perp
            self.sweep_ax.plot(energies, active_vals, linewidth=2, label=active)
            self.sweep_ax.set_xlabel("Energy E (eV)"); self.sweep_ax.set_ylabel(r"$\int_{s_{min}}^{s_{max}} d\rho/dk_\parallel\, ds$")
            self.sweep_ax.set_title(f"Energy-dependent LDOS integral ({active}), s={s_min:g} to {s_max:g}")
            self.sweep_ax.legend(frameon=False); self.sweep_ax.set_ylim(bottom=0); self.sweep_fig.tight_layout(); self.sweep_canvas.draw()

            self.energy_spectra_result = {
                "s": s,
                "kpar_nm": kpar_grid,
                "sweep_values": energies,
                "sweep_label": "energy_ev",
                "dpds_horizontal": spectra_hor,
                "dpds_perpendicular": spectra_perp,
            }
            if self.energy_show_spectra_var.get():
                ysets = spectra_hor if active == "horizontal" else spectra_perp
                labels = [f"E={E:.4g} eV" for E in energies]
                # Overlay curves use s for a common x grid if k_parallel varies with energy.
                if self.angular_x_var.get() == "s":
                    xplot = s
                else:
                    # For a conventional overlay, use each spectrum's own k_parallel grid.
                    self.plot_energy_spectra_overlay_variable_x(ysets, labels, s, kpar_grid)
                    xplot = None
                if xplot is not None:
                    self.plot_spectra_on_axis(self.energy_spectra_ax, self.energy_spectra_fig, self.energy_spectra_canvas, xplot, ysets, labels, "Angular spectra for energy sweep")
            else:
                self.energy_spectra_ax.clear(); self.energy_spectra_ax.set_title("Angular-spectrum overlay not requested — 2D map is still available"); self.energy_spectra_canvas.draw()

            self.plot_energy_sweep_map()
            self.status_var.set("Energy sweep done. 2D map updated."); self.notebook.select(2)
        except Exception as exc:
            traceback.print_exc(); messagebox.showerror("Energy sweep error", str(exc)); self.status_var.set("Energy sweep failed: " + str(exc))


    def run_thickness_sweep(self):
        try:
            layer_idx = int(self.thick_layer_idx_var.get())
            if not 0 <= layer_idx < len(self.layers):
                raise ValueError("Thickness sweep layer index out of range")
            t_min, t_max, t_pts = float(self.thick_min_nm_var.get()), float(self.thick_max_nm_var.get()), int(self.thick_points_var.get())
            if t_pts < 2:
                raise ValueError("Thickness points must be at least 2")
            energy = float(self.energy_var.get())
            s_min, s_max, npts = float(self.thick_int_s_min_var.get()), float(self.thick_int_s_max_var.get()), int(self.s_points_var.get())
            dip_idx, dip_frac = int(self.dipole_idx_var.get()), float(self.dipole_frac_var.get())
            thicknesses_nm = np.linspace(t_min, t_max, t_pts)
            s = np.linspace(s_min, s_max, npts)
            original_thickness = self.layers[layer_idx].thickness_m
            hor, perp, spectra_hor, spectra_perp = [], [], [], []
            kpar_nm_for_plot = None
            try:
                for t_nm in thicknesses_nm:
                    self.layers[layer_idx].thickness_m = float(t_nm) * 1e-9
                    res = calculate_ldos(self.layers, dip_idx, dip_frac, energy, s)
                    hor.append(res["ldos_hor"]); perp.append(res["ldos_perp"])
                    spectra_hor.append(np.asarray(res["dpds_hor"]))
                    spectra_perp.append(np.asarray(res["dpds_perp"]))
                    kpar_nm_for_plot = np.asarray(res["kpar"]) / 1e9
                    self.status_var.set(f"Thickness sweep running: layer {layer_idx}, d={t_nm:.4g} nm"); self.update_idletasks()
            finally:
                self.layers[layer_idx].thickness_m = original_thickness
                self.refresh_tree()
            spectra_hor = np.asarray(spectra_hor)
            spectra_perp = np.asarray(spectra_perp)
            self.thickness_sweep_result = {"thickness_nm": thicknesses_nm, "ldos_horizontal": np.asarray(hor), "ldos_perpendicular": np.asarray(perp), "layer_index": np.array([layer_idx]), "energy_ev": np.array([energy]), "s_min": np.array([s_min]), "s_max": np.array([s_max]), "active_orientation": np.array([self.plot_kind_var.get()])}
            self.thick_ax.clear()
            active = self.plot_kind_var.get()
            active_vals = hor if active == "horizontal" else perp
            self.thick_ax.plot(thicknesses_nm, active_vals, linewidth=2, label=active)
            self.thick_ax.set_xlabel(f"Layer {layer_idx} thickness (nm)"); self.thick_ax.set_ylabel(r"$\int_{s_{min}}^{s_{max}} d\rho/dk_\parallel\, ds$")
            self.thick_ax.set_title(f"Thickness-dependent LDOS integral ({active}), E={energy:g} eV, s={s_min:g} to {s_max:g}")
            self.thick_ax.legend(frameon=False); self.thick_ax.set_ylim(bottom=0); self.thick_fig.tight_layout(); self.thick_canvas.draw()

            self.thickness_spectra_result = {
                "s": s,
                "kpar_nm": np.asarray(kpar_nm_for_plot),
                "sweep_values": thicknesses_nm,
                "sweep_label": "thickness_nm",
                "dpds_horizontal": spectra_hor,
                "dpds_perpendicular": spectra_perp,
            }
            if self.thick_show_spectra_var.get():
                ysets = spectra_hor if active == "horizontal" else spectra_perp
                labels = [f"d={t:.4g} nm" for t in thicknesses_nm]
                xplot, _ = self.get_angular_x(s, kpar_nm_for_plot)
                self.plot_spectra_on_axis(self.thick_spectra_ax, self.thick_spectra_fig, self.thick_spectra_canvas, xplot, ysets, labels, f"Angular spectra for thickness sweep, layer {layer_idx}")
            else:
                self.thick_spectra_ax.clear(); self.thick_spectra_ax.set_title("Angular-spectrum overlay not requested — 2D map is still available"); self.thick_spectra_canvas.draw()

            self.plot_thickness_sweep_map()
            self.status_var.set("Thickness sweep done. 2D map updated."); self.notebook.select(3)
        except Exception as exc:
            traceback.print_exc(); messagebox.showerror("Thickness sweep error", str(exc)); self.status_var.set("Thickness sweep failed: " + str(exc))


    def _attach_plot_properties_button(self, toolbar, ax, canvas, mappable_getter=None, colorbar_getter=None, title="Plot properties"):
        """Add a small properties button to an embedded Matplotlib toolbar."""
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=4, pady=2)
        ttk.Button(
            toolbar,
            text="Plot properties",
            command=lambda: self.open_plot_properties(
                ax, canvas,
                mappable_getter=mappable_getter,
                colorbar_getter=colorbar_getter,
                title=title,
            ),
        ).pack(side="left", padx=2)

    def open_plot_properties(self, ax, canvas, mappable_getter=None, colorbar_getter=None, title="Plot properties"):
        """Open a lightweight Matplotlib-style editor for the currently displayed plot.

        Axis limits and axis scales can be changed for every plot. If a 2D-map
        mappable is supplied, the dialog also exposes vmin/vmax, linear/log
        color normalization, and the colormap. Changes only affect the display;
        they do not rerun the LDOS calculation.
        """
        win = tk.Toplevel(self)
        win.title(title)
        win.transient(self)
        win.resizable(False, False)
        frm = ttk.Frame(win, padding=10)
        frm.pack(fill="both", expand=True)

        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        xmin_var = tk.StringVar(value=f"{xlim[0]:.8g}")
        xmax_var = tk.StringVar(value=f"{xlim[1]:.8g}")
        ymin_var = tk.StringVar(value=f"{ylim[0]:.8g}")
        ymax_var = tk.StringVar(value=f"{ylim[1]:.8g}")
        xscale_var = tk.StringVar(value=ax.get_xscale())
        yscale_var = tk.StringVar(value=ax.get_yscale())

        axes_box = ttk.LabelFrame(frm, text="Axes", padding=8)
        axes_box.grid(row=0, column=0, sticky="ew")
        labels = [("X min", xmin_var), ("X max", xmax_var), ("Y min", ymin_var), ("Y max", ymax_var)]
        for i, (lab, var) in enumerate(labels):
            ttk.Label(axes_box, text=lab).grid(row=i//2, column=(i%2)*2, sticky="w", padx=3, pady=3)
            ttk.Entry(axes_box, textvariable=var, width=14).grid(row=i//2, column=(i%2)*2+1, padx=3, pady=3)
        ttk.Label(axes_box, text="X scale").grid(row=2, column=0, sticky="w", padx=3, pady=3)
        ttk.Combobox(axes_box, textvariable=xscale_var, values=["linear", "log"], state="readonly", width=11).grid(row=2, column=1, padx=3, pady=3)
        ttk.Label(axes_box, text="Y scale").grid(row=2, column=2, sticky="w", padx=3, pady=3)
        ttk.Combobox(axes_box, textvariable=yscale_var, values=["linear", "log"], state="readonly", width=11).grid(row=2, column=3, padx=3, pady=3)

        mappable = mappable_getter() if mappable_getter is not None else None
        vmin_var = vmax_var = norm_var = cmap_var = None
        if mappable is not None:
            color_box = ttk.LabelFrame(frm, text="2D map colors", padding=8)
            color_box.grid(row=1, column=0, sticky="ew", pady=(8, 0))
            vmin, vmax = mappable.get_clim()
            vmin_var = tk.StringVar(value=f"{vmin:.8g}" if vmin is not None else "")
            vmax_var = tk.StringVar(value=f"{vmax:.8g}" if vmax is not None else "")
            norm_var = tk.StringVar(value="log" if isinstance(mappable.norm, LogNorm) else "linear")
            cmap_var = tk.StringVar(value=mappable.get_cmap().name)
            ttk.Label(color_box, text="vmin").grid(row=0, column=0, sticky="w", padx=3, pady=3)
            ttk.Entry(color_box, textvariable=vmin_var, width=14).grid(row=0, column=1, padx=3, pady=3)
            ttk.Label(color_box, text="vmax").grid(row=0, column=2, sticky="w", padx=3, pady=3)
            ttk.Entry(color_box, textvariable=vmax_var, width=14).grid(row=0, column=3, padx=3, pady=3)
            ttk.Label(color_box, text="Color scale").grid(row=1, column=0, sticky="w", padx=3, pady=3)
            ttk.Combobox(color_box, textvariable=norm_var, values=["linear", "log"], state="readonly", width=11).grid(row=1, column=1, padx=3, pady=3)
            ttk.Label(color_box, text="Colormap").grid(row=1, column=2, sticky="w", padx=3, pady=3)
            ttk.Combobox(color_box, textvariable=cmap_var, values=["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "gray"], width=11).grid(row=1, column=3, padx=3, pady=3)

        btns = ttk.Frame(frm)
        btns.grid(row=2, column=0, sticky="e", pady=(10, 0))

        def apply_changes():
            try:
                xmin, xmax = float(xmin_var.get()), float(xmax_var.get())
                ymin, ymax = float(ymin_var.get()), float(ymax_var.get())
                if xmin == xmax or ymin == ymax:
                    raise ValueError("Axis minimum and maximum must be different.")
                if xscale_var.get() == "log" and (xmin <= 0 or xmax <= 0):
                    raise ValueError("Logarithmic x-axis limits must be positive.")
                if yscale_var.get() == "log" and (ymin <= 0 or ymax <= 0):
                    raise ValueError("Logarithmic y-axis limits must be positive.")
                ax.set_xscale(xscale_var.get())
                ax.set_yscale(yscale_var.get())
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

                current_mappable = mappable_getter() if mappable_getter is not None else None
                if current_mappable is not None and vmin_var is not None:
                    vmin = float(vmin_var.get()); vmax = float(vmax_var.get())
                    if vmax <= vmin:
                        raise ValueError("vmax must be larger than vmin.")
                    if norm_var.get() == "log":
                        if vmin <= 0:
                            raise ValueError("For logarithmic color scaling, vmin must be positive.")
                        current_mappable.set_norm(LogNorm(vmin=vmin, vmax=vmax))
                    else:
                        current_mappable.set_norm(Normalize(vmin=vmin, vmax=vmax))
                    current_mappable.set_cmap(cmap_var.get().strip() or "viridis")
                    cb = colorbar_getter() if colorbar_getter is not None else None
                    if cb is not None:
                        cb.update_normal(current_mappable)
                canvas.draw_idle()
            except Exception as exc:
                messagebox.showerror("Plot properties", str(exc), parent=win)

        ttk.Button(btns, text="Apply", command=apply_changes).pack(side="left", padx=3)
        ttk.Button(btns, text="Close", command=win.destroy).pack(side="left", padx=3)

    def _remove_map_colorbar(self, which: str):
        attr = "energy_map_colorbar" if which == "energy" else "thickness_map_colorbar"
        cb = getattr(self, attr, None)
        if cb is not None:
            try:
                cb.remove()
            except Exception:
                pass
            setattr(self, attr, None)

    def _map_values_for_orientation(self, spectra_result):
        if self.plot_kind_var.get() == "horizontal":
            return np.asarray(spectra_result["dpds_horizontal"], dtype=float)
        return np.asarray(spectra_result["dpds_perpendicular"], dtype=float)

    def plot_energy_sweep_map(self):
        if self.energy_spectra_result is None:
            return
        r = self.energy_spectra_result
        Z = self._map_values_for_orientation(r)
        energies = np.asarray(r["sweep_values"], dtype=float)
        s = np.asarray(r["s"], dtype=float)
        self._remove_map_colorbar("energy")
        ax = self.energy_map_ax
        ax.clear()
        if self.angular_x_var.get() == "s":
            X, Y = np.meshgrid(s, energies)
            xlabel = r"$s=k_{\parallel}/k_0$ (vacuum)"
        else:
            X = np.asarray(r["kpar_nm"], dtype=float)
            if X.ndim == 1:
                X = np.broadcast_to(X[None, :], Z.shape)
            Y = np.broadcast_to(energies[:, None], Z.shape)
            xlabel = r"$k_{//}$ (nm$^{-1}$)"
        mesh = ax.pcolormesh(X, Y, np.ma.masked_invalid(Z), shading="auto")
        self.energy_map_mappable = mesh
        self.energy_map_colorbar = self.energy_map_fig.colorbar(mesh, ax=ax, pad=0.02)
        self.energy_map_colorbar.set_label(r"$d\rho/dk_\parallel\, k_0/\rho_0$")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Energy E (eV)")
        ax.set_title(f"Energy–k LDOS map ({self.plot_kind_var.get()})")
        ax.grid(False)
        self.energy_map_fig.tight_layout()
        self.energy_map_canvas.draw()

    def plot_thickness_sweep_map(self):
        if self.thickness_spectra_result is None:
            return
        r = self.thickness_spectra_result
        Z = self._map_values_for_orientation(r)
        thicknesses = np.asarray(r["sweep_values"], dtype=float)
        s = np.asarray(r["s"], dtype=float)
        self._remove_map_colorbar("thickness")
        ax = self.thickness_map_ax
        ax.clear()
        if self.angular_x_var.get() == "s":
            x = s
            xlabel = r"$s=k_{\parallel}/k_0$ (vacuum)"
        else:
            x = np.asarray(r["kpar_nm"], dtype=float)
            xlabel = r"$k_{//}$ (nm$^{-1}$)"
        X, Y = np.meshgrid(x, thicknesses)
        mesh = ax.pcolormesh(X, Y, np.ma.masked_invalid(Z), shading="auto")
        self.thickness_map_mappable = mesh
        self.thickness_map_colorbar = self.thickness_map_fig.colorbar(mesh, ax=ax, pad=0.02)
        self.thickness_map_colorbar.set_label(r"$d\rho/dk_\parallel\, k_0/\rho_0$")
        ax.set_xlabel(xlabel)
        layer_idx = int(self.thickness_sweep_result["layer_index"][0]) if self.thickness_sweep_result is not None else int(self.thick_layer_idx_var.get())
        ax.set_ylabel(f"Layer {layer_idx} thickness (nm)")
        ax.set_title(f"Thickness–k LDOS map ({self.plot_kind_var.get()})")
        ax.grid(False)
        self.thickness_map_fig.tight_layout()
        self.thickness_map_canvas.draw()

    def plot_energy_spectra_overlay_variable_x(self, ysets, labels, s_values, kpar_grid):
        ax = self.energy_spectra_ax
        ax.clear()
        max_legend = 25
        for i, y in enumerate(ysets):
            label = labels[i] if i < max_legend else None
            x = np.asarray(kpar_grid[i], dtype=float)
            self.plot_one_spectrum(ax, x, y, linewidth=1.1, label=label)
        self.apply_angular_axis_options(ax, r"$k_{//}$ (nm$^{-1}$)")
        ax.set_ylabel(r"$d\rho/dk_\parallel\, k_0/\rho_0$")
        ax.set_title("Angular spectra for energy sweep" + (" (horizontal)" if self.plot_kind_var.get() == "horizontal" else " (perpendicular)"))
        if len(ysets) <= max_legend:
            ax.legend(frameon=False, fontsize=7)
        else:
            ax.text(0.02, 0.98, f"{len(ysets)} spectra; legend hidden", transform=ax.transAxes, va="top")
        self.energy_spectra_fig.tight_layout()
        self.energy_spectra_canvas.draw()

    def get_angular_x(self, s_values: np.ndarray, kpar_nm: np.ndarray):
        if self.angular_x_var.get() == "s":
            return np.asarray(s_values), r"$s=k_{\parallel}/k_0 (vacuum)$"
        return np.asarray(kpar_nm), r"$k_{//}$ (nm$^{-1}$)"

    def plot_one_spectrum(self, ax, x: np.ndarray, y: np.ndarray, **kwargs):
        ax.plot(np.asarray(x, dtype=float), np.asarray(y, dtype=float), **kwargs)

    def apply_angular_axis_options(self, ax, xlabel: str):
        ax.set_xlabel(xlabel)
        ax.set_xscale("linear")
        ax.set_yscale("linear")
        ax.set_ylim(bottom=0)
        ax.grid(True, which="both", alpha=0.25)

    def redraw_current_angular_plots(self):
        if self.result is not None:
            s = np.asarray(self.result["s"])
            k = np.asarray(self.result["kpar"]) / 1e9
            x, xlabel = self.get_angular_x(s, k)
            y = np.asarray(self.result["dpds_hor"] if self.plot_kind_var.get() == "horizontal" else self.result["dpds_perp"])
            self.ax.clear(); self.plot_one_spectrum(self.ax, x, y, linewidth=2)
            self.apply_angular_axis_options(self.ax, xlabel)
            self.ax.set_ylabel(r"$d\rho/dk_\parallel\, k_0/\rho_0$")
            self.ax.set_title(f"LDOS angular spectrum ({self.plot_kind_var.get()})")
            self.fig.tight_layout(); self.canvas.draw()
        if self.energy_spectra_result is not None:
            r = self.energy_spectra_result
            ysets = r["dpds_horizontal"] if self.plot_kind_var.get() == "horizontal" else r["dpds_perpendicular"]
            labels = [f"E={v:.4g} eV" for v in np.asarray(r["sweep_values"])]
            if self.energy_show_spectra_var.get():
                if self.angular_x_var.get() == "s":
                    self.plot_spectra_on_axis(self.energy_spectra_ax, self.energy_spectra_fig, self.energy_spectra_canvas, np.asarray(r["s"]), ysets, labels, "Angular spectra for energy sweep")
                else:
                    self.plot_energy_spectra_overlay_variable_x(ysets, labels, np.asarray(r["s"]), np.asarray(r["kpar_nm"]))
            self.plot_energy_sweep_map()
        if self.thickness_spectra_result is not None:
            r = self.thickness_spectra_result
            ysets = r["dpds_horizontal"] if self.plot_kind_var.get() == "horizontal" else r["dpds_perpendicular"]
            labels = [f"d={v:.4g} nm" for v in np.asarray(r["sweep_values"])]
            if self.thick_show_spectra_var.get():
                x, _ = self.get_angular_x(np.asarray(r["s"]), np.asarray(r["kpar_nm"]))
                self.plot_spectra_on_axis(self.thick_spectra_ax, self.thick_spectra_fig, self.thick_spectra_canvas, x, ysets, labels, "Angular spectra for thickness sweep")
            self.plot_thickness_sweep_map()
        self.status_var.set("Redrew angular-spectrum plots and 2D maps using current axis/orientation options.")


    def plot_spectra_on_axis(self, ax, fig, canvas, x_values: np.ndarray, ysets: Sequence[np.ndarray], labels: Sequence[str], title: str):
        ax.clear()
        max_legend = 25
        for i, y in enumerate(ysets):
            label = labels[i] if i < max_legend else None
            self.plot_one_spectrum(ax, x_values, y, linewidth=1.1, label=label)
        xlabel = r"$s=k_{\parallel}/k_0 (vacuum)$" if self.angular_x_var.get() == "s" else r"$k_{//}$ (nm$^{-1}$)"
        self.apply_angular_axis_options(ax, xlabel)
        ax.set_ylabel(r"$d\rho/dk_\parallel\, k_0/\rho_0$")
        ax.set_title(title + (" (horizontal)" if self.plot_kind_var.get() == "horizontal" else " (perpendicular)"))
        if len(ysets) <= max_legend:
            ax.legend(frameon=False, fontsize=7)
        else:
            ax.text(0.02, 0.98, f"{len(ysets)} spectra; legend hidden", transform=ax.transAxes, va="top")
        fig.tight_layout()
        canvas.draw()

    def show_spectra_overlay(self, x_nm: np.ndarray, ysets: Sequence[np.ndarray], labels: Sequence[str], title: str):
        win = tk.Toplevel(self)
        win.title(title)
        fig = Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)
        max_legend = 25
        for i, y in enumerate(ysets):
            label = labels[i] if i < max_legend else None
            ax.plot(x_nm, y, linewidth=1.2, label=label)
        ax.set_xlabel(r"$k_{//}$ (nm$^{-1}$)")
        ax.set_ylabel(r"$d\rho/dk_\parallel\, k_0/\rho_0$")
        ax.set_title(title + (" (horizontal)" if self.plot_kind_var.get() == "horizontal" else " (perpendicular)"))
        ax.set_ylim(bottom=0)
        if len(ysets) <= max_legend:
            ax.legend(frameon=False, fontsize=8)
        else:
            ax.text(0.02, 0.98, f"{len(ysets)} spectra; legend hidden", transform=ax.transAxes, va="top")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        canvas.draw()

    def update_stack_visualization(self):
        if not hasattr(self, "stack_ax"): return
        ax = self.stack_ax; ax.clear()
        n = len(self.layers)
        if n == 0:
            ax.text(0.5, 0.5, "No layers", ha="center", va="center"); self.stack_canvas.draw(); return
        dip_idx = int(self.dipole_idx_var.get()) if self.dipole_idx_var.get().strip().isdigit() else -1
        dip_frac = float(self.dipole_frac_var.get()) if self._is_float(self.dipole_frac_var.get()) else 0.5
        height = 1.0 / n
        for i, layer in enumerate(self.layers):
            y = 1.0 - (i + 1) * height
            rect = Rectangle((0.15, y), 0.7, height * 0.92, alpha=0.55)
            ax.add_patch(rect)
            d_text = "inf" if not np.isfinite(layer.thickness_m) else f"{layer.thickness_m/1e-9:g} nm"
            ax.text(0.5, y + height * 0.46, f"{i}: {layer.name}\n{d_text}", ha="center", va="center", fontsize=10)
            if i == dip_idx:
                yd = y + dip_frac * height * 0.92
                marker = Polygon([[0.88, yd], [0.95, yd + 0.025], [0.95, yd - 0.025]], closed=True)
                ax.add_patch(marker)
                ax.text(0.965, yd, "dipole", va="center", fontsize=10)
        ax.set_xlim(0, 1.08); ax.set_ylim(0, 1); ax.axis("off"); ax.set_title("Layer stack, top to bottom (not to scale)")
        self.stack_fig.tight_layout(); self.stack_canvas.draw()

    @staticmethod
    def _is_float(x: str) -> bool:
        try: float(x); return True
        except ValueError: return False



    def config_dict(self) -> dict:
        return {
            "energy_ev": self.energy_var.get(), "s_start": self.s_start_var.get(), "s_stop": self.s_stop_var.get(), "s_points": self.s_points_var.get(),
            "dipole_layer_index": self.dipole_idx_var.get(), "dipole_fraction_from_bottom": self.dipole_frac_var.get(),
            "dipole_orientation": self.plot_kind_var.get(),
            "angular_spectrum_display": {"x_axis": self.angular_x_var.get()},
            "energy_sweep": {"energy_min": self.energy_min_var.get(), "energy_max": self.energy_max_var.get(), "energy_points": self.energy_points_var.get(), "integral_s_min": self.int_s_min_var.get(), "integral_s_max": self.int_s_max_var.get(), "show_angular_spectra": bool(self.energy_show_spectra_var.get())},
            "thickness_sweep": {"layer_index": self.thick_layer_idx_var.get(), "thickness_min_nm": self.thick_min_nm_var.get(), "thickness_max_nm": self.thick_max_nm_var.get(), "thickness_points": self.thick_points_var.get(), "integral_s_min": self.thick_int_s_min_var.get(), "integral_s_max": self.thick_int_s_max_var.get(), "show_angular_spectra": bool(self.thick_show_spectra_var.get())},
            "layers": [layer.to_dict() for layer in self.layers],
        }

    def save_config(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path: return
        with open(path, "w", encoding="utf-8") as f: json.dump(self.config_dict(), f, indent=2)
        self.status_var.set(f"Saved configuration: {path}")

    def load_config(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f: cfg = json.load(f)
            self.energy_var.set(str(cfg.get("energy_ev", "1.651"))); self.s_start_var.set(str(cfg.get("s_start", "1e-20"))); self.s_stop_var.set(str(cfg.get("s_stop", "0.99999"))); self.s_points_var.set(str(cfg.get("s_points", "1000")))
            self.dipole_idx_var.set(str(cfg.get("dipole_layer_index", "2"))); self.dipole_frac_var.set(str(cfg.get("dipole_fraction_from_bottom", "0.5")))
            self.plot_kind_var.set(str(cfg.get("dipole_orientation", cfg.get("plot_kind", "horizontal"))))
            disp = cfg.get("angular_spectrum_display", {})
            self.angular_x_var.set(str(disp.get("x_axis", "kparallel")))
            sw = cfg.get("energy_sweep", {})
            self.energy_min_var.set(str(sw.get("energy_min", "1.5"))); self.energy_max_var.set(str(sw.get("energy_max", "1.8"))); self.energy_points_var.set(str(sw.get("energy_points", "31")))
            self.int_s_min_var.set(str(sw.get("integral_s_min", "1.01"))); self.int_s_max_var.set(str(sw.get("integral_s_max", "300"))); self.energy_show_spectra_var.set(bool(sw.get("show_angular_spectra", False)))
            th = cfg.get("thickness_sweep", {})
            self.thick_layer_idx_var.set(str(th.get("layer_index", "4"))); self.thick_min_nm_var.set(str(th.get("thickness_min_nm", "0"))); self.thick_max_nm_var.set(str(th.get("thickness_max_nm", "3.3")))
            self.thick_points_var.set(str(th.get("thickness_points", "11"))); self.thick_int_s_min_var.set(str(th.get("integral_s_min", "1.01"))); self.thick_int_s_max_var.set(str(th.get("integral_s_max", "300"))); self.thick_show_spectra_var.set(bool(th.get("show_angular_spectra", False)))
            self.layers = [Layer.from_dict(x) for x in cfg.get("layers", [])]
            self.refresh_tree(); self.status_var.set(f"Loaded configuration: {path}")
        except Exception as exc:
            messagebox.showerror("Load configuration error", str(exc))

    def export_csv(self):
        if self.result is None:
            messagebox.showinfo("Export CSV", "Calculate first."); return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path: return
        arr = np.column_stack([np.asarray(self.result["s"]), np.asarray(self.result["kpar"]) / 1e9, np.asarray(self.result["dpds_hor"]), np.asarray(self.result["dpds_perp"])])
        np.savetxt(path, arr, delimiter=",", header="s,k_parallel_nm^-1,dpds_horizontal,dpds_perpendicular", comments="")
        self.status_var.set(f"Exported {path}")

    def export_sweep_csv(self):
        if self.sweep_result is None:
            messagebox.showinfo("Export sweep CSV", "Run an energy sweep first."); return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path: return
        arr = np.column_stack([self.sweep_result["energy_ev"], self.sweep_result["ldos_horizontal"], self.sweep_result["ldos_perpendicular"]])
        np.savetxt(path, arr, delimiter=",", header="energy_ev,ldos_horizontal,ldos_perpendicular", comments="")
        self.status_var.set(f"Exported sweep {path}")

    def export_thickness_sweep_csv(self):
        if self.thickness_sweep_result is None:
            messagebox.showinfo("Export thickness sweep CSV", "Run a thickness sweep first."); return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path: return
        arr = np.column_stack([self.thickness_sweep_result["thickness_nm"], self.thickness_sweep_result["ldos_horizontal"], self.thickness_sweep_result["ldos_perpendicular"]])
        np.savetxt(path, arr, delimiter=",", header="thickness_nm,ldos_horizontal,ldos_perpendicular", comments="")
        self.status_var.set(f"Exported thickness sweep {path}")

    def _export_spectra_result_csv(self, spectra_result, title: str):
        if spectra_result is None:
            messagebox.showinfo(title, "Run the sweep first."); return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path: return
        svals = np.asarray(spectra_result["s"])
        sweep_vals = np.asarray(spectra_result["sweep_values"])
        k = np.asarray(spectra_result["kpar_nm"])
        hor = np.asarray(spectra_result["dpds_horizontal"])
        perp = np.asarray(spectra_result["dpds_perpendicular"])
        label = spectra_result.get("sweep_label", "sweep")

        # Thickness sweeps have one common k_parallel grid. Energy sweeps do not:
        # k_parallel = s*k0(E), so export each spectrum together with its own k grid.
        if k.ndim == 1:
            cols = [svals, k]
            headers = ["s", "k_parallel_nm^-1"]
            for i, v in enumerate(sweep_vals):
                cols.extend([hor[i], perp[i]])
                headers.extend([f"dpds_horizontal_{label}_{v:.8g}", f"dpds_perpendicular_{label}_{v:.8g}"])
        else:
            cols = [svals]
            headers = ["s"]
            for i, v in enumerate(sweep_vals):
                cols.extend([k[i], hor[i], perp[i]])
                headers.extend([
                    f"k_parallel_nm^-1_{label}_{v:.8g}",
                    f"dpds_horizontal_{label}_{v:.8g}",
                    f"dpds_perpendicular_{label}_{v:.8g}",
                ])
        arr = np.column_stack(cols)
        np.savetxt(path, arr, delimiter=",", header=",".join(headers), comments="")
        self.status_var.set(f"Exported angular spectra {path}")


    def export_energy_spectra_csv(self):
        self._export_spectra_result_csv(self.energy_spectra_result, "Export energy angular spectra CSV")

    def export_thickness_spectra_csv(self):
        self._export_spectra_result_csv(self.thickness_spectra_result, "Export thickness angular spectra CSV")


if __name__ == "__main__":
    LDOSApp().mainloop()
