"""
HIGHLINE ROPE-SWING FEA-STYLE EXPLORATORY SIMULATOR
====================================================

This is an educational, non-validated numerical model. It is not suitable for
rigging design, clearance decisions, equipment selection, anchor assessment,
or life-safety decisions.

Unlike the original two-spring model, this version discretizes both the
highline and the swing rope into independent chains of two-node axial
finite elements. Every element has its own length, strain rate, tension-only
constitutive law, and damping. Mass is distributed to the nodes, and the rope
has internal moving nodes rather than being a single scalar spring.

The model is still intentionally small and transparent. It omits bending
stiffness, contact, knots, connectors, friction, hysteresis, permanent set,
material rate dependence, three-dimensional motion, anchor compliance, and
body/harness geometry. A single stretch datum is only used to estimate a
linear axial rigidity EA.
"""

from __future__ import annotations

import math
import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from typing import Any
from tkinter import messagebox, ttk

import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

G = 9.80665


class InputError(ValueError):
    pass


@dataclass
class FEAConfig:
    span: float = 60.0
    attach_x: float = 30.0
    jumper_offset: float = 4.0
    line_elements: int = 24
    rope_elements: int = 12
    jumper_mass: float = 80.0
    line_mass_gpm: float = 80.0
    rope_mass_gpm: float = 20.0
    line_stretch_pct: float = 5.0
    line_ref_force: float = 5000.0
    rope_stretch_pct: float = 10.0
    rope_ref_force: float = 10000.0
    pretension: float = 1000.0
    rope_length: float = 12.0
    damping_pct: float = 3.0
    nodal_damping_pct: float = 8.0
    duration: float = 8.0
    dt: float = 0.001
    jump_side: str = "Right"

    @property
    def line_ea(self):
        return self.line_ref_force / (self.line_stretch_pct / 100.0)

    @property
    def rope_ea(self):
        return self.rope_ref_force / (self.rope_stretch_pct / 100.0)


@dataclass
class SimulationResult:
    time: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    element_tension: np.ndarray
    element_length: np.ndarray
    energy: np.ndarray
    line_element_count: int
    rope_element_count: int
    attachment_index: int
    jumper_index: int
    initial_residual: float
    relaxation_steps: int


class CableFEModel:
    """Planar lumped-mass cable model with independent highline and rope meshes."""

    def __init__(self, config: FEAConfig):
        self.c = config
        self.line_count_left = max(2, round(config.line_elements * config.attach_x / config.span))
        self.line_count_right = max(2, config.line_elements - self.line_count_left)
        self.line_count = self.line_count_left + self.line_count_right
        self.rope_count = config.rope_elements

        left_x = np.linspace(0.0, config.attach_x, self.line_count_left + 1)
        right_x = np.linspace(config.attach_x, config.span, self.line_count_right + 1)[1:]
        line_xy = np.column_stack((np.r_[left_x, right_x], np.zeros(self.line_count + 1)))
        self.left_anchor = 0
        self.right_anchor = self.line_count
        self.attachment = self.line_count_left
        self.line_node_count = self.line_count + 1

        jumper_x = config.attach_x + (
            abs(config.jumper_offset) if config.jump_side == "Right" else -abs(config.jumper_offset)
        )
        rope_fraction = np.linspace(0.0, 1.0, self.rope_count + 1)
        rope_internal = np.column_stack((
            config.attach_x + (jumper_x - config.attach_x) * rope_fraction[1:-1],
            np.zeros(max(0, self.rope_count - 1)),
        ))
        if config.rope_length > abs(config.jumper_offset):
            rope_internal[:, 1] = self._initial_rope_sag(
                abs(config.jumper_offset), config.rope_length, rope_fraction[1:-1]
            )
        all_xy = np.vstack((line_xy, rope_internal, [[jumper_x, 0.0]]))
        self.jumper = len(all_xy) - 1
        self.node_count = len(all_xy)
        self.initial_xy = all_xy.copy()

        line_pairs = np.column_stack((np.arange(self.line_count), np.arange(1, self.line_count + 1)))
        rope_start = self.line_node_count
        rope_pairs = np.column_stack((
            np.r_[self.attachment, np.arange(rope_start, rope_start + self.rope_count - 1)],
            np.arange(rope_start, rope_start + self.rope_count),
        ))
        self.line_elements = line_pairs
        self.rope_elements = rope_pairs
        self.elements = np.vstack((line_pairs, rope_pairs))
        self.line_element_count = len(line_pairs)
        self.rope_element_count = len(rope_pairs)
        self.element_count = len(self.elements)
        self.ea = np.r_[
            np.full(self.line_element_count, config.line_ea),
            np.full(config.rope_elements, config.rope_ea),
        ]
        line_work_lengths = np.linalg.norm(line_xy[1:] - line_xy[:-1], axis=1)
        rope_rest_lengths = np.full(config.rope_elements, config.rope_length / config.rope_elements)
        self.rest_length = np.r_[
            line_work_lengths / (1.0 + config.pretension / config.line_ea),
            rope_rest_lengths,
        ]
        self.element_mass = np.r_[
            line_work_lengths * config.line_mass_gpm / 1000.0,
            rope_rest_lengths * config.rope_mass_gpm / 1000.0,
        ]
        self.mass = np.zeros(self.node_count)
        for element_index, (node_a, node_b) in enumerate(self.elements):
            half_mass = 0.5 * self.element_mass[element_index]
            self.mass[node_a] += half_mass
            self.mass[node_b] += half_mass
        self.mass[self.jumper] += config.jumper_mass
        self.mass[self.mass < 1e-9] = 1e-9
        self.inv_mass = 1.0 / self.mass
        zeta = config.damping_pct / 100.0
        effective_mass = (self.mass[self.elements[:, 0]] * self.mass[self.elements[:, 1]]) / (
            self.mass[self.elements[:, 0]] + self.mass[self.elements[:, 1]]
        )
        element_stiffness = self.ea / self.rest_length
        self.element_damping = 2.0 * zeta * np.sqrt(element_stiffness * effective_mass)
        # Axial dashpots do not damp transverse cable modes. This small
        # broadband nodal term represents distributed material/hysteretic loss.
        nodal_zeta = config.nodal_damping_pct / 100.0
        first_mode_omega = math.sqrt(config.line_ea / (config.span * max(config.line_mass_gpm * config.span / 1000.0, 0.1)))
        self.nodal_damping = 2.0 * nodal_zeta * first_mode_omega * self.mass
        # This term represents distributed cable loss, not jumper drag. The
        # jumper's horizontal momentum must be changed by rope forces, not by
        # damping sized from the highline's first mode.
        self.nodal_damping[self.jumper] = 0.0
        self.fixed_xy = np.zeros((self.node_count, 2), dtype=bool)
        self.fixed_xy[self.left_anchor, :] = True
        self.fixed_xy[self.right_anchor, :] = True
        self.fixed_xy[self.attachment, 0] = True
        self.fixed_values = self.initial_xy.copy()
        self.fixed_values[self.left_anchor] = [0.0, 0.0]
        self.fixed_values[self.right_anchor] = [config.span, 0.0]
        self.fixed_values[self.attachment, 0] = config.attach_x
        if not np.array_equal(self.line_elements[0], [self.left_anchor, 1]):
            raise InputError("Highline left anchor is not connected to the first element.")
        if not np.array_equal(self.line_elements[-1], [self.right_anchor - 1, self.right_anchor]):
            raise InputError("Highline right anchor is not connected to the final element.")
        if self.attachment not in self.line_elements[self.line_count_left - 1]:
            raise InputError("Swing attachment is not connected to the left highline segment.")
        if self.attachment not in self.line_elements[self.line_count_left]:
            raise InputError("Swing attachment is not connected to the right highline segment.")
        if self.rope_elements[0, 0] != self.attachment or self.rope_elements[-1, 1] != self.jumper:
            raise InputError("Swing rope endpoints are not connected to attachment and jumper.")
        self._relax_dt = min(config.dt, self.stable_dt * 0.5)

    @staticmethod
    def _initial_rope_sag(horizontal_span, rope_length, fractions):
        """Return a parabolic hanging shape with the requested discrete length."""
        if not len(fractions) or rope_length <= horizontal_span:
            return np.zeros_like(fractions)

        def discrete_length(sag):
            points = np.linspace(0.0, 1.0, len(fractions) + 2)
            y = 4.0 * sag * points * (1.0 - points)
            x = horizontal_span * points
            return float(np.sum(np.hypot(np.diff(x), np.diff(y))))

        lower, upper = 0.0, rope_length
        for _ in range(80):
            midpoint = 0.5 * (lower + upper)
            if discrete_length(midpoint) < rope_length:
                lower = midpoint
            else:
                upper = midpoint
        sag = 0.5 * (lower + upper)
        return 4.0 * sag * fractions * (1.0 - fractions)

    @property
    def stable_dt(self):
        """Conservative explicit stability estimate for the stiffest axial element."""
        element_stiffness = self.ea / self.rest_length
        effective_mass = (self.mass[self.elements[:, 0]] * self.mass[self.elements[:, 1]]) / (
            self.mass[self.elements[:, 0]] + self.mass[self.elements[:, 1]]
        )
        return 0.20 * float(np.min(np.sqrt(effective_mass / element_stiffness)))

    def apply_constraints(self, positions, velocities):
        positions[self.fixed_xy] = self.fixed_values[self.fixed_xy]
        velocities[self.fixed_xy] = 0.0

    def element_state(self, positions, velocities):
        node_a = self.elements[:, 0]
        node_b = self.elements[:, 1]
        delta = positions[node_b] - positions[node_a]
        length = np.linalg.norm(delta, axis=1)
        length = np.maximum(length, 1e-12)
        unit = delta / length[:, None]
        relative_velocity = velocities[node_b] - velocities[node_a]
        extension_rate = np.sum(relative_velocity * unit, axis=1)
        extension = length - self.rest_length
        elastic = self.ea * extension / self.rest_length
        active = extension > 0.0
        tension = np.where(active, np.maximum(0.0, elastic + self.element_damping * extension_rate), 0.0)
        return length, extension_rate, tension

    def forces(self, positions, velocities):
        length, _, tension = self.element_state(positions, velocities)
        force = np.zeros_like(positions)
        node_a = self.elements[:, 0]
        node_b = self.elements[:, 1]
        delta = positions[node_b] - positions[node_a]
        unit = delta / np.maximum(length, 1e-12)[:, None]
        element_force = tension[:, None] * unit
        np.add.at(force, node_a, element_force)
        np.add.at(force, node_b, -element_force)
        force[:, 1] += self.mass * G
        force -= self.nodal_damping[:, None] * velocities
        force[self.fixed_xy] = 0.0
        return force, tension, length

    def static_relaxation(self, progress_callback=None):
        """Settle the supported cable with the jumper held at its release point."""
        positions = self.initial_xy.copy()
        velocities = np.zeros_like(positions)
        held = self.fixed_xy.copy()
        held[self.jumper, :] = True
        free_nodes = ~np.any(held, axis=1)
        held_values = positions.copy()
        dt = self._relax_dt
        for step in range(1, 12001):
            force, _, _ = self.forces(positions, velocities)
            acceleration = force * self.inv_mass[:, None]
            acceleration[held] = 0.0
            velocities = velocities + dt * acceleration
            velocities *= 0.92
            positions = positions + dt * velocities
            positions[held] = held_values[held]
            velocities[held] = 0.0
            if step % 100 == 0:
                residual = np.max(np.linalg.norm(force[free_nodes], axis=1))
                if progress_callback is not None:
                    progress_callback(0.35 * step / 12000.0, f"Static relaxation: {step}/12000 iterations")
                if residual < max(0.02, 0.0005 * self.mass.sum() * G):
                    break
        self.initial_xy = positions.copy()
        self.fixed_values = positions.copy()
        self.fixed_values[self.left_anchor] = [0.0, 0.0]
        self.fixed_values[self.right_anchor] = [self.c.span, 0.0]
        self.fixed_values[self.attachment, 0] = self.c.attach_x
        self.fixed_values[self.jumper] = positions[self.jumper]
        initial_residual = float(np.max(np.linalg.norm(force[free_nodes], axis=1)))
        return positions, initial_residual, step

    def mechanical_energy(self, positions, velocities):
        kinetic = 0.5 * np.sum(self.mass[:, None] * velocities * velocities)
        gravitational = np.sum(self.mass * G * positions[:, 1])
        length, _, _ = self.element_state(positions, velocities)
        extension = np.maximum(0.0, length - self.rest_length)
        elastic = 0.5 * np.sum(self.ea * extension * extension / self.rest_length)
        return kinetic + gravitational + elastic

    def run(self, progress_callback=None):
        positions, residual, relax_steps = self.static_relaxation(progress_callback)
        velocities = np.zeros_like(positions)
        self.apply_constraints(positions, velocities)
        dt = self.c.dt
        count = int(math.ceil(self.c.duration / dt)) + 1
        time = np.linspace(0.0, dt * (count - 1), count)
        positions_history = np.empty((count, self.node_count, 2))
        velocities_history = np.empty_like(positions_history)
        tension_history = np.empty((count, self.element_count))
        length_history = np.empty_like(tension_history)
        energy_history = np.empty(count)
        positions_history[0] = positions
        velocities_history[0] = velocities
        _, tension_history[0], length_history[0] = self.forces(positions, velocities)
        energy_history[0] = self.mechanical_energy(positions, velocities)

        substeps = max(1, int(math.ceil(dt / self.stable_dt)))
        h = dt / substeps
        for index in range(1, count):
            for _ in range(substeps):
                force, _, _ = self.forces(positions, velocities)
                velocities += h * force * self.inv_mass[:, None]
                positions += h * velocities
                self.apply_constraints(positions, velocities)
            if not np.all(np.isfinite(positions)) or np.max(np.abs(positions)) > 1e6:
                raise InputError("The solution diverged. Refine the mesh or reduce the timestep.")
            positions_history[index] = positions
            velocities_history[index] = velocities
            _, tension_history[index], length_history[index] = self.forces(positions, velocities)
            energy_history[index] = self.mechanical_energy(positions, velocities)
            if progress_callback is not None and (index == 1 or index % max(1, count // 100) == 0 or index == count - 1):
                progress_callback(0.35 + 0.65 * index / (count - 1), f"Dynamic solve: {index}/{count - 1} output steps")
        return SimulationResult(
            time, positions_history, velocities_history, tension_history, length_history,
            energy_history, self.line_element_count, self.rope_element_count,
            self.attachment, self.jumper, residual, relax_steps,
        )


class App:
    def __init__(self, root):
        self.root = root
        root.title("Highline Rope-Swing: Discretized Cable FEA Explorer")
        root.geometry("1500x930")
        root.minsize(1150, 760)
        self.result = None
        self.model = None
        self.animation = None
        self.speed = tk.DoubleVar(value=1.0)
        self.vars = {}
        self.progress_queue = queue.Queue()
        self.solve_thread = None
        self._build_layout()

    def _build_layout(self):
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(1, weight=1)
        warning = tk.Label(outer, text=(
            "EDUCATIONAL MODEL ONLY: discretization does not make this suitable for rigging or life-safety decisions."
        ), bg="#7f1d1d", fg="white", font=("TkDefaultFont", 10, "bold"), pady=8)
        warning.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self._build_controls(outer)
        self._build_plots(outer)

    def _build_controls(self, parent):
        panel = ttk.Frame(parent, padding=(4, 0, 10, 0))
        panel.grid(row=1, column=0, sticky="ns")
        canvas = tk.Canvas(panel, width=370, highlightthickness=0)
        scrollbar = ttk.Scrollbar(panel, orient="vertical", command=canvas.yview)
        holder = ttk.Frame(canvas)
        holder.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=holder, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="y")
        scrollbar.pack(side="right", fill="y")
        ttk.Label(holder, text="Discretized cable model", font=("TkDefaultFont", 14, "bold")).pack(anchor="w", pady=(0, 8))
        fields = [
            ("Anchor span", "span", "60", "m"),
            ("Attachment from left", "attach_x", "30", "m"),
            ("Jumper offset", "jumper_offset", "4", "m"),
            ("Highline elements", "line_elements", "24", "count"),
            ("Rope elements", "rope_elements", "12", "count"),
            ("Jumper mass", "jumper_mass", "80", "kg"),
            ("Highline mass", "line_mass_gpm", "80", "g/m"),
            ("Rope mass", "rope_mass_gpm", "20", "g/m"),
            ("Highline stretch datum", "line_stretch_pct", "5", "%"),
            ("Highline reference force", "line_ref_force_kn", "5", "kN"),
            ("Rope stretch datum", "rope_stretch_pct", "10", "%"),
            ("Rope reference force", "rope_ref_force_kn", "10", "kN"),
            ("Initial highline pretension", "pretension_kn", "1", "kN"),
            ("Rope tension-free length", "rope_length", "12", "m"),
            ("Element damping ratio", "damping_pct", "3", "%"),
            ("Broadband nodal damping", "nodal_damping_pct", "8", "%"),
            ("Duration", "duration", "8", "s"),
            ("Output timestep", "dt", "0.001", "s"),
        ]
        for label, key, default, unit in fields:
            row = ttk.Frame(holder)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=27).pack(side="left")
            variable = tk.StringVar(value=default)
            self.vars[key] = variable
            ttk.Entry(row, textvariable=variable, width=10).pack(side="left", padx=5)
            ttk.Label(row, text=unit).pack(side="left")
        row = ttk.Frame(holder)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Jump side", width=27).pack(side="left")
        self.side_var = tk.StringVar(value="Right")
        ttk.Combobox(row, textvariable=self.side_var, values=("Left", "Right"), state="readonly", width=8).pack(side="left", padx=5)
        ttk.Label(holder, text="The rope is an independent chain with distributed mass and internal nodes.", foreground="#555", wraplength=350).pack(anchor="w", pady=(6, 2))
        ttk.Label(holder, text="Initial state is obtained by damped relaxation while the jumper is held.", foreground="#555", wraplength=350).pack(anchor="w", pady=(0, 6))
        buttons = ttk.Frame(holder)
        buttons.pack(fill="x", pady=8)
        ttk.Button(buttons, text="Run / animate", command=self.run).pack(side="left", fill="x", expand=True)
        ttk.Button(buttons, text="Stop", command=self.stop).pack(side="left", padx=(6, 0))
        speed_row = ttk.Frame(holder)
        speed_row.pack(fill="x", pady=(0, 8))
        ttk.Label(speed_row, text="Playback speed", width=27).pack(side="left")
        ttk.Scale(speed_row, from_=0.25, to=4.0, variable=self.speed, command=self.update_speed).pack(side="left", fill="x", expand=True, padx=5)
        self.speed_label = ttk.Label(speed_row, text="1.00x", width=6)
        self.speed_label.pack(side="left")
        self.progress_label = ttk.Label(holder, text="Ready", foreground="#555")
        self.progress_label.pack(anchor="w", pady=(2, 2))
        self.progress = ttk.Progressbar(holder, orient="horizontal", mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(0, 8))
        self.summary = tk.Text(holder, width=46, height=18, wrap="word", bg="#f3f4f6")
        self.summary.pack(fill="x")
        self.summary.insert("1.0", "Run the discretized model to see results and diagnostics.\n")
        self.summary.configure(state="disabled")

    def _build_plots(self, parent):
        self.tabs = ttk.Notebook(parent)
        self.tabs.grid(row=1, column=1, sticky="nsew")
        plot_tab = ttk.Frame(self.tabs)
        notes_tab = ttk.Frame(self.tabs)
        self.tabs.add(plot_tab, text="Simulation and diagnostics")
        self.tabs.add(notes_tab, text="What this model means")
        self.fig = Figure(figsize=(10, 7), constrained_layout=True)
        grid = self.fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
        self.ax_scene = self.fig.add_subplot(grid[0, :])
        self.ax_tension = self.fig.add_subplot(grid[1, 0])
        self.ax_energy = self.fig.add_subplot(grid[1, 1])
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, plot_tab).update()
        notes = tk.Text(notes_tab, wrap="word", padx=18, pady=18)
        notes.pack(fill="both", expand=True)
        notes.insert("1.0", """DISCRETIZATION

The highline is a chain of two-node axial elements between fixed anchors. The attachment is a real highline node: its horizontal position is constrained, but its vertical position is free. The swing rope is a separate chain of axial elements sharing only that attachment node and the jumper endpoint. Rope internal nodes carry rope mass and gravity, so rope inertia and changing shape can influence the transient.

ELEMENT LAW

Each element calculates its current length and axial extension rate. Its tension is zero in compression or slack, and otherwise follows T = max(0, EA*(L-L0)/L0 + c*dL/dt). Endpoint forces are equal and opposite. The highline and rope have independent EA, mass, rest lengths, and damping.

INITIALIZATION AND SOLVER

The supported system is damped toward static equilibrium while the jumper is held at its release point. The jumper is then released with zero velocity. A semi-implicit Euler update is used, with automatic substepping based on the stiffest element estimate. This is simple and robust for exploratory work, but it is not a replacement for an implicit nonlinear structural solver.

CHECKS TO RUN

Repeat with half the timestep. Repeat with more highline and rope elements. Compare peak tension, jumper displacement, energy behavior, and the static residual. If those values move materially, the result is not numerically converged.

LIMITATIONS

This remains a planar axial cable model. It has no bending stiffness, bending waves, transverse beam mechanics, contact, knots, connectors, friction, hysteresis, rate-dependent material behavior, permanent set, anchor compliance, body geometry, harness dynamics, air drag, or failure criteria. The output is not a clearance or safety result.
""")
        notes.configure(state="disabled")
        self.draw_placeholder()

    def draw_placeholder(self):
        for axis in (self.ax_scene, self.ax_tension, self.ax_energy):
            axis.clear()
        self.ax_scene.set_title("Enter parameters and select Run / animate")
        self.ax_scene.set_xlabel("Horizontal position (m)")
        self.ax_scene.set_ylabel("Downward displacement (m)")
        self.ax_scene.grid(True, alpha=0.25)
        self.ax_tension.set_title("Element tensions")
        self.ax_energy.set_title("Mechanical energy")
        self.canvas.draw_idle()

    def read_config(self):
        try:
            values: dict[str, Any] = {key: float(variable.get()) for key, variable in self.vars.items()}
        except ValueError as exc:
            raise InputError("Every input must be a valid number.") from exc
        integer_fields = ("line_elements", "rope_elements")
        for key in integer_fields:
            values[key] = int(values[key])
            if values[key] < 2:
                raise InputError(f"{key.replace('_', ' ').title()} must be at least 2.")
        values["line_ref_force"] = values.pop("line_ref_force_kn") * 1000.0
        values["rope_ref_force"] = values.pop("rope_ref_force_kn") * 1000.0
        values["pretension"] = values.pop("pretension_kn") * 1000.0
        values["jump_side"] = self.side_var.get()
        config = FEAConfig(**values)  # type: ignore[arg-type]
        positive = ("span", "jumper_mass", "line_mass_gpm", "rope_mass_gpm", "line_stretch_pct", "line_ref_force", "rope_stretch_pct", "rope_ref_force", "rope_length", "duration", "dt")
        if any(getattr(config, key) <= 0 for key in positive):
            raise InputError("Lengths, masses, stretch data, forces, duration, and timestep must be positive.")
        if not 0 < config.attach_x < config.span:
            raise InputError("Attachment position must lie strictly between the anchors.")
        offset = abs(config.jumper_offset)
        jumper_x = config.attach_x + (offset if config.jump_side == "Right" else -offset)
        if not 0 <= jumper_x <= config.span:
            raise InputError("The initial jumper position must lie between the anchors.")
        if config.pretension < 0 or config.damping_pct < 0 or config.nodal_damping_pct < 0:
            raise InputError("Pretension and damping cannot be negative.")
        if config.dt > 0.02:
            raise InputError("Use an output timestep of 0.02 s or smaller.")
        if config.duration / config.dt > 300000:
            raise InputError("Too many output steps. Increase timestep or reduce duration.")
        return config

    def stop(self):
        if self.animation is not None:
            self.animation.event_source.stop()
            self.animation = None

    def update_speed(self, value):
        speed = float(value)
        self.speed_label.configure(text=f"{speed:.2f}x")
        if self.animation is not None:
            self.animation.event_source.interval = max(10, self.base_interval / speed)

    def run(self):
        self.stop()
        try:
            config = self.read_config()
            model = CableFEModel(config)
            self.progress.configure(value=0)
            self.progress_label.configure(text="Starting solver in background...")
            self.model = model
            self.result = None

            def report_progress(fraction, message):
                self.progress_queue.put(("progress", fraction, message))

            def solve():
                try:
                    result = model.run(progress_callback=report_progress)
                    self.progress_queue.put(("result", result, ""))
                except Exception as exc:
                    self.progress_queue.put(("error", exc, ""))

            self.solve_thread = threading.Thread(target=solve, daemon=True)
            self.solve_thread.start()
            self.poll_solver()
        except Exception as exc:
            self.progress_label.configure(text="Input error")
            messagebox.showerror("Simulation error", str(exc))
            return

    def poll_solver(self):
        try:
            while True:
                kind, value, message = self.progress_queue.get_nowait()
                if kind == "progress":
                    self.progress.configure(value=100.0 * value)
                    self.progress_label.configure(text=f"{message} ({100.0 * value:.0f}%)")
                elif kind == "result":
                    self.result = value
                    self.progress.configure(value=100)
                    self.progress_label.configure(text="Complete (100%)")
                    self.update_plots()
                    self.solve_thread = None
                    return
                elif kind == "error":
                    self.progress_label.configure(text="Solver failed")
                    self.solve_thread = None
                    messagebox.showerror("Simulation error", str(value))
                    return
        except queue.Empty:
            pass
        if self.solve_thread is not None and self.solve_thread.is_alive():
            self.root.after(50, self.poll_solver)

    def update_plots(self):
        result = self.result
        model = self.model
        if result is None or model is None:
            return
        time = result.time
        positions = result.positions
        tension = result.element_tension
        self.ax_scene.clear()
        self.ax_tension.clear()
        self.ax_energy.clear()
        max_y = max(2.0, float(np.max(positions[:, :, 1])) * 1.12)
        min_y = min(-1.0, float(np.min(positions[:, :, 1])) * 1.12)
        margin = max(2.0, model.c.span * 0.05)
        self.ax_scene.set_xlim(-margin, model.c.span + margin)
        self.ax_scene.set_ylim(max_y, min_y)
        self.ax_scene.set_aspect("equal", adjustable="box")
        self.ax_scene.set_title("Independent highline and swing-rope finite-element meshes")
        self.ax_scene.set_xlabel("Horizontal position (m)")
        self.ax_scene.set_ylabel("Downward displacement (m)")
        self.ax_scene.grid(True, alpha=0.25)
        line_artists = []
        for _ in range(model.line_element_count):
            artist, = self.ax_scene.plot([], [], color="#0891b2", lw=2.4)
            line_artists.append(artist)
        rope_artists = []
        for _ in range(model.rope_element_count):
            artist, = self.ax_scene.plot([], [], color="#f59e0b", lw=2.0)
            rope_artists.append(artist)
        jumper_dot, = self.ax_scene.plot([], [], "o", color="#e11d48", ms=9)
        attach_dot, = self.ax_scene.plot([], [], "o", color="#7c3aed", ms=7)
        self.ax_scene.plot([0, model.c.span], [0, 0], "ks", ms=7)
        time_label = self.ax_scene.text(0.02, 0.95, "", transform=self.ax_scene.transAxes, va="top", bbox=dict(facecolor="white", alpha=0.8))
        highline_handle = Line2D([0], [0], color="#0891b2", lw=2.4, label="Highline elements")
        rope_handle = Line2D([0], [0], color="#f59e0b", lw=2.0, label="Independent rope elements")
        self.ax_scene.legend(handles=[highline_handle, rope_handle], loc="upper right", fontsize="small")
        rope_slice = slice(result.line_element_count, None)
        self.ax_tension.plot(time, np.max(tension[:, :result.line_element_count], axis=1) / 1000.0, color="#0891b2", label="Highline max")
        self.ax_tension.plot(time, np.max(tension[:, rope_slice], axis=1) / 1000.0, color="#f59e0b", label="Rope max")
        self.ax_tension.set_title("Maximum element tension by component")
        self.ax_tension.set_xlabel("Time (s)")
        self.ax_tension.set_ylabel("Tension (kN)")
        self.ax_tension.grid(True, alpha=0.3)
        self.ax_tension.legend(fontsize="small")
        relative_energy = result.energy - result.energy[0]
        self.ax_energy.plot(time, relative_energy / 1000.0, color="#334155")
        self.ax_energy.set_title("Mechanical energy change")
        self.ax_energy.set_xlabel("Time (s)")
        self.ax_energy.set_ylabel("Change (kJ)")
        self.ax_energy.grid(True, alpha=0.3)
        stride = max(1, len(time) // 300)
        frames = np.arange(0, len(time), stride)
        moving = line_artists + rope_artists + [jumper_dot, attach_dot, time_label]
        for artist in moving:
            artist.set_animated(True)

        def update(index):
            current = positions[index]
            for element_index, (node_a, node_b) in enumerate(model.line_elements):
                line_artists[element_index].set_data(current[[node_a, node_b], 0], current[[node_a, node_b], 1])
            for rope_index, (node_a, node_b) in enumerate(model.rope_elements):
                rope_artists[rope_index].set_data(current[[node_a, node_b], 0], current[[node_a, node_b], 1])
            jumper_dot.set_data([current[result.jumper_index, 0]], [current[result.jumper_index, 1]])
            attach_dot.set_data([current[result.attachment_index, 0]], [current[result.attachment_index, 1]])
            time_label.set_text(f"t = {time[index]:.2f} s")
            return tuple(moving)

        update(0)
        self.canvas.draw()
        base_interval = max(12, 1000.0 * model.c.duration / len(frames))
        self.base_interval = base_interval
        self.animation = FuncAnimation(self.fig, update, frames=frames, interval=base_interval / self.speed.get(), blit=True, repeat=True, init_func=lambda: tuple(moving))
        self.canvas.draw_idle()
        line_max = float(np.max(tension[:, :result.line_element_count]))
        rope_max = float(np.max(tension[:, rope_slice]))
        rope_taut_fraction = float(np.mean(tension[:, rope_slice] > 0.0))
        jumper_drop = float(np.max(positions[:, result.jumper_index, 1]))
        jumper_horizontal_speed = np.abs(result.velocities[:, result.jumper_index, 0])
        max_horizontal_speed = float(np.max(jumper_horizontal_speed))
        horizontal_speed_at_drop = float(result.velocities[np.argmax(positions[:, result.jumper_index, 1]), result.jumper_index, 0])
        energy_change = float(np.max(np.abs(relative_energy)))
        summary = (
            "EXPLORATORY OUTPUTS ONLY\n\n"
            f"Mesh: {result.line_element_count} highline + {result.rope_element_count} rope elements\n"
            f"Automatic solver substep estimate: {max(1, math.ceil(model.c.dt / model.stable_dt))}\n"
            f"Initial relaxation residual: {result.initial_residual:.3f} N\n"
            f"Relaxation iterations: {result.relaxation_steps}\n"
            f"Maximum jumper drop: {jumper_drop:.3f} m\n"
            f"Maximum jumper horizontal speed: {max_horizontal_speed:.3f} m/s\n"
            f"Horizontal speed at maximum drop: {horizontal_speed_at_drop:.3f} m/s\n"
            f"Maximum highline element tension: {line_max / 1000.0:.3f} kN\n"
            f"Maximum rope element tension: {rope_max / 1000.0:.3f} kN\n"
            f"Rope element samples taut: {100.0 * rope_taut_fraction:.1f}%\n"
            f"Largest absolute energy change: {energy_change / 1000.0:.3f} kJ\n\n"
            "Run timestep and mesh refinement checks before interpreting any output."
        )
        self.summary.configure(state="normal")
        self.summary.delete("1.0", "end")
        self.summary.insert("1.0", summary)
        self.summary.configure(state="disabled")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
