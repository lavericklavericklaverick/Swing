# Highline Rope-Swing Exploratory Simulator

A self-contained browser model for exploring the coupled vertical motion of a highline attachment and a jumper on a dynamic swing rope.


> **Safety warning:** This is an exploratory educational model, not an engineering design or rigging tool. Do not use its results to determine safe loads, clearances, anchor capacities, equipment suitability, or operating procedures.

## Inputs

All values use SI units:

- **Highline:** anchor span, webbing mass per metre, reference stretch and tension, and initial pretension.
- **Swing:** unloaded effective rope length, reference stretch and tension, and attachment position.
- **Jumper:** mass, horizontal starting offset, and side.
- **Simulation:** assumed damping ratio, duration, timestep, and animation speed.

The timestep affects the numerical solution. Animation speed affects playback only. Repeat important runs with a smaller timestep and compare results.

## Outputs

- Animated side elevation of the highline, swing rope, and jumper path.
- Jumper vertical displacement against time.
- Swing-rope tension and mean highline-segment tension against time.
- Overlay comparison of saved cases.
- A separate distributed multibody animation with peak rope tension, left and right anchor tensions, and jumper load factor.

## Multibody simulation

Open **Multibody simulation**, choose the number of highline and rope elements, enter the swing-rope mass per metre and integration timestep, then select **Run multibody simulation**. This mode reuses the main geometry, material, jumper, damping, and duration inputs.

The highline and swing rope are represented as two-dimensional chains of point masses joined by tension-only axial springs. The highline anchors are fixed, the attachment is a shared moving node, and the jumper mass is added to the free end of the rope. Gravity acts on the distributed webbing and rope masses. The starting slack rope is arranged as a hanging curve with approximately its unloaded length, and the equations are advanced with a semi-implicit Euler step.

More elements improve spatial resolution but require a smaller timestep. Results should be repeated with twice as many elements and half the timestep. Material curves, bending stiffness, contact, knots and exact static settling remain omitted, so this mode is also exploratory rather than safety-validating.

## Model

The simulator uses a two-degree-of-freedom lumped model:

- equal-height, rigid anchors;
- a highline attachment constrained to move vertically at a fixed horizontal position;
- two tension-only, linearly elastic highline segments;
- a point-mass jumper moving in a vertical plane;
- a tension-only, linearly elastic swing rope with assumed viscous damping;
- one third of the webbing mass as the effective moving highline mass; and
- fixed-step fourth-order Runge–Kutta integration.

For either elastic line, stiffness is inferred from one reference force-extension datum:

$$
EA = \frac{F_{\mathrm{ref}}}{\varepsilon_{\mathrm{ref}}},
\qquad
T = \max\left(0, EA\left(\frac{L}{L_0}-1\right)\right).
$$

Initial highline sag is solved using the jumper and effective highline weight. At release, gravity acts on the jumper while the swing rope remains slack until its effective unloaded length is reached.

See **Maths, assumptions, limitations** inside the application for the model's assumptions.

## Current status

The application runs as a useful qualitative and sensitivity-study prototype. It is not yet a defensible predictive simulator for real rigging.

Recommended next work, in priority order:

1. **Validate the equations and initial condition** against hand calculations and controlled test data, especially the transition from pre-release loading to rope engagement.
2. **Add engineering result summaries:** peak tension in each highline segment, peak rope tension, peak jumper acceleration/load factor, minimum clearance, and the time of each peak.
3. **Add numerical checks:** automatic half-timestep convergence comparison, energy/work diagnostics, and warnings when results are timestep-sensitive.
4. **Improve material behaviour:** measured loading/unloading curves, hysteresis, rate effects, knots/connectors, permanent set, and separate damping values for the highline and swing rope.
5. **Improve system dynamics:** distributed line mass, compliant anchors, attachment sliding or horizontal motion, harness/body geometry, and aerodynamic drag where relevant.
6. **Represent uncertainty:** parameter ranges or Monte Carlo runs rather than single assumed values.
7. **Add automated tests:** static equilibrium, symmetry, zero-slack and slack-rope cases, force direction, import/export round trips, and regression fixtures.
8. **Improve case handling:** editable case names, local persistence, downloadable JSON, and explicit schema/version validation.

Any safety-related use would additionally require calibrated inputs, documented acceptance criteria and safety factors, controlled physical validation, and independent review by suitably qualified engineers and rigging professionals.

## Browser compatibility

A current browser with Canvas, `ResizeObserver`, `requestAnimationFrame`, and modern JavaScript support is required. Everything runs locally; the page does not send data to a server.
