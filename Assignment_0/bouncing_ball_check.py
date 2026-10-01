import numpy as np
import matplotlib.pyplot as plt

from models import bouncing_ball as model
from integrators import rk4 as integrator
from simulate import run_simulation

params = model.generate_params()
initial_state = np.array([1.0, 0.0])  # drop from 1 m, starting at rest
sim_time = 3.0

# The contact spring is stiff (natural frequency ~sqrt(stiffness/mass)),
# so it needs a much smaller timestep than the pendulum to resolve --
# too large a dt here is exactly the "tunneling through the floor"
# failure mode described in bouncing_ball.py.
timestep = 2e-4

# --- Check 1: undamped contact -> total energy should stay constant ---
params_undamped = dict(params, ground_damping=0.0)
time_traj, state_traj = run_simulation(
    model, integrator, params_undamped, initial_state, timestep, sim_time
)
kinetic_energy, potential_energy = model.calculate_energy(state_traj, params_undamped)
total_energy = kinetic_energy + potential_energy
relative_drift = np.max(np.abs(total_energy - total_energy[0])) / abs(total_energy[0])
print(f"Undamped: relative energy drift = {relative_drift:.3%}")
assert relative_drift < 0.02, "undamped bounce should conserve energy to within integrator error"

# --- Check 2 & 3: damped contact -> energy bleeds out, apex heights decrease ---
time_traj_d, state_traj_d = run_simulation(
    model, integrator, params, initial_state, timestep, sim_time
)
kinetic_energy_d, potential_energy_d = model.calculate_energy(state_traj_d, params)
total_energy_d = kinetic_energy_d + potential_energy_d
assert np.all(np.diff(total_energy_d) <= 1e-6), "damped total energy should be non-increasing"

height_d = state_traj_d[0]
velocity_d = state_traj_d[1]
is_apex = (velocity_d[:-1] > 0) & (velocity_d[1:] <= 0)
apex_heights = height_d[:-1][is_apex]
print("Damped: successive bounce apex heights:", np.round(apex_heights, 4))
assert np.all(np.diff(apex_heights) <= 1e-6), "each bounce should not exceed the previous one's height"

# --- Check 4: ball should never fall meaningfully through the floor ---
min_height = np.min(height_d)
print(f"Damped: minimum height reached = {min_height:.4f} m (radius = {params['radius']} m)")
assert min_height > -0.01, "ball penetrated the ground far more than the contact model allows -- timestep too large?"

print("\nAll bouncing-ball sanity checks passed.")

# --- plots ---
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 9))
axes[0].plot(time_traj, state_traj[0], label="undamped (elastic)")
axes[0].plot(time_traj_d, height_d, label="damped")
axes[0].axhline(params["radius"], color="gray", linestyle="--", linewidth=0.75, label="ground contact")
axes[0].set_ylabel("Height (m)")
axes[0].set_title("Bouncing ball")
axes[0].legend()

# kinetic vs. potential energy breakdown, same style as the pendulum's energy plot
axes[1].plot(time_traj, potential_energy, label="Potential energy")
axes[1].plot(time_traj, kinetic_energy, label="Kinetic energy")
axes[1].plot(time_traj, total_energy, label="Total energy")
axes[1].set_ylabel("Energy (J)")
axes[1].set_title("Undamped (elastic)")
axes[1].legend()

axes[2].plot(time_traj_d, potential_energy_d, label="Potential energy")
axes[2].plot(time_traj_d, kinetic_energy_d, label="Kinetic energy")
axes[2].plot(time_traj_d, total_energy_d, label="Total energy")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Energy (J)")
axes[2].set_title("Damped")
axes[2].legend()

plt.tight_layout()
plt.show()
