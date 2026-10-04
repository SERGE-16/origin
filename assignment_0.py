import numpy as np
import matplotlib.pyplot as plt

from models import pendulum as model
from integrators import rk4 as integrator #change this to correct import 

# Basic simulation of the pendulum

params = {
    "gravity": 9.81,  # gravity m/s^2)
    "length": 1,  # rod length (m)
    "mass": 0.2,  # point mass at end of rod (kg)
    "damping_coeff": 0.0,  # damping coefficient (kg*m^2/s)
    "torque": 0.0,  # torque (Nm)
}


# some set-up
initial_state = np.array([np.pi / 4, 0.0])

#changing these variables 
timestep = .04159 #2e-6 is what I found after sweeping with the code commented out below for the euler
sim_time = 5.0



flag=True
while (timestep<2 and flag):
    #just used the same name as the initial euler file I had made, in the future I would change the python file name from explicit_euler to something else
    time_traj = np.arange(int(sim_time / timestep) + 1) * timestep
    state_traj = np.zeros((2, time_traj.size))
    state_traj[:, 0] = initial_state
    for step, t in enumerate(time_traj[:-1]):
        state_traj[:, step + 1] = integrator(model.dynamics, t, state_traj[:, step], timestep, params)

    #calculates energy with a model library? probably where negative KE comes from (switched the order to fix)
    kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)

    a=kinetic_energy[0] + potential_energy[0]
    b=kinetic_energy[-1] + potential_energy[-1] #had originally compared to the middle, this is mroe generalizable 
    if (np.isclose(a,b, rtol=1e-05, atol=1e-06, equal_nan=False)):
        #print(f"Timestep is {timestep}")
        timestep+=1e-5
    else:
        #print("Total energy is conserved")
        flag=False
        #print(f"The best timestep is {timestep}")



#calculates energy with a model library? probably where negative KE comes from (switched the order to fix)
#kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)

plt.figure()
plt.plot(time_traj, potential_energy, label="Potential energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Pendulum energy")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(state_traj[0, :], state_traj[1, :])
plt.xlabel("Position (rad) - x")
plt.ylabel("Velocity (rad/s) - f(x)")
plt.title("Phase portrait of the pendululm")
plt.tight_layout()
plt.show()

