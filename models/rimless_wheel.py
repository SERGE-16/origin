import numpy as np


#model of the pendulum, need to figure out how to add in the switch of coordinates
#need to figure out how to switch the coordinates
def dynamics(t, state, params):
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]
    damping_coeff = params["damping_coeff"]
    N=params["N"]
    gamma=params["gamma"]

    angle = state[0]
    angular_velocity = state[1]

    angular_acceleration = (
        mass * gravity * length * np.sin(angle)
        - damping_coeff * angular_velocity  # <-- DAMPING TERM
    ) / (mass * length**2)

    state_derivative = np.array([angular_velocity, angular_acceleration])
    return state_derivative


def generate_params():
    params = {
        "gravity": 9.81,  # gravity m/s^2)
        "length": 1,  # rod length (m)
        "mass": 1,  # point mass at end of rod (kg)
        "damping_coeff": 0,  # damping coefficient (kg*m^2/s)
        "N": 6, #number of spokes 
        "gamma": np.pi/6 #angle of incline
    }
    return params


def generate_initial_condition():
    """Start just after impact (theta = gamma - alpha) rolling forward."""
    params = generate_params()
    alpha = np.pi / params["N"]
    return np.array([params["gamma"] - alpha, 1.0])

#need to figure out how to detect if the 2nd leg is touching
def is_touching(state, params):
    gamma=params["gamma"]
    N=params["N"]
    two_alpha=(2*np.pi)/N
    alpha=two_alpha/2

    theta=state[0]

    return (theta>=gamma+alpha) #checks to see if the second leg is touching (Boolean)


def reset_params(state,params):
    N=params["N"]
    two_alpha=(2*np.pi)/N #angle beween adjacent spokes
    theta_minus=state[0]
    ang_vel_minus=state[1]

    theta_plus=theta_minus-two_alpha
    ang_vel_plus=ang_vel_minus*np.cos(two_alpha)

    return (np.array([theta_plus,ang_vel_plus]))



def calculate_energy(state, params):
    """Compute energies for a state ``(2,)`` or trajectory ``(2, N)``."""
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    angle = state[0]  # indexes entire row "vectorized" if state is (2, N)
    angular_velocity = state[1]

    kinetic_energy = 0.5 * mass * (length * angular_velocity) ** 2
    potential_energy = mass * gravity * length * np.cos(angle)
    return kinetic_energy, potential_energy

def calculate_momentum(state,params):
    mass = params["mass"]
    length = params["length"]
    theta_dot=state[1]
    L=mass*length**2*theta_dot
    return(L)