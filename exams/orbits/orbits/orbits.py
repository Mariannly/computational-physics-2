# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
import os
from astropy.constants import c
from astropy.constants import G
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import argparse


# Defining constants
c =  c.to('AU/yr').value  # Speed of light in AU/yr
G =  G.to('AU3/(M_sun yr2)').value # AU^3 / (yr^2 * M_sun)


# Slope functions
def classical_slope(t, s, M):
    """
    Classical Newtonian slope function for the two-body problem.
    Args:
        t (float): Time.
        s (list): State vector [x, y, vx, vy].
        M (float): Mass of the black hole in solar masses.
    Returns:
        dsdt (list): Derivative of the state vector.
    """

    r = np.sqrt(s[0]**2 + s[1]**2)
    k = - G * M / r**3
    dsdt = np.array([
        s[2],
        s[3],
        k * s[0],
        k * s[1]
    ])

    return dsdt

def relativistic_slope(t, s, M):
    """
    Relativistic slope function for the two-body problem.
    Args:
        t (float): Time.
        s (list): State vector [x, y, vx, vy].
        M (float): Mass of the black hole in solar masses.
    Returns:
        dsdt (list): Derivative of the state vector.
    """
    # Implement relativistic gravitational slope

    r = np.sqrt(s[0]**2 + s[1]**2)
    L = np.abs(s[0] * s[3] - s[1] * s[2])
    k = - G * M * (1 + (3 * L) / (r**2 * c**2))/ r**3
    dsdt = np.array([
        s[2],
        s[3],
        k * s[0],
        k * s[1]
    ])

    return dsdt



# State system class
class TwoBodySystem:
    def __init__(self, a, e, M, N, method_name, slope_type, save_map=False):
        """Initialize the simulation with given parameters.
        Args:
            a (float): Semi-major axis in AU.
            e (float): Eccentricity.
            M (float): Mass of the black hole in solar masses.
            N (int): Number of orbits to simulate.
            method_name (str): Integration method ('trapezoidal', 'rk3', 'scipy').
            slope_type (str): Slope type ('classical', 'relativistic').
            save_map (bool): Whether to save the initial map with Schwarzschild radius circle.
        """
        self.a = a
        self.e = e
        self.M = M
        self.save_map = save_map
        self.s0 = self.initialize()
        self.rs = self.schwarzschild_radius()
        self.period = self.orbital_period()
        self.t_span = (0, self.period * N)
        self.method_name = method_name
        self.slope_type = slope_type

    def initialize(self):
        """Compute the initial conditions for the simulation."""
        # Compute initial conditions and return initial state vector
        s0 = [0, self.a * (1 - self.e),  # Periapsis
             -np.sqrt((G*self.M/self.a) * ((1+self.e)/(1-self.e))), 0]  # Velocity at periapsis
        return np.array(s0)
    
    def schwarzschild_radius(self):
        """Compute the Schwarzschild radius of the black hole."""
        return 2 * G * self.M / c**2  # AU

    def orbital_period(self):
        """Compute the orbital period of the planet."""
        return 2 * np.pi * np.sqrt(self.a**3 / (G * self.M))  # in years

    def save_initial_map(self):
        """Save the initial map with Schwarzschild radius circle."""

        # Get the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Go up one level and into "output"
        output_dir = os.path.join(current_dir, "..", "analysis/outputfolder")
        os.makedirs(output_dir, exist_ok=True)

        # Plot and save the initial map with Schwarzschild radius circle
        fig, ax = plt.subplots()

        plt.rcParams['font.family'] = 'DejaVu Sans'
        planet_marker = '\u2205'
        bh_marker = '\u2726'
        ax.text(self.s0[0], self.s0[1], planet_marker, color='orange', size=15, va='center', ha='center', clip_on=True)  # Initial position of planet
        ax.text(0, 0, bh_marker, color='black', size=20, va='center', ha='center', clip_on=True)  # Position of black hole
        circle = Circle((0, 0), self.rs, color='black', fill=False, linestyle=':', linewidth=0.9)
        ax.add_patch(circle)

        ax.set_aspect('equal')
        ax.set_xlim(-self.a * 1.2 , self.a * 1.2 )
        ax.set_ylim(-self.a * 1.2, self.a * 1.2 )
        ax.set_title("Initial Map")
        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")
        ax.grid(color='gray', linestyle='-', linewidth=0.3)
        
        plt.savefig(os.path.join(output_dir, "initial_map_a{:.0f}_e{:.2f}_M{:-.0E}_{}_{}.png".format(self.a, self.e, self.M,self.method_name, self.slope_type)))
        plt.close(fig)


# Integrator class
class SimulationRunner:
    def __init__(self, method_name, slope_type, dt, system):
        """
        Run the simulation with the specified method and slope type.
        Args:
            method_name (str): Integration method ('trapezoidal', 'rk3', 'scipy').
            slope_type (str): Slope type ('classical', 'relativistic').
            dt (float): Time step.
            system (TwoBodySystem): Instance of TwoBodySystem.
        """
        self.method_name = method_name
        self.slope_type = slope_type
        self.dt = dt
        self.system = system
        self.slope = self.select_slope()


    # Integrators methods
    def trapezoidal_euler(self, f, y0, t_span, dt, M):
        """
        Numerical integration of the ODEs using the trapezoidal method.
        Args:
            f (function): Function describing the ODEs.
            y0 (list): Initial state variables.
            t_span (tuple): Time span for the simulation.
            dt (float): Time step.
            M (float): Mass of the black hole.
        Returns:
            t_values (ndarray): Time values.
            y_values (ndarray): Integrated state variables.
        """
        # Initializing
        t_values = np.arange(t_span[0], t_span[1], dt)
        y_values = np.zeros((len(t_values), len(y0)))
        y_values[0] = y0

        # Loop
        for i in range(0, len(t_values) - 1):

            # Frist approximation of the state variables at the next time step
            y_values[i+1] = y_values[i] + dt * f(t_values[i], y_values[i], M)

            # Correcting the approximation with trapezoidal method
            y_values[i+1] = y_values[i] + 0.5 * dt * (f(t_values[i], y_values[i], M) + f(t_values[i+1], y_values[i+1], M))

        return t_values, y_values

    def runge_kutta3(self, f, y0, t_span, dt, M):
        """
        Numerical integration of the ODEs using the third-order Runge-Kutta method.
        Args:
            f (function): Function describing the ODEs.
            y0 (list): Initial state variables.
            t_span (tuple): Time span for the simulation.
            dt (float): Time step.
            M (float): Mass of the black hole.
        Returns:
            t_values (ndarray): Time values.
            y_values (ndarray): Integrated state variables.
        """
        # Initializing
        t_values = np.arange(t_span[0], t_span[1], dt)
        y_values = np.zeros((len(t_values), len(y0)))
        y_values[0] = y0      

        # Loop
        for i in range(0, len(t_values) - 1):
            # First slope
            k1 = f(t_values[i], y_values[i], M)

            # Second slope
            k2 = f(t_values[i] + 0.5 * dt, y_values[i] + 0.5 * dt * k1, M)

            # Third slope
            k3 = f(t_values[i] + dt, y_values[i] - dt * k1 + 2 * dt * k2, M)

            # Updating the state variables
            y_values[i+1] = y_values[i] + dt * (k1 + 4 * k2 + k3) / 6  

        return t_values, y_values

    def scipy_integrator(self, f, y0, t_span, dt, M):
        """
        Numerical integration of the ODEs using SciPy's solve_ivp with RK45.
        Args:
            f (function): Function describing the ODEs.
            y0 (list): Initial state variables.
            t_span (tuple): Time span for the simulation.
            dt (float): Time step.
            M (float): Mass of the black hole.
        Returns:
            t_values (ndarray): Time values.
            y_values (ndarray): Integrated state variables.
        """
        # Initializing
        t_values = np.arange(t_span[0], t_span[1], dt)
        y_values = np.zeros((len(t_values), len(y0)))
        y_values[0] = y0

        # Use solve_ivp to integrate the ODEs
        sol = solve_ivp(f, t_span, y0, method='RK45', t_eval=t_values, args=(M,), vectorized=True)

        # Extract the time and state variables
        t_values = sol.t
        y_values = sol.y.T

        # Check if the integration was successful
        if not sol.success:
            raise RuntimeError("Integration failed: " + sol.message)
        
        return t_values, y_values


    # Select method and slope
    def select_method(self, f, y0, t_span, dt, M):
        if self.method_name == "trapezoidal":
            return self.trapezoidal_euler(f, y0, t_span, dt, M)
        elif self.method_name == "rk3":
            return self.runge_kutta3(f, y0, t_span, dt, M)
        elif self.method_name == "scipy":
            return self.scipy_integrator(f, y0, t_span, dt, M)
        else:
            raise ValueError("Invalid method")

    def select_slope(self):
        if self.slope_type == "classical":
            return classical_slope
        elif self.slope_type == "relativistic":
            return relativistic_slope
        else:
            raise ValueError("Invalid slope type")

    # Run the simulation!
    def run(self):
        """Run the simulation using the selected method and slope function."""
        # slope function
        f = self.slope
        # Initial conditions
        y0 = self.system.s0
        # time span from N periods
        t_span = self.system.t_span
        # Time step
        dt = self.dt
        # Mass of the black hole
        M = self.system.M

        # Run the chosen integrator
        t_values, y_values = self.select_method(f, y0, t_span, dt, M)

        
        # Get the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Go up one level and into "output"
        output_dir = os.path.join(current_dir, "..", "analysis/outputfolder")
        os.makedirs(output_dir, exist_ok=True)

        # Save to outputfolder/
        orbital_history = pd.DataFrame({
            "t": t_values,
            "x": y_values[:, 0],
            "y": y_values[:, 1],
            "vx": y_values[:, 2],
            "vy": y_values[:, 3]
        })
        # Save the data to a CSV file
        orbital_history.to_csv(os.path.join(output_dir, "orbit_data_a{:.0f}_e{:.2f}_M{:-.0E}_{}_{}.csv".format(self.system.a, self.system.e, M, self.method_name, self.slope_type)), index=False)
        
# Animator class
class OrbitalAnimator(SimulationRunner):
    def __init__(self, method_name, slope_type, dt, system):
        """
        Child Class. Initialize the animator with the path to the data file.
        Args:
            method_name (str): Integration method ('trapezoidal', 'rk3', 'scipy').
            slope_type (str): Slope type ('classical', 'relativistic').
            dt (float): Time step.
            system (TwoBodySystem): Instance of TwoBodySystem.
        Returns:
            Animation in gif format.
        """
        super().__init__(method_name, slope_type, dt, system)
        self.rs = system.rs
        self.a = system.a
        self.e = system.e
        self.M = system.M

        # Get the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Go up one level and into "output"
        self.output_dir = os.path.join(current_dir, "..", "analysis/outputfolder")
        os.makedirs(self.output_dir, exist_ok=True)

        # Create the filename for the data file
        filename = "orbit_data_a{:.0f}_e{:.2f}_M{:-.0E}_{}_{}.csv".format(self.a, self.e, self.M, self.method_name, self.slope_type)
        
        # Create the full path to the data file
        self.filepath = os.path.join(self.output_dir, filename)

    def create_gif(self):
        # Read history and animate
        data = pd.read_csv(self.filepath)

        output_filaname = "orbit_a{:.0f}_e{:.2f}_M{:-.0E}_{}_{}.gif".format(self.a, self.e, self.M, self.method_name, self.slope_type)
        output_gif = os.path.join(self.output_dir, output_filaname)

        # Create the figure and axis
        fig, ax = plt.subplots()

        plt.rcParams['font.family'] = 'DejaVu Sans'
        planet_marker = '\u2205'
        bh_marker = '\u2726'

        planet_trail = ax.plot([], [], color='orange', alpha=0.5, marker='', linestyle='-')[0]  # Trajectory of planet
        planet_velocity = ax.quiver([], [], [], [], color='green', scale=60000, width=0.002)  # Velocity vector of planet
        planet = ax.text([], [], planet_marker, color='red', size=15, va='center', ha='center', clip_on=True)  # Position of planet

        ax.text(0, 0, bh_marker, color='black', size=30, va='center', ha='center', clip_on=True)  # Position of black hole
        circle = Circle((0, 0), self.rs, color='black', fill=False, linestyle=':', linewidth=0.9)
        ax.add_patch(circle)

        ax.set_aspect('equal')
        ax.set_xlim(-self.a * 1.4 , self.a * 1.4 )
        ax.set_ylim(-self.a * 1.4 - self.a * self.e, self.a * 1.4 - self.a * self.e ) #Try to center the trajectory
        ax.set_title("Simulation of a planet orbiting a black hole")
        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")
        ax.grid(color='gray', linestyle='-', linewidth=0.3)

        # Create the animation
        def update(frame):
            
            planet_trail.set_data(data.iloc[:frame]['x'], data.iloc[:frame]['y'])
            planet_velocity.set_offsets((data.iloc[frame]['x'], data.iloc[frame]['y']))
            planet_velocity.set_UVC(data.iloc[frame]['vx'], data.iloc[frame]['vy'])
            planet.set_position((float(data.iloc[frame]['x']), float(data.iloc[frame]['y'])))
            
            return planet_trail, planet_velocity, planet

        ani = FuncAnimation(fig, update, frames=len(data), blit=False, interval=200)
        ani.save(output_gif, writer='pillow', fps=20)

        plt.close(fig)


        

# Main function to parse arguments and run the simulation
def parse_args():
    parser = argparse.ArgumentParser(description="Two-body simulation")
    parser.add_argument("--e", type=float, required=True, help="Eccentricity")
    parser.add_argument("--M", type=float, required=True, help="Mass of black hole (solar masses)")
    parser.add_argument("--a", type=float, required=True, help="Semi-major axis in AU")
    parser.add_argument("--N", type=int, default=1, help="Number of orbits")
    parser.add_argument("--dt", type=float, default=1e-3, help="Time step in years")
    parser.add_argument("--method", choices=["trapezoidal", "rk3", "scipy"], default="scipy")
    parser.add_argument("--slope", choices=["classical", "relativistic"], default="classical")
    parser.add_argument("--save_map", action="store_true", help="Save initial map", default=False)
    parser.add_argument("--animate", action="store_true", help="Generate GIF animation", default=False)
    return parser.parse_args()

def main():
    """Main function to initialize, run, and visualize the simulation."""
    args = parse_args()
    
    # Initialize the two-body system
    system = TwoBodySystem(args.a, args.e, args.M, args.N, args.method, args.slope, args.save_map)
    if args.save_map:
        system.save_initial_map()
    
    # Run the simulation
    simulation = SimulationRunner(args.method, args.slope, args.dt, system)
    simulation.run()

    # Generate the animation if requested
    if args.animate:
        animator = OrbitalAnimator(args.method, args.slope, args.dt, system)
        animator.create_gif()

if __name__ == "__main__":
    main()