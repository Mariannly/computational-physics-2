# Two-Body Orbital Simulation

A Python module for simulating the orbital dynamics of a planet around a black hole, including both classical Newtonian and relativistic calculations.

## Features

- Classical and relativistic gravitational calculations
- Multiple numerical integration methods:
  - Trapezoidal method
  - 3rd order Runge-Kutta
  - SciPy's RK45 integrator
- Initial configuration visualization
- Orbital animation generation
- Data export in CSV format


## Usage

Basic example:

```bash
python relative_path_to_orbits.py --e 0 --M 5e6 --a 1 --N 3 --dt 1e-6 --method scipy --slope relativistic --save_map --animate
```
(Don't know why my shorcut command is not working, but you can run the script with the full path)


### Arguments

- `--e`: Eccentricity of the orbit
- `--M`: Mass of the black hole (solar masses)
- `--a`: Semi-major axis (AU)
- `--N`: Number of orbits to simulate (default: 1)
- `--dt`: Time step in years (default: 1e-3)
- `--method`: Integration method ['trapezoidal', 'rk3', 'scipy'] (default: 'scipy')
- `--slope`: Force calculation method ['classical', 'relativistic'] (default: 'classical')
- `--save_map`: Save initial configuration plot
- `--animate`: Generate animation of the orbit

## Output

The simulation generates:
- CSV file with orbital data (time, position, velocity)
- Initial configuration plot (optional)
- GIF animation of the orbit (optional)

Output files are saved in `analysis/outputfolder/`.

## Dependencies

- numpy
- matplotlib
- scipy
- astropy
- pandas
- os
- argparse

## License

MIT License

## Author

Mariannly Marquez
mariannly.marquez@yachaytech.edu.ec