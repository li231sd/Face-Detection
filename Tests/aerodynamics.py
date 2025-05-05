import math
import numpy as np
import matplotlib.pyplot as plt

# Constants
air_density = 1.225  # kg/m^3
gravity = 9.81       # m/s^2

# Drone parameters
volume_m3 = 0.00023013021985  # m³
density_material = 1200       # kg/m³ (plastic assumption)
mass = density_material * volume_m3  # kg
weight = mass * gravity       # N

num_propellers = 4
propeller_diameter_m = 0.127  # meters (5 inch)
propeller_radius_m = propeller_diameter_m / 2
propeller_area = math.pi * (propeller_radius_m ** 2)

drag_coefficient = 1.2  # rough estimate for clunky shapes
frontal_area = 0.03     # m² (small front face of body + camera)
horizontal_speeds = np.linspace(2.2, 4.4, 10)  # 5-10 mph range in m/s

# Simulate different propeller exit velocities
prop_speeds = np.linspace(20, 80, 100)  # m/s 
thrusts = []

for prop_speed in prop_speeds:
    thrust_per_prop = air_density * propeller_area * (prop_speed ** 2)
    total_thrust = thrust_per_prop * num_propellers
    thrusts.append(total_thrust)

# Plot thrust vs. propeller air exit speed
plt.figure(figsize=(8,6))
plt.plot(prop_speeds, thrusts, label='Total Thrust (N)')
plt.axhline(y=weight, color='r', linestyle='--', label='Drone Weight')
plt.title('Drone Thrust vs Propeller Air Speed')
plt.xlabel('Propeller Air Exit Speed (m/s)')
plt.ylabel('Total Thrust (N)')
plt.grid(True)
plt.legend()
plt.show()

# Calculate drag at typical forward speed
print("\n--- Drag Force Estimates ---")
for v in horizontal_speeds:
    drag_force = 0.5 * air_density * drag_coefficient * frontal_area * (v ** 2)
    print(f"At {v:.2f} m/s ({v*2.237:.1f} mph): Drag Force = {drag_force:.2f} N")

# Estimate minimum airspeed needed for takeoff
min_required_prop_speed = math.sqrt(weight / (air_density * propeller_area * num_propellers))
print("\n--- Takeoff Requirements ---")
print(f"Minimum propeller air exit speed needed for takeoff: {min_required_prop_speed:.2f} m/s")
