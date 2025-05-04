import smbus2
import time

# MPU6050 constants
MPU6050_ADDRESS = 0x68
PWR_MGMT_1 = 0x6B
GYRO_XOUT_H = 0x43

bus = smbus2.SMBus(1)
bus.write_byte_data(MPU6050_ADDRESS, PWR_MGMT_1, 0)  # Wake MPU6050

def read_word(register):
    high = bus.read_byte_data(MPU6050_ADDRESS, register)
    low = bus.read_byte_data(MPU6050_ADDRESS, register + 1)
    value = (high << 8) + low
    if value >= 0x8000:
        value -= 65536
    return value

def get_gyro_data():
    gx = read_word(GYRO_XOUT_H) / 131.0
    gy = read_word(GYRO_XOUT_H + 2) / 131.0
    gz = read_word(GYRO_XOUT_H + 4) / 131.0
    return gx, gy, gz

# PID controller
class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.previous_error = 0

    def calculate(self, setpoint, current, dt):
        error = setpoint - current
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

# Initialize PID controllers
roll_pid = PID(0.1, 0.01, 0.05)
pitch_pid = PID(0.1, 0.01, 0.05)
yaw_pid = PID(0.1, 0.01, 0.05)

# Target orientations (in degrees per second)
target_roll = 0.0
target_pitch = 0.0
target_yaw = 0.0

# Base motor speed (placeholder value)
base_speed = 1500

# Main control loop
dt = 0.01  # 10ms loop = 100Hz
try:
    while True:
        gyro_x, gyro_y, gyro_z = get_gyro_data()

        roll_output = roll_pid.calculate(target_roll, gyro_x, dt)
        pitch_output = pitch_pid.calculate(target_pitch, gyro_y, dt)
        yaw_output = yaw_pid.calculate(target_yaw, gyro_z, dt)

        # Mix motor outputs (FL, FR, BL, BR)
        motor_fl = base_speed + pitch_output - roll_output + yaw_output
        motor_fr = base_speed + pitch_output + roll_output - yaw_output
        motor_bl = base_speed - pitch_output - roll_output - yaw_output
        motor_br = base_speed - pitch_output + roll_output + yaw_output

        # Here you would write PWM values to your ESCs
        print(f"Motors FL:{motor_fl:.1f}, FR:{motor_fr:.1f}, BL:{motor_bl:.1f}, BR:{motor_br:.1f}")

        time.sleep(dt)

except KeyboardInterrupt:
    print("Flight control loop stopped.")
