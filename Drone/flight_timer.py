import time

class FlightTimer:
    def __init__(self):
        self.start_time = None
        self.total_flight_time = 0.0
        self.flying = False

    def start_flight(self):
        if not self.flying:
            self.start_time = time.time()
            self.flying = True
            print("[INFO] Flight started...")

    def stop_flight(self):
        if self.flying:
            elapsed = time.time() - self.start_time
            self.total_flight_time += elapsed
            self.flying = False
            print(f"[INFO] Flight stopped. This session: {elapsed:.2f} seconds")

    def get_total_flight_time(self):
        if self.flying:
            current_flight_time = time.time() - self.start_time
            return self.total_flight_time + current_flight_time
        return self.total_flight_time


#Simulation (Main Loop)
'''
if __name__ == "__main__":
    timer = FlightTimer()
    print("Press 's' to start flight, 'e' to end flight, 'q' to quit.")

    while True:
        command = input("Command: ").strip().lower()

        if command == 's':
            timer.start_flight()
        elif command == 'e':
            timer.stop_flight()
            print(f"Total flight time: {timer.get_total_flight_time():.2f} seconds")
        elif command == 'q':
            if timer.flying:
                timer.stop_flight()
            print(f"[EXIT] Final total flight time: {timer.get_total_flight_time():.2f} seconds")
            break
        else:
            print("Unknown command. Use 's', 'e', or 'q'.")

'''