import signal
import subprocess


class MemoryTracker:
    def start(self):
        self.process = subprocess.Popen(
            [
                "/home/sm2942/reversible_solvers/reversible_solvers/tracking/track_gpu_memory.sh",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def end(self):
        self.process.send_signal(signal.SIGINT)
        self.process.wait()
        stdout, stderr = self.process.communicate()
        peak_memory = stdout.decode().strip()
        return peak_memory
