import torch
import time
from codecarbon import EmissionsTracker

class EnergyTracker:
    def __init__(self, project_name="centralized_baseline"):
        self.tracker = EmissionsTracker(project_name=project_name, log_level='warning')
        self.start_time = None
        self.total_flops = 0
    
    def start(self):
        self.start_time = time.time()
        self.tracker.start()
    
    def stop(self):
        emissions = self.tracker.stop()
        elapsed_time = time.time() - self.start_time
        
        return {
            'emissions_kg': emissions,
            'time_seconds': elapsed_time,
            'time_minutes': elapsed_time / 60
        }