import time
from collections import deque
from typing import Optional


class ETACalculator:
    """Calculate ETA for training progress"""

    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: Number of recent steps to average
        """
        self.window_size = window_size
        self.step_times = deque(maxlen=window_size)
        self.start_time = time.time()
        self.last_step_time = None

    def update(self):
        """Record a step completion"""
        current_time = time.time()
        if self.last_step_time is not None:
            step_duration = current_time - self.last_step_time
            self.step_times.append(step_duration)
        self.last_step_time = current_time

    def get_avg_step_time(self) -> float:
        """Get average time per step"""
        if not self.step_times:
            return 0.0
        return sum(self.step_times) / len(self.step_times)

    def get_eta(self, current_step: int, total_steps: int) -> str:
        """
        Get estimated time remaining

        Args:
            current_step: Current training step
            total_steps: Total training steps

        Returns:
            Formatted ETA string (e.g., "1h 23m 45s")
        """
        if current_step >= total_steps:
            return "0s"

        avg_step_time = self.get_avg_step_time()
        if avg_step_time == 0:
            return "calculating..."

        remaining_steps = total_steps - current_step
        remaining_seconds = int(remaining_steps * avg_step_time)

        return format_time(remaining_seconds)

    def get_elapsed_time(self) -> str:
        """Get elapsed time since start"""
        elapsed_seconds = int(time.time() - self.start_time)
        return format_time(elapsed_seconds)


def format_time(seconds: int) -> str:
    """
    Format seconds to human readable string

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s", "23m 45s", "45s")
    """
    if seconds < 60:
        return f"{seconds}s"

    minutes = seconds // 60
    seconds = seconds % 60

    if minutes < 60:
        return f"{minutes}m {seconds}s"

    hours = minutes // 60
    minutes = minutes % 60

    if hours < 24:
        return f"{hours}h {minutes}m {seconds}s"

    days = hours // 24
    hours = hours % 24
    return f"{days}d {hours}h {minutes}m"
