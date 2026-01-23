"""
Thunderbolt Traffic Monitor

Captures network traffic on Thunderbolt Bridge interface
to visualize bandwidth patterns during distributed training.

Usage:
    monitor = ThunderboltMonitor()
    monitor.start()
    # ... run your distributed workload ...
    samples = monitor.stop()
    monitor.save("traffic.json")
"""

import threading
import time
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional
from datetime import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not installed. Install with: pip install psutil")


@dataclass
class TrafficSample:
    """Single traffic measurement."""
    timestamp: float          # Relative time from start (seconds)
    absolute_time: float      # Unix timestamp
    send_bytes: int           # Total bytes sent
    recv_bytes: int           # Total bytes received
    send_rate_gbs: float      # Send rate (GB/s)
    recv_rate_gbs: float      # Receive rate (GB/s)
    total_rate_gbs: float     # Combined rate (GB/s)
    send_rate_gbps: float     # Send rate (Gbps)
    recv_rate_gbps: float     # Receive rate (Gbps)


class ThunderboltMonitor:
    """
    Monitor Thunderbolt Bridge traffic in real-time.
    
    On macOS, Thunderbolt networking appears as 'bridge0' or 
    'Thunderbolt Bridge' interface.
    """
    
    # Common Thunderbolt interface names on macOS
    THUNDERBOLT_INTERFACES = ["bridge0", "en5", "en6", "en7", "Thunderbolt Bridge"]
    
    def __init__(self, interface: Optional[str] = None, sample_rate_hz: int = 100):
        """
        Args:
            interface: Network interface name. Auto-detects if None.
            sample_rate_hz: Samples per second (default 100 = 10ms resolution)
        """
        if not PSUTIL_AVAILABLE:
            raise ImportError("psutil required: pip install psutil")
        
        self.interface = interface or self._detect_interface()
        self.sample_rate = sample_rate_hz
        self.samples: List[TrafficSample] = []
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.start_time: float = 0
        self.markers: List[dict] = []  # For annotating events (layer boundaries, etc.)
        
    def _detect_interface(self) -> str:
        """Auto-detect Thunderbolt interface."""
        available = psutil.net_io_counters(pernic=True).keys()
        
        for iface in self.THUNDERBOLT_INTERFACES:
            if iface in available:
                print(f"Detected Thunderbolt interface: {iface}")
                return iface
        
        # Fallback: show available interfaces
        print(f"Available interfaces: {list(available)}")
        print("Thunderbolt not detected. Using 'lo0' (loopback) for testing.")
        return "lo0"
    
    def start(self):
        """Start monitoring in background thread."""
        if self.running:
            return
        
        self.samples = []
        self.markers = []
        self.running = True
        self.start_time = time.perf_counter()
        
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()
        print(f"Traffic monitoring started on {self.interface} at {self.sample_rate} Hz")
    
    def stop(self) -> List[TrafficSample]:
        """Stop monitoring and return samples."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        duration = time.perf_counter() - self.start_time
        print(f"Traffic monitoring stopped. Captured {len(self.samples)} samples over {duration:.1f}s")
        return self.samples
    
    def mark(self, label: str):
        """Add a marker at current time (e.g., 'layer_start', 'allreduce')."""
        self.markers.append({
            "timestamp": time.perf_counter() - self.start_time,
            "label": label
        })
    
    def _sample_loop(self):
        """Background sampling loop."""
        interval = 1.0 / self.sample_rate
        
        # Get initial stats
        stats = psutil.net_io_counters(pernic=True).get(self.interface)
        if not stats:
            print(f"Warning: Interface {self.interface} not found")
            return
        
        last_sent = stats.bytes_sent
        last_recv = stats.bytes_recv
        last_time = time.perf_counter()
        
        while self.running:
            time.sleep(interval)
            
            stats = psutil.net_io_counters(pernic=True).get(self.interface)
            if not stats:
                continue
            
            now = time.perf_counter()
            dt = now - last_time
            
            if dt > 0:
                sent_delta = stats.bytes_sent - last_sent
                recv_delta = stats.bytes_recv - last_recv
                
                send_rate_gbs = sent_delta / dt / 1e9
                recv_rate_gbs = recv_delta / dt / 1e9
                
                sample = TrafficSample(
                    timestamp=now - self.start_time,
                    absolute_time=now,
                    send_bytes=stats.bytes_sent,
                    recv_bytes=stats.bytes_recv,
                    send_rate_gbs=send_rate_gbs,
                    recv_rate_gbs=recv_rate_gbs,
                    total_rate_gbs=send_rate_gbs + recv_rate_gbs,
                    send_rate_gbps=send_rate_gbs * 8,
                    recv_rate_gbps=recv_rate_gbs * 8
                )
                self.samples.append(sample)
            
            last_sent = stats.bytes_sent
            last_recv = stats.bytes_recv
            last_time = now
    
    def save(self, filepath: str):
        """Save samples to JSON file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        
        data = {
            "metadata": {
                "interface": self.interface,
                "sample_rate_hz": self.sample_rate,
                "num_samples": len(self.samples),
                "duration_seconds": self.samples[-1].timestamp if self.samples else 0,
                "captured_at": datetime.now().isoformat()
            },
            "markers": self.markers,
            "samples": [asdict(s) for s in self.samples]
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Traffic data saved to {filepath}")
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        if not self.samples:
            return {}
        
        send_rates = [s.send_rate_gbs for s in self.samples]
        recv_rates = [s.recv_rate_gbs for s in self.samples]
        total_rates = [s.total_rate_gbs for s in self.samples]
        
        return {
            "duration_seconds": self.samples[-1].timestamp,
            "num_samples": len(self.samples),
            "send_gbs": {
                "avg": sum(send_rates) / len(send_rates),
                "max": max(send_rates),
                "min": min(send_rates)
            },
            "recv_gbs": {
                "avg": sum(recv_rates) / len(recv_rates),
                "max": max(recv_rates),
                "min": min(recv_rates)
            },
            "total_gbs": {
                "avg": sum(total_rates) / len(total_rates),
                "max": max(total_rates),
                "min": min(total_rates)
            },
            "thunderbolt_utilization_percent": (max(total_rates) / 5.0) * 100  # TB4 = 5 GB/s
        }


def test_monitor():
    """Test the monitor locally."""
    print("Testing ThunderboltMonitor...")
    
    monitor = ThunderboltMonitor(sample_rate_hz=10)
    monitor.start()
    
    # Simulate some activity markers
    time.sleep(0.5)
    monitor.mark("layer_1_start")
    time.sleep(0.3)
    monitor.mark("allreduce_1")
    time.sleep(0.2)
    monitor.mark("layer_1_end")
    time.sleep(0.5)
    
    samples = monitor.stop()
    
    print(f"\nSummary:")
    for k, v in monitor.get_summary().items():
        print(f"  {k}: {v}")
    
    # Save test results
    os.makedirs("results/raw/test", exist_ok=True)
    monitor.save("results/raw/test/traffic_test.json")


if __name__ == "__main__":
    test_monitor()
