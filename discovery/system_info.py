"""
Hardware Discovery for Mac ML Benchmarks
Gathers detailed system information about Apple Silicon hardware.
"""

import subprocess
import platform
import json
import re
from datetime import datetime
from typing import Dict, Any
import os

def run_cmd(cmd: str) -> str:
    """Run shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def get_sysctl(key: str) -> str:
    """Get sysctl value."""
    return run_cmd(f"sysctl -n {key} 2>/dev/null")

def parse_system_profiler(data_type: str) -> Dict[str, Any]:
    """Parse system_profiler output to JSON."""
    try:
        output = run_cmd(f"system_profiler {data_type} -json")
        return json.loads(output)
    except:
        return {}

def get_chip_info() -> Dict[str, Any]:
    """Get Apple Silicon chip details."""
    info = {}
    
    # Basic chip info from sysctl
    info['brand_string'] = get_sysctl('machdep.cpu.brand_string')
    info['core_count'] = int(get_sysctl('machdep.cpu.core_count') or 0)
    info['thread_count'] = int(get_sysctl('hw.logicalcpu') or 0)
    
    # Try to parse chip name (M1, M2, M3, M4, M5, etc.)
    brand = info['brand_string']
    chip_match = re.search(r'Apple (M\d+\s*(?:Pro|Max|Ultra)?)', brand)
    info['chip_name'] = chip_match.group(1) if chip_match else brand
    
    # Performance vs efficiency cores (Apple Silicon specific)
    hw_info = parse_system_profiler('SPHardwareDataType')
    if hw_info and 'SPHardwareDataType' in hw_info:
        hw = hw_info['SPHardwareDataType'][0]
        info['model_name'] = hw.get('machine_name', 'Unknown')
        info['model_identifier'] = hw.get('machine_model', 'Unknown')
        info['chip_type'] = hw.get('chip_type', info['chip_name'])
        
        # Try to get core breakdown
        cores = hw.get('number_processors', '')
        if 'proc' in str(cores).lower():
            info['cores_description'] = cores
    
    return info

def get_memory_info() -> Dict[str, Any]:
    """Get memory information."""
    info = {}
    
    # Total physical memory
    mem_bytes = int(get_sysctl('hw.memsize') or 0)
    info['total_gb'] = mem_bytes / (1024**3)
    info['total_bytes'] = mem_bytes
    
    # Memory pressure and usage (vm_stat)
    vm_stat = run_cmd('vm_stat')
    if vm_stat:
        # Parse page size
        page_match = re.search(r'page size of (\d+) bytes', vm_stat)
        page_size = int(page_match.group(1)) if page_match else 16384
        
        # Parse free pages
        free_match = re.search(r'Pages free:\s+(\d+)', vm_stat)
        if free_match:
            free_pages = int(free_match.group(1))
            info['free_gb'] = (free_pages * page_size) / (1024**3)
    
    # Theoretical bandwidth (Apple doesn't publish, but we can estimate)
    # M1/M2/M3: ~200-400 GB/s depending on variant
    # This will be measured empirically
    info['theoretical_bandwidth_gbs'] = "TBD (will measure)"
    
    return info

def get_gpu_info() -> Dict[str, Any]:
    """Get GPU (Metal) information."""
    info = {}
    
    # Get display/GPU info
    gpu_info = parse_system_profiler('SPDisplaysDataType')
    if gpu_info and 'SPDisplaysDataType' in gpu_info:
        displays = gpu_info['SPDisplaysDataType']
        if displays:
            gpu = displays[0]
            info['gpu_name'] = gpu.get('sppci_model', 'Apple Silicon GPU')
            info['metal_family'] = gpu.get('spdisplays_mtlgpufamilysupport', 'Unknown')
            
            # Try to get GPU core count from chip name
            # This is approximate - Apple doesn't expose directly
            chip = get_sysctl('machdep.cpu.brand_string')
            
            # Rough estimates based on known chips
            gpu_cores_map = {
                'M1': 8, 'M1 Pro': 16, 'M1 Max': 32, 'M1 Ultra': 64,
                'M2': 10, 'M2 Pro': 19, 'M2 Max': 38, 'M2 Ultra': 76,
                'M3': 10, 'M3 Pro': 18, 'M3 Max': 40, 'M3 Ultra': 80,
                'M4': 10, 'M4 Pro': 20, 'M4 Max': 40,
                'M5': 12, 'M5 Pro': 22, 'M5 Max': 44,  # Estimated
            }
            
            for name, cores in gpu_cores_map.items():
                if name in chip:
                    info['gpu_cores_estimate'] = cores
                    break
            else:
                info['gpu_cores_estimate'] = "Unknown"
    
    # Check MPS availability
    try:
        import torch
        info['mps_available'] = torch.backends.mps.is_available()
        info['mps_built'] = torch.backends.mps.is_built()
    except ImportError:
        info['mps_available'] = "PyTorch not installed"
        info['mps_built'] = "PyTorch not installed"
    
    # Check MLX availability
    try:
        import mlx.core as mx
        info['mlx_available'] = True
        info['mlx_default_device'] = str(mx.default_device())
    except ImportError:
        info['mlx_available'] = False
        info['mlx_default_device'] = "MLX not installed"
    
    return info

def get_neural_engine_info() -> Dict[str, Any]:
    """Get Neural Engine information (if available)."""
    info = {}
    
    # Neural Engine cores (Apple doesn't expose directly)
    # All recent chips have 16-core Neural Engine
    chip = get_sysctl('machdep.cpu.brand_string')
    if 'M1' in chip or 'M2' in chip or 'M3' in chip or 'M4' in chip or 'M5' in chip:
        info['neural_engine_cores'] = 16
        info['neural_engine_available'] = True
    else:
        info['neural_engine_cores'] = 0
        info['neural_engine_available'] = False
    
    return info

def get_storage_info() -> Dict[str, Any]:
    """Get storage information."""
    info = {}
    
    # Disk space
    statvfs = os.statvfs('/')
    total = statvfs.f_blocks * statvfs.f_frsize
    free = statvfs.f_bavail * statvfs.f_frsize
    
    info['total_gb'] = total / (1024**3)
    info['free_gb'] = free / (1024**3)
    info['used_gb'] = (total - free) / (1024**3)
    
    # Check if SSD (always true for modern Macs)
    info['is_ssd'] = True
    
    return info

def get_network_info() -> Dict[str, Any]:
    """Get network interface information."""
    info = {}
    
    # Get active interfaces
    interfaces = run_cmd("networksetup -listallhardwareports")
    
    # WiFi info
    wifi_info = run_cmd("networksetup -getinfo Wi-Fi 2>/dev/null")
    if wifi_info and 'IP address' in wifi_info:
        ip_match = re.search(r'IP address: ([\d.]+)', wifi_info)
        info['wifi_ip'] = ip_match.group(1) if ip_match else None
    
    # Thunderbolt Bridge (for distributed)
    tb_info = run_cmd("networksetup -getinfo 'Thunderbolt Bridge' 2>/dev/null")
    if tb_info and 'IP address' in tb_info:
        ip_match = re.search(r'IP address: ([\d.]+)', tb_info)
        info['thunderbolt_ip'] = ip_match.group(1) if ip_match else None
    else:
        info['thunderbolt_ip'] = "Not configured"
    
    # Ethernet
    eth_info = run_cmd("networksetup -getinfo 'Ethernet' 2>/dev/null")
    if eth_info and 'IP address' in eth_info:
        ip_match = re.search(r'IP address: ([\d.]+)', eth_info)
        info['ethernet_ip'] = ip_match.group(1) if ip_match else None
    
    return info

def get_software_info() -> Dict[str, Any]:
    """Get relevant software versions."""
    info = {}
    
    # macOS version
    info['macos_version'] = platform.mac_ver()[0]
    info['macos_build'] = platform.mac_ver()[2]
    
    # Python version
    info['python_version'] = platform.python_version()
    
    # PyTorch version
    try:
        import torch
        info['pytorch_version'] = torch.__version__
    except ImportError:
        info['pytorch_version'] = "Not installed"
    
    # MLX version
    try:
        import mlx.core as mx
        info['mlx_version'] = getattr(mx, '__version__', 'installed (version unknown)')
    except ImportError:
        info['mlx_version'] = "Not installed"
    
    # Transformers version
    try:
        import transformers
        info['transformers_version'] = transformers.__version__
    except ImportError:
        info['transformers_version'] = "Not installed"
    
    return info

def get_system_info() -> Dict[str, Any]:
    """Get complete system information."""
    return {
        'timestamp': datetime.now().isoformat(),
        'hostname': platform.node(),
        'chip': get_chip_info(),
        'memory': get_memory_info(),
        'gpu': get_gpu_info(),
        'neural_engine': get_neural_engine_info(),
        'storage': get_storage_info(),
        'network': get_network_info(),
        'software': get_software_info(),
    }

def print_system_info(info: Dict[str, Any] = None):
    """Pretty print system information."""
    if info is None:
        info = get_system_info()
    
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box
        
        console = Console()
        
        # Header
        console.print(Panel.fit(
            f"[bold cyan]Mac ML Benchmark - Hardware Discovery[/bold cyan]\n"
            f"[dim]{info['timestamp']}[/dim]",
            box=box.DOUBLE
        ))
        
        # Chip Info
        chip_table = Table(title="🔲 Chip Information", box=box.ROUNDED)
        chip_table.add_column("Property", style="cyan")
        chip_table.add_column("Value", style="green")
        
        chip = info['chip']
        chip_table.add_row("Chip", chip.get('chip_name', 'Unknown'))
        chip_table.add_row("CPU Cores", str(chip.get('core_count', 'Unknown')))
        chip_table.add_row("Threads", str(chip.get('thread_count', 'Unknown')))
        if 'cores_description' in chip:
            chip_table.add_row("Core Config", chip['cores_description'])
        
        console.print(chip_table)
        
        # Memory Info
        mem_table = Table(title="💾 Memory Information", box=box.ROUNDED)
        mem_table.add_column("Property", style="cyan")
        mem_table.add_column("Value", style="green")
        
        mem = info['memory']
        mem_table.add_row("Total Memory", f"{mem.get('total_gb', 0):.1f} GB")
        if 'free_gb' in mem:
            mem_table.add_row("Free Memory", f"{mem.get('free_gb', 0):.1f} GB")
        mem_table.add_row("Type", "Unified Memory (shared CPU/GPU)")
        
        console.print(mem_table)
        
        # GPU Info
        gpu_table = Table(title="🎮 GPU Information", box=box.ROUNDED)
        gpu_table.add_column("Property", style="cyan")
        gpu_table.add_column("Value", style="green")
        
        gpu = info['gpu']
        gpu_table.add_row("GPU Name", gpu.get('gpu_name', 'Unknown'))
        gpu_table.add_row("GPU Cores (est.)", str(gpu.get('gpu_cores_estimate', 'Unknown')))
        gpu_table.add_row("Metal Family", gpu.get('metal_family', 'Unknown'))
        gpu_table.add_row("MPS Available", "✅" if gpu.get('mps_available') else "❌")
        gpu_table.add_row("MLX Available", "✅" if gpu.get('mlx_available') else "❌")
        
        console.print(gpu_table)
        
        # Software Info
        sw_table = Table(title="📦 Software Versions", box=box.ROUNDED)
        sw_table.add_column("Package", style="cyan")
        sw_table.add_column("Version", style="green")
        
        sw = info['software']
        sw_table.add_row("macOS", sw.get('macos_version', 'Unknown'))
        sw_table.add_row("Python", sw.get('python_version', 'Unknown'))
        sw_table.add_row("PyTorch", sw.get('pytorch_version', 'Not installed'))
        sw_table.add_row("MLX", sw.get('mlx_version', 'Not installed'))
        sw_table.add_row("Transformers", sw.get('transformers_version', 'Not installed'))
        
        console.print(sw_table)
        
        # Network Info
        net_table = Table(title="🌐 Network Interfaces", box=box.ROUNDED)
        net_table.add_column("Interface", style="cyan")
        net_table.add_column("IP Address", style="green")
        
        net = info['network']
        if net.get('wifi_ip'):
            net_table.add_row("WiFi", net['wifi_ip'])
        if net.get('ethernet_ip'):
            net_table.add_row("Ethernet", net['ethernet_ip'])
        net_table.add_row("Thunderbolt Bridge", net.get('thunderbolt_ip', 'Not configured'))
        
        console.print(net_table)
        
    except ImportError:
        # Fallback to plain print
        print("\n" + "="*60)
        print("MAC ML BENCHMARK - HARDWARE DISCOVERY")
        print("="*60)
        print(json.dumps(info, indent=2, default=str))

def save_system_info(info: Dict[str, Any], filepath: str):
    """Save system info to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(info, f, indent=2, default=str)
    print(f"System info saved to: {filepath}")


if __name__ == "__main__":
    info = get_system_info()
    print_system_info(info)
    
    # Save to file
    import os
    os.makedirs("results/raw", exist_ok=True)
    save_system_info(info, "results/raw/system_info.json")
