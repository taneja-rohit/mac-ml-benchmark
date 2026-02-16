"""
KV Cache Transfer over Thunderbolt TCP

Handles serialization, transfer, and deserialization of KV caches
between M5 (prefill) and M4 Pro (decode) over Thunderbolt bridge.

KV Cache size for Mistral-7B (32 layers, 8 KV heads, 128 dim):
  seq_len=512:  ~64 MB  → ~14ms at 4.7 GB/s
  seq_len=1024: ~128 MB → ~27ms at 4.7 GB/s
  seq_len=2048: ~256 MB → ~55ms at 4.7 GB/s
"""

import io
import socket
import struct
import time
import torch
from typing import Tuple, Dict, Any

# Network config
M5_IP = "169.254.1.1"
M4_IP = "169.254.1.2"
KV_PORT = 9999
BUFFER_SIZE = 4 * 1024 * 1024  # 4MB chunks for socket transfer


def serialize_kv_cache(past_key_values: Tuple) -> bytes:
    """
    Serialize HuggingFace past_key_values to bytes.
    
    past_key_values is a tuple of (key, value) tensors per layer.
    Each tensor shape: [batch, num_kv_heads, seq_len, head_dim]
    
    Returns raw bytes for TCP transfer.
    """
    # Move everything to CPU for serialization
    cpu_kv = []
    for layer_kv in past_key_values:
        # layer_kv is a DynamicCache or tuple of (key, value)
        if hasattr(layer_kv, 'key_cache'):
            # DynamicCache format
            cpu_kv.append((
                layer_kv.key_cache.cpu(),
                layer_kv.value_cache.cpu()
            ))
        else:
            cpu_kv.append((
                layer_kv[0].cpu(),
                layer_kv[1].cpu()
            ))
    
    buffer = io.BytesIO()
    torch.save(cpu_kv, buffer)
    return buffer.getvalue()


def serialize_kv_from_dynamic_cache(cache) -> bytes:
    """
    Serialize a DynamicCache object.
    Supports both old API (key_cache/value_cache) and new API (layers).
    """
    cpu_kv = []
    
    if hasattr(cache, 'layers') and len(cache.layers) > 0:
        # New transformers >= 4.50 API
        for layer in cache.layers:
            cpu_kv.append((
                layer.keys.cpu(),
                layer.values.cpu()
            ))
    elif hasattr(cache, 'key_cache'):
        # Old transformers API
        for layer_idx in range(len(cache.key_cache)):
            cpu_kv.append((
                cache.key_cache[layer_idx].cpu(),
                cache.value_cache[layer_idx].cpu()
            ))
    else:
        raise ValueError(f"Unknown cache format: {type(cache)}, attrs: {dir(cache)}")
    
    buffer = io.BytesIO()
    torch.save(cpu_kv, buffer)
    return buffer.getvalue()


def deserialize_kv_cache(data: bytes, device: str = "mps"):
    """
    Deserialize bytes back to a DynamicCache on target device.
    
    Returns DynamicCache with all layers moved to target device.
    """
    from transformers.cache_utils import DynamicCache
    
    buffer = io.BytesIO(data)
    cpu_kv = torch.load(buffer, weights_only=True)
    
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(cpu_kv):
        cache.update(k.to(device), v.to(device), layer_idx)
    
    return cache


def send_kv_cache(sock: socket.socket, kv_data: bytes, metadata: Dict[str, Any] = None) -> Dict[str, float]:
    """
    Send serialized KV cache over a socket with size header.
    
    Protocol:
      [4 bytes: metadata_size][metadata_json][4 bytes: kv_size][kv_data]
    
    Returns timing info.
    """
    import json
    
    timings = {}
    
    # Send metadata
    meta_bytes = json.dumps(metadata or {}).encode()
    sock.sendall(struct.pack('!I', len(meta_bytes)))
    sock.sendall(meta_bytes)
    
    # Send KV cache size
    sock.sendall(struct.pack('!Q', len(kv_data)))
    
    # Send KV cache data in chunks
    t_start = time.perf_counter()
    total_sent = 0
    mv = memoryview(kv_data)
    while total_sent < len(kv_data):
        chunk_end = min(total_sent + BUFFER_SIZE, len(kv_data))
        sent = sock.send(mv[total_sent:chunk_end])
        total_sent += sent
    t_end = time.perf_counter()
    
    timings['transfer_time_s'] = t_end - t_start
    timings['kv_size_mb'] = len(kv_data) / (1024 * 1024)
    timings['throughput_gbps'] = (len(kv_data) * 8) / (timings['transfer_time_s'] * 1e9)
    timings['throughput_gbs'] = len(kv_data) / (timings['transfer_time_s'] * 1e9)
    
    return timings


def recv_kv_cache(sock: socket.socket) -> Tuple[bytes, Dict[str, Any], Dict[str, float]]:
    """
    Receive serialized KV cache from a socket.
    
    Returns (kv_data, metadata, timings).
    """
    import json
    
    timings = {}
    
    # Receive metadata size
    meta_size_data = _recv_exact(sock, 4)
    meta_size = struct.unpack('!I', meta_size_data)[0]
    
    # Receive metadata
    meta_bytes = _recv_exact(sock, meta_size)
    metadata = json.loads(meta_bytes.decode())
    
    # Receive KV cache size
    kv_size_data = _recv_exact(sock, 8)
    kv_size = struct.unpack('!Q', kv_size_data)[0]
    
    # Receive KV cache data
    t_start = time.perf_counter()
    kv_data = _recv_exact(sock, kv_size)
    t_end = time.perf_counter()
    
    timings['transfer_time_s'] = t_end - t_start
    timings['kv_size_mb'] = kv_size / (1024 * 1024)
    timings['throughput_gbps'] = (kv_size * 8) / (timings['transfer_time_s'] * 1e9)
    timings['throughput_gbs'] = kv_size / (timings['transfer_time_s'] * 1e9)
    
    return kv_data, metadata, timings


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    """Receive exactly `size` bytes from socket."""
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(min(BUFFER_SIZE, size - len(data)))
        if not chunk:
            raise ConnectionError(f"Connection closed, received {len(data)}/{size} bytes")
        data.extend(chunk)
    return bytes(data)


class KVServer:
    """TCP server for sending KV cache (runs on prefill node / M5)."""
    
    def __init__(self, host: str = M5_IP, port: int = KV_PORT):
        self.host = host
        self.port = port
        self.sock = None
    
    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Set large send buffer for Thunderbolt throughput
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)
        except OSError:
            pass  # Use OS default if can't set
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print(f"[KV Server] Listening on {self.host}:{self.port}")
    
    def wait_for_client(self) -> socket.socket:
        conn, addr = self.sock.accept()
        try:
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)
        except OSError:
            pass
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[KV Server] Client connected from {addr}")
        return conn
    
    def close(self):
        if self.sock:
            self.sock.close()


class KVClient:
    """TCP client for receiving KV cache (runs on decode node / M4 Pro)."""
    
    def __init__(self, server_host: str = M5_IP, port: int = KV_PORT):
        self.server_host = server_host
        self.port = port
        self.sock = None
    
    def connect(self, timeout: float = 30.0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
        except OSError:
            pass
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(timeout)
        print(f"[KV Client] Connecting to {self.server_host}:{self.port}...")
        self.sock.connect((self.server_host, self.port))
        print(f"[KV Client] Connected!")
    
    def close(self):
        if self.sock:
            self.sock.close()
