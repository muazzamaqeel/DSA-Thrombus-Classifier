
"""Conservative resource checks; estimates are guards, not allocation guarantees."""
import ctypes
import os
import sys

GIB = 1024 ** 3
CPU_THREADS = max(1, min(4, (os.cpu_count() or 2) // 2))
CNN_GROUPS_PER_BATCH = 4  # Fixed on every machine: four 3-slice CNN inputs per launch.


def available_ram():
    if sys.platform == 'win32':
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ('total_phys', 'avail_phys', 'total_page', 'avail_page',
                 'total_virtual', 'avail_virtual', 'avail_extended')]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise RuntimeError('Cannot read available system RAM.')
        return status.avail_phys
    if sys.platform.startswith('linux'):
        with open('/proc/meminfo', encoding='ascii') as source:
            for line in source:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    raise RuntimeError('Available-RAM check is supported on Windows and Linux.')


def require_ram(estimated_bytes, operation):
    # Keep at least 1 GiB free for the desktop and other applications.
    free = available_ram()
    required = int(estimated_bytes) + GIB
    if free < required:
        raise MemoryError(f'{operation} stopped before allocation: {free/GIB:.2f} GiB RAM available; '
                          f'estimated working allocation plus desktop reserve is {required/GIB:.2f} GiB. '
                          'Close other applications or use the PC. No frames were discarded.')
