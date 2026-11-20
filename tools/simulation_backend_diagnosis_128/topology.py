"""Read-only Windows CPU topology and calling-thread placement.

API/layout references:
https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getsystemcpusetinformation
https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-system_cpu_set_information
https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getcurrentprocessornumber

EfficiencyClass is relative: higher means faster and less power-efficient.
Current processor is a momentary sample, not a residency or frequency measure.
No affinity, priority, process state or experimental output is changed here.
"""
from __future__ import annotations

import ctypes
import os


_DWORD = ctypes.c_uint32
_WORD = ctypes.c_uint16
_BYTE = ctypes.c_uint8
_ERROR_INSUFFICIENT_BUFFER = 122
_CPU_SET_INFORMATION = 0
_MAXIMUM_BUFFER_BYTES = 1024 * 1024


class _CpuSet(ctypes.Structure):
    # AllFlags overlays the documented bitfield; Reserved overlays
    # SchedulingClass. Neither field is interpreted by this read-only helper.
    _fields_ = [
        ("Id", _DWORD), ("Group", _WORD),
        ("LogicalProcessorIndex", _BYTE), ("CoreIndex", _BYTE),
        ("LastLevelCacheIndex", _BYTE), ("NumaNodeIndex", _BYTE),
        ("EfficiencyClass", _BYTE), ("AllFlags", _BYTE),
        ("Reserved", _DWORD), ("AllocationTag", ctypes.c_uint64),
    ]


class _InformationUnion(ctypes.Union):
    _fields_ = [("CpuSet", _CpuSet)]


class _Information(ctypes.Structure):
    _fields_ = [("Size", _DWORD), ("Type", _DWORD), ("Data", _InformationUnion)]


def _kernel():
    if os.name != "nt":
        raise OSError("CPU-set topology helper requires Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetSystemCpuSetInformation.argtypes = [
        ctypes.c_void_p, _DWORD, ctypes.POINTER(_DWORD), ctypes.c_void_p, _DWORD]
    kernel.GetSystemCpuSetInformation.restype = ctypes.c_int
    kernel.GetCurrentProcessorNumber.argtypes = []
    kernel.GetCurrentProcessorNumber.restype = _DWORD
    return kernel


_KERNEL = None


def _api():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _kernel()
    return _KERNEL


def cpu_sets():
    """Query the system once, allowing bounded retries for topology changes.

    Records may grow in later Windows versions, so advance by each record's
    declared Size, never by sizeof the current documented structure. Unknown
    record types are skipped. Malformed/truncated/duplicate mappings fail.
    """
    api = _api()
    if ctypes.sizeof(_Information) != 32 or _Information.Data.offset != 8:
        raise RuntimeError("Unexpected Windows CPU-set structure layout")
    length = _DWORD()
    ok = api.GetSystemCpuSetInformation(None, 0, ctypes.byref(length), None, 0)
    error = ctypes.get_last_error()
    if not ok and error != _ERROR_INSUFFICIENT_BUFFER:
        raise ctypes.WinError(error)
    if length.value == 0:
        return []
    for _ in range(3):
        capacity = length.value
        if not 8 <= capacity <= _MAXIMUM_BUFFER_BYTES:
            raise ValueError("Invalid or excessive CPU-set buffer size")
        buffer = ctypes.create_string_buffer(capacity)
        ok = api.GetSystemCpuSetInformation(buffer, capacity, ctypes.byref(length), None, 0)
        error = ctypes.get_last_error()
        if ok:
            break
        if error != _ERROR_INSUFFICIENT_BUFFER or length.value <= capacity:
            raise ctypes.WinError(error)
    else:
        raise RuntimeError("CPU-set topology changed during all bounded query attempts")
    if length.value > capacity:
        raise ValueError("CPU-set result exceeds allocated buffer")
    rows, offset = [], 0
    while offset < length.value:
        remaining = length.value - offset
        if remaining < 8:
            raise ValueError("Truncated CPU-set record header")
        size = _DWORD.from_buffer_copy(buffer, offset).value
        kind = _DWORD.from_buffer_copy(buffer, offset + 4).value
        if size < 8 or size > remaining:
            raise ValueError("Invalid CPU-set record bounds")
        if kind == _CPU_SET_INFORMATION:
            if size < ctypes.sizeof(_Information):
                raise ValueError("Truncated CPU-set information record")
            cpu = _Information.from_buffer_copy(buffer, offset).Data.CpuSet
            rows.append({"id": int(cpu.Id), "group": int(cpu.Group),
                         "logical_processor_index": int(cpu.LogicalProcessorIndex),
                         "core_index": int(cpu.CoreIndex),
                         "efficiency_class": int(cpu.EfficiencyClass)})
        offset += size
    if len({row["id"] for row in rows}) != len(rows) or len({
            (row["group"], row["logical_processor_index"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate CPU-set mapping")
    return sorted(rows, key=lambda row: (row["group"], row["logical_processor_index"]))


def current_processor():
    """Return the calling thread's current group-relative logical processor.

    This host has one processor group. Multi-group callers must instead use
    GetCurrentProcessorNumberEx before joining samples to a topology mapping.
    """
    return int(_api().GetCurrentProcessorNumber())
