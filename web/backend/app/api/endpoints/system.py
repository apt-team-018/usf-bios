"""System metrics and status endpoints"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Literal, Optional
import os
import subprocess

from ...core.config import settings
from ...core.capabilities import get_validator, is_system_expired, SystemExpiredError

router = APIRouter(prefix="/system", tags=["system"])


class SystemStatus(BaseModel):
    """System status response model"""
    status: Literal["live", "starting", "degraded", "offline", "error"]
    message: str
    gpu_available: bool
    gpu_name: Optional[str] = None
    cuda_available: bool
    bios_installed: bool
    backend_ready: bool
    details: dict


def check_bios_installation() -> tuple[bool, str]:
    """Check if USF BIOS training system is properly installed"""
    try:
        # Check if core packages are available
        import transformers
        import peft
        import trl
        import accelerate
        import datasets
        
        # CRITICAL: Check if usf_bios module can be imported
        # This catches missing dependencies like json_repair
        try:
            import usf_bios
        except ImportError as e:
            return False, f"USF BIOS module error: {str(e)}"
        except Exception as e:
            return False, f"USF BIOS initialization error: {str(e)}"
        
        return True, "BIOS training packages installed"
    except ImportError as e:
        return False, f"Missing required packages: {str(e)}"
    except Exception as e:
        return False, f"Installation check failed: {str(e)}"


def check_gpu_availability() -> tuple[bool, str, Optional[str]]:
    """Check if GPU is available and functional"""
    gpu_name = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            # Test GPU is actually usable
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            return True, f"GPU ready: {gpu_name}", gpu_name
        else:
            return False, "CUDA not available", None
    except Exception as e:
        return False, "GPU check failed", None


def check_cuda_available() -> bool:
    """Check if CUDA is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


def get_gpu_metrics():
    """
    Get GPU metrics using pynvml (NVIDIA Management Library).
    Returns None for unavailable metrics to avoid showing wrong data.
    
    IMPORTANT: We use pynvml for accurate GPU metrics because:
    - torch.cuda.memory_allocated() only shows PyTorch's allocation, not actual GPU usage
    - nvidia-smi subprocess is slower and less reliable
    - pynvml gives real-time accurate data directly from NVIDIA driver
    """
    result = {
        "gpu_utilization": None,
        "gpu_memory_used": None,
        "gpu_memory_total": None,
        "gpu_temperature": None,
        "gpu_available": False,
        "metrics_source": "none"
    }
    
    try:
        import pynvml
        pynvml.nvmlInit()
        
        # Get device count to ensure we have GPUs
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            pynvml.nvmlShutdown()
            return result
        
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        result["gpu_available"] = True
        result["metrics_source"] = "pynvml"
        
        # GPU Utilization (0-100%)
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            # Validate: utilization should be 0-100
            if 0 <= util.gpu <= 100:
                result["gpu_utilization"] = int(util.gpu)
            else:
                result["gpu_utilization"] = None  # Invalid value
        except pynvml.NVMLError:
            pass  # Keep as None
        
        # GPU Memory (in GB)
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # Validate: memory values should be positive
            if mem_info.used >= 0 and mem_info.total > 0:
                result["gpu_memory_used"] = round(mem_info.used / (1024**3), 2)
                result["gpu_memory_total"] = round(mem_info.total / (1024**3), 2)
                # Sanity check: used should not exceed total
                if result["gpu_memory_used"] > result["gpu_memory_total"]:
                    result["gpu_memory_used"] = None
                    result["gpu_memory_total"] = None
        except pynvml.NVMLError:
            pass  # Keep as None
        
        # GPU Temperature (in Celsius)
        try:
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            # Validate: reasonable temperature range (0-120°C)
            # GPUs typically operate between 30-90°C, throttle around 83-90°C
            if 0 <= temperature <= 120:
                result["gpu_temperature"] = int(temperature)
            else:
                result["gpu_temperature"] = None  # Unrealistic value
        except pynvml.NVMLError:
            pass  # Keep as None
        
        pynvml.nvmlShutdown()
        return result
        
    except ImportError:
        # pynvml not available - try nvidia-smi as fallback
        try:
            import subprocess
            # Use nvidia-smi for metrics
            cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits"
            output = subprocess.check_output(cmd.split(), timeout=5).decode().strip()
            values = output.split(',')
            
            if len(values) >= 4:
                result["gpu_available"] = True
                result["metrics_source"] = "nvidia-smi"
                
                # Parse utilization
                try:
                    util = int(values[0].strip())
                    if 0 <= util <= 100:
                        result["gpu_utilization"] = util
                except (ValueError, IndexError):
                    pass
                
                # Parse memory used (MiB to GB)
                try:
                    mem_used = float(values[1].strip()) / 1024
                    if mem_used >= 0:
                        result["gpu_memory_used"] = round(mem_used, 2)
                except (ValueError, IndexError):
                    pass
                
                # Parse memory total (MiB to GB)
                try:
                    mem_total = float(values[2].strip()) / 1024
                    if mem_total > 0:
                        result["gpu_memory_total"] = round(mem_total, 2)
                except (ValueError, IndexError):
                    pass
                
                # Parse temperature
                try:
                    temp = int(values[3].strip())
                    if 0 <= temp <= 120:
                        result["gpu_temperature"] = temp
                except (ValueError, IndexError):
                    pass
                
                return result
        except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Last resort: try torch for basic info (limited accuracy)
        try:
            import torch
            if torch.cuda.is_available():
                result["gpu_available"] = True
                result["metrics_source"] = "torch"
                # Note: torch only gives total memory, not actual usage
                result["gpu_memory_total"] = round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
                )
                # torch.cuda.memory_allocated() only shows PyTorch allocations
                # This is NOT accurate for total GPU usage, so we don't use it
        except:
            pass
        
        return result
        
    except Exception as e:
        # Log error but don't expose internal details
        import logging
        logging.warning(f"GPU metrics error: {e}")
        return result


def get_cpu_metrics():
    """
    Get CPU and RAM metrics using psutil.
    Returns None for unavailable metrics to avoid showing wrong data.
    """
    result = {
        "cpu_percent": None,
        "ram_used": None,
        "ram_total": None,
        "cpu_available": False
    }
    
    try:
        import psutil
        result["cpu_available"] = True
        
        # CPU Utilization (0-100%)
        # interval=0.1 gives a quick snapshot, but may be less accurate
        # For better accuracy, consider using interval=1.0 but it blocks
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            # Validate: should be 0-100
            if 0 <= cpu_percent <= 100:
                result["cpu_percent"] = round(cpu_percent, 1)
        except Exception:
            pass
        
        # RAM Usage
        try:
            mem = psutil.virtual_memory()
            # Validate: memory values should be positive
            if mem.used >= 0 and mem.total > 0:
                ram_used = mem.used / (1024**3)  # GB
                ram_total = mem.total / (1024**3)  # GB
                # Sanity check: used should not exceed total
                if ram_used <= ram_total:
                    result["ram_used"] = round(ram_used, 2)
                    result["ram_total"] = round(ram_total, 2)
        except Exception:
            pass
        
        return result
        
    except ImportError:
        # psutil not available
        return result
    except Exception as e:
        import logging
        logging.warning(f"CPU metrics error: {e}")
        return result


@router.get("/metrics")
async def get_system_metrics():
    """Get comprehensive system metrics including GPU and CPU"""
    gpu_metrics = get_gpu_metrics()
    cpu_metrics = get_cpu_metrics()
    
    return {
        **gpu_metrics,
        **cpu_metrics
    }


@router.get("/gpu")
async def get_gpu_info():
    """Get GPU-specific information"""
    return get_gpu_metrics()


@router.get("/gpus")
async def get_all_gpus():
    """
    Get list of all available GPUs with their details.
    Used by frontend to show available GPU options for selection.
    Returns device_count and list of GPUs with id, name, memory info.
    """
    result = {
        "available": False,
        "device_count": 0,
        "gpus": []
    }
    
    try:
        import pynvml
        pynvml.nvmlInit()
        
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            pynvml.nvmlShutdown()
            return result
        
        result["available"] = True
        result["device_count"] = device_count
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Get GPU name
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            
            # Get memory info
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_total_gb = round(mem_info.total / (1024**3), 1)
                memory_used_gb = round(mem_info.used / (1024**3), 1)
                memory_free_gb = round(mem_info.free / (1024**3), 1)
            except:
                memory_total_gb = None
                memory_used_gb = None
                memory_free_gb = None
            
            # Get utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization = util.gpu
            except:
                utilization = None
            
            result["gpus"].append({
                "id": i,
                "name": name,
                "memory_total_gb": memory_total_gb,
                "memory_used_gb": memory_used_gb,
                "memory_free_gb": memory_free_gb,
                "utilization": utilization
            })
        
        pynvml.nvmlShutdown()
        return result
        
    except ImportError:
        # pynvml not available - try torch
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                result["available"] = True
                result["device_count"] = device_count
                
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    result["gpus"].append({
                        "id": i,
                        "name": props.name,
                        "memory_total_gb": round(props.total_memory / (1024**3), 1),
                        "memory_used_gb": None,
                        "memory_free_gb": None,
                        "utilization": None
                    })
        except:
            pass
        
        return result
    except Exception:
        return result


@router.get("/cpu")
async def get_cpu_info():
    """Get CPU and memory information"""
    return get_cpu_metrics()


@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    """
    Get comprehensive system status for frontend display.
    Returns status: live, starting, degraded, offline, error
    Frontend should block job submission unless status is 'live'
    """
    details = {}
    
    # Check system expiration first
    expired, exp_msg = is_system_expired()
    if expired:
        return SystemStatus(
            status="error",
            message=exp_msg,
            gpu_available=False,
            gpu_name=None,
            cuda_available=False,
            bios_installed=False,
            backend_ready=False,
            details={"error": exp_msg}
        )
    
    # Check BIOS installation
    bios_ok, bios_msg = check_bios_installation()
    details["bios"] = bios_msg
    
    # Check GPU
    gpu_ok, gpu_msg, gpu_name = check_gpu_availability()
    details["gpu"] = gpu_msg
    
    # Check CUDA
    cuda_ok = check_cuda_available()
    details["cuda"] = "Available" if cuda_ok else "Not available"
    
    # Backend is ready if we got this far
    backend_ready = True
    details["backend"] = "Running"
    
    # Determine overall status
    if bios_ok and gpu_ok and cuda_ok:
        status = "live"
        message = "System fully operational - Ready for training"
    elif bios_ok and not gpu_ok:
        status = "degraded"
        message = "GPU not available - Training will fail"
    elif not bios_ok and gpu_ok:
        status = "degraded"
        message = "BIOS packages not installed - Training will fail"
    elif not bios_ok and not gpu_ok:
        status = "offline"
        message = "System not ready - Missing GPU and BIOS packages"
    else:
        status = "error"
        message = "Unknown system state"
    
    return SystemStatus(
        status=status,
        message=message,
        gpu_available=gpu_ok,
        gpu_name=gpu_name,
        cuda_available=cuda_ok,
        bios_installed=bios_ok,
        backend_ready=backend_ready,
        details=details
    )


@router.get("/ready")
async def readiness_check():
    """
    Simple readiness check for job submission.
    Returns {"ready": true/false, "reason": "..."}
    """
    # Check expiration first
    expired, exp_msg = is_system_expired()
    if expired:
        return {"ready": False, "reason": exp_msg}
    
    bios_ok, bios_msg = check_bios_installation()
    gpu_ok, gpu_msg, _ = check_gpu_availability()
    
    if bios_ok and gpu_ok:
        return {"ready": True, "reason": "System ready for training"}
    
    reasons = []
    if not bios_ok:
        reasons.append(f"BIOS: {bios_msg}")
    if not gpu_ok:
        reasons.append(f"GPU: {gpu_msg}")
    
    return {"ready": False, "reason": "; ".join(reasons)}


# =============================================================================
# System Configuration Endpoints (Hidden from API docs)
# =============================================================================

class SystemInfo(BaseModel):
    """System information - minimal, does not expose restrictions"""
    ready: bool = True


class ValidationRequest(BaseModel):
    """Request to check if configuration works with this system"""
    model_path: str
    model_source: str = "local"
    architecture: Optional[str] = None


class ValidationResponse(BaseModel):
    """Response indicating if configuration is compatible"""
    is_supported: bool
    message: Optional[str] = None


@router.get("/info", response_model=SystemInfo, include_in_schema=False)
async def get_system_info():
    """Get minimal system info (hidden from docs)."""
    return SystemInfo(ready=True)


def check_external_storage() -> dict:
    """
    Check if external storage is mounted.
    Common mount points for cloud GPU providers:
    - RunPod: /runpod-volume
    - Lambda Labs: /home/ubuntu/data
    - Vast.ai: /workspace
    - Generic: /mnt/storage, /shared
    """
    EXTERNAL_STORAGE_PATHS = [
        "/runpod-volume",
        "/workspace", 
        "/mnt/storage",
        "/shared",
        "/data/external",
    ]
    
    # Also check env var for custom storage path
    custom_path = os.getenv("EXTERNAL_STORAGE_PATH")
    if custom_path:
        EXTERNAL_STORAGE_PATHS.insert(0, custom_path)
    
    for path in EXTERNAL_STORAGE_PATHS:
        if os.path.exists(path) and os.path.isdir(path):
            # Check if writable
            try:
                test_file = os.path.join(path, ".write_test")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                return {
                    "has_external_storage": True,
                    "storage_path": path,
                    "storage_writable": True
                }
            except:
                return {
                    "has_external_storage": True,
                    "storage_path": path,
                    "storage_writable": False
                }
    
    return {
        "has_external_storage": False,
        "storage_path": None,
        "storage_writable": False
    }


@router.get("/capabilities", include_in_schema=False)
async def get_system_capabilities():
    """Capabilities including storage detection and source restrictions."""
    storage_info = check_external_storage()
    validator = get_validator()
    
    # Get supported sources - presented as "what the system supports"
    # NOT as "what is restricted" - stealth messaging
    info = validator.get_info()
    
    return {
        "ready": True,
        "supported_model_sources": info.get("supported_sources", ["local"]),
        "supported_dataset_sources": info.get("supported_dataset_sources", ["local"]),
        **storage_info
    }


@router.get("/locked-models", include_in_schema=False)
async def get_locked_models():
    """
    Get list of allowed/locked models for frontend.
    This endpoint provides the list of models that can be used for training/inference.
    Frontend should fetch this instead of hardcoding model list.
    
    Response format:
    {
        "is_locked": true/false,
        "models": [
            {
                "name": "USF Omega",
                "path": "/workspace/models/usf_omega",
                "source": "local",
                "modality": "text",
                "description": "USF Omega Model"
            }
        ]
    }
    """
    validator = get_validator()
    
    return {
        "is_locked": validator.is_model_locked(),
        "models": validator.get_locked_models()
    }


@router.get("/output-path-config", include_in_schema=False)
async def get_output_path_config():
    """
    Get output path configuration for frontend.
    Tells frontend whether output path is locked, base-locked, or free.
    
    Response format:
    {
        "mode": "locked" | "base_locked" | "free",
        "base_path": "/workspace/output",
        "is_locked": true/false,
        "user_can_customize": true/false,
        "user_can_add_path": true/false
    }
    
    When is_locked=true: Frontend should NOT show output path input
    When user_can_add_path=true: Show input for intermediate path only
    When user_can_customize=true: Show full path input
    """
    validator = get_validator()
    return validator.get_output_path_config()


@router.get("/model-lock", include_in_schema=False)
async def get_model_capabilities_legacy():
    """Legacy endpoint - hidden from docs."""
    validator = get_validator()
    return {
        "ready": True,
        "is_locked": validator.is_model_locked(),
        "models": validator.get_locked_models()
    }


@router.post("/validate", response_model=ValidationResponse, include_in_schema=False)
async def validate_configuration(request: ValidationRequest):
    """Validate if configuration works with this system (hidden from docs)."""
    validator = get_validator()
    
    # Validate model path and source
    is_valid, message = validator.validate_model_path(request.model_path, request.model_source)
    if not is_valid:
        return ValidationResponse(is_supported=False, message=message)
    
    # Validate architecture if provided
    if request.architecture:
        is_valid, message = validator.validate_architecture(request.architecture)
        if not is_valid:
            return ValidationResponse(is_supported=False, message=message)
    
    return ValidationResponse(is_supported=True)


@router.post("/validate-config", include_in_schema=False)
async def validate_system_config(request: ValidationRequest):
    """Legacy endpoint - hidden from docs."""
    return await validate_configuration(request)


@router.post("/validate-model", include_in_schema=False)
async def validate_model_for_training(request: ValidationRequest):
    """Legacy endpoint - hidden from docs."""
    return await validate_configuration(request)
