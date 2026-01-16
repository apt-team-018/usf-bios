"""System metrics endpoints"""
from fastapi import APIRouter

router = APIRouter(prefix="/api/system", tags=["system"])


def get_gpu_metrics():
    """Get GPU metrics using pynvml or fallback"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # Utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_utilization = util.gpu
        
        # Memory
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory_used = mem_info.used / (1024**3)  # GB
        gpu_memory_total = mem_info.total / (1024**3)  # GB
        
        # Temperature
        try:
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except:
            temperature = 0
        
        pynvml.nvmlShutdown()
        
        return {
            "gpu_utilization": gpu_utilization,
            "gpu_memory_used": round(gpu_memory_used, 2),
            "gpu_memory_total": round(gpu_memory_total, 2),
            "gpu_temperature": temperature
        }
    except ImportError:
        # pynvml not available, try torch
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return {
                    "gpu_utilization": 0,  # Not available without pynvml
                    "gpu_memory_used": round(gpu_memory_used, 2),
                    "gpu_memory_total": round(gpu_memory_total, 2),
                    "gpu_temperature": 0
                }
        except:
            pass
        
        return {
            "gpu_utilization": 0,
            "gpu_memory_used": 0,
            "gpu_memory_total": 0,
            "gpu_temperature": 0
        }
    except Exception as e:
        return {
            "gpu_utilization": 0,
            "gpu_memory_used": 0,
            "gpu_memory_total": 0,
            "gpu_temperature": 0,
            "error": str(e)
        }


def get_cpu_metrics():
    """Get CPU and RAM metrics"""
    try:
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        mem = psutil.virtual_memory()
        ram_used = mem.used / (1024**3)  # GB
        ram_total = mem.total / (1024**3)  # GB
        
        return {
            "cpu_percent": round(cpu_percent, 1),
            "ram_used": round(ram_used, 2),
            "ram_total": round(ram_total, 2)
        }
    except ImportError:
        return {
            "cpu_percent": 0,
            "ram_used": 0,
            "ram_total": 0
        }
    except Exception as e:
        return {
            "cpu_percent": 0,
            "ram_used": 0,
            "ram_total": 0,
            "error": str(e)
        }


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


@router.get("/cpu")
async def get_cpu_info():
    """Get CPU and memory information"""
    return get_cpu_metrics()
