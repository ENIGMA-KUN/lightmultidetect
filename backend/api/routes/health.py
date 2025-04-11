from fastapi import APIRouter, Depends
from pydantic import BaseModel
import psutil
import time

from backend.core.security import get_current_admin_user
from backend.models.user import User

router = APIRouter()


class HealthStatus(BaseModel):
    """
    API health status
    """
    status: str
    version: str
    uptime: float
    cpu_usage: float
    memory_usage: float


@router.get("/", response_model=HealthStatus)
async def health_check():
    """
    Check API health status
    
    Returns:
        API health status
    """
    # Get system metrics
    cpu_usage = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    memory_usage = memory.percent
    
    # Calculate uptime
    # In a real application, you would track the actual start time
    uptime = 0.0
    
    return HealthStatus(
        status="healthy",
        version="1.0.0",
        uptime=uptime,
        cpu_usage=cpu_usage,
        memory_usage=memory_usage
    )


class DetailedHealthStatus(HealthStatus):
    """
    Detailed API health status
    """
    disk_usage: float
    network_io: dict
    num_threads: int
    open_files: int


@router.get("/detailed", response_model=DetailedHealthStatus)
async def detailed_health_check(current_user: User = Depends(get_current_admin_user)):
    """
    Check detailed API health status (admin only)
    
    Args:
        current_user: Current authenticated admin user
    
    Returns:
        Detailed API health status
    """
    # Get system metrics
    cpu_usage = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    memory_usage = memory.percent
    disk = psutil.disk_usage('/')
    disk_usage = disk.percent
    network = psutil.net_io_counters()
    network_io = {
        "bytes_sent": network.bytes_sent,
        "bytes_recv": network.bytes_recv,
        "packets_sent": network.packets_sent,
        "packets_recv": network.packets_recv,
    }
    num_threads = len(psutil.Process().threads())
    open_files = len(psutil.Process().open_files())
    
    # Calculate uptime
    # In a real application, you would track the actual start time
    uptime = 0.0
    
    return DetailedHealthStatus(
        status="healthy",
        version="1.0.0",
        uptime=uptime,
        cpu_usage=cpu_usage,
        memory_usage=memory_usage,
        disk_usage=disk_usage,
        network_io=network_io,
        num_threads=num_threads,
        open_files=open_files
    ) 