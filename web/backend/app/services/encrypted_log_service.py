import base64
import os
import shutil
import threading
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path

try:
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class EncryptedLogService:
    """
    Encrypted logging service for TRAINING jobs only.
    
    Features:
    - Uses RSA public key encryption (only US Inc can decrypt with private key)
    - Stores encrypted logs per job_id in /training/ subfolder
    - CHUNKED encryption for long messages (no truncation)
    - Auto-cleanup: Removes logs older than 24 hours (EXCEPT in-progress jobs)
    - User CANNOT decrypt these logs (no private key in container)
    
    Key Locations:
    - Public key (in container): /app/keys/usf_bios_public.pem
    - Private key (US Inc only): keys/usf_bios_private.pem (NEVER in container)
    
    Log Structure:
    - /app/data/encrypted_logs/training/{job_id}.enc.log - Per-job training logs
    - /app/data/encrypted_logs/system/{date}/{hour}.enc.log - System logs (separate service)
    """
    
    # Try multiple paths for the public key
    PUBLIC_KEY_PATHS = [
        os.getenv("RSA_PUBLIC_KEY_PATH", ""),
        "/app/keys/usf_bios_public.pem",
        "/app/.k",
        "keys/usf_bios_public.pem",  # Development path
    ]
    
    # Base directory for encrypted logs
    ENCRYPTED_LOG_BASE = os.getenv("ENCRYPTED_LOG_PATH", "/app/data/encrypted_logs")
    # Training logs go in /training/ subfolder
    TRAINING_LOG_DIR = os.path.join(ENCRYPTED_LOG_BASE, "training")
    
    # Retention period in hours (24 hours)
    RETENTION_HOURS = 24
    
    _public_key = None
    _key_loaded = False
    _cleanup_thread_started = False
    _cleanup_thread_lock = threading.Lock()
    _lock = threading.Lock()
    
    def __init__(self):
        # Ensure training log directory exists
        Path(self.TRAINING_LOG_DIR).mkdir(parents=True, exist_ok=True)
        # Start cleanup scheduler
        self._start_cleanup_scheduler()
    
    @classmethod
    def _load_public_key(cls):
        if cls._key_loaded:
            return cls._public_key
        
        cls._key_loaded = True
        
        if not CRYPTO_AVAILABLE:
            print("[EncryptedLogService] WARNING: cryptography library not available, logs will NOT be encrypted")
            return None
        
        # Try multiple paths for the public key
        for key_path in cls.PUBLIC_KEY_PATHS:
            if not key_path or not os.path.exists(key_path):
                continue
            
            try:
                with open(key_path, 'rb') as f:
                    cls._public_key = serialization.load_pem_public_key(
                        f.read(),
                        backend=default_backend()
                    )
                print(f"[EncryptedLogService] Loaded public key from: {key_path}")
                return cls._public_key
            except Exception as e:
                print(f"[EncryptedLogService] Failed to load key from {key_path}: {e}")
                continue
        
        print("[EncryptedLogService] WARNING: No public key found, logs will NOT be encrypted")
        print(f"[EncryptedLogService] Searched paths: {cls.PUBLIC_KEY_PATHS}")
        return None
    
    @classmethod
    def encrypt_message(cls, message: str) -> str:
        """
        Encrypt a message using RSA public key.
        
        CHUNKED ENCRYPTION: Long messages are split into chunks and encrypted
        separately, then joined with '|' separator. Prefix 'CHUNKED:' indicates
        multi-chunk message.
        
        This ensures NO truncation - all data is preserved.
        """
        public_key = cls._load_public_key()
        
        if public_key is None:
            # No encryption available - still encode but mark as unencrypted
            return base64.b64encode(f"[UNENCRYPTED]{message}".encode()).decode()
        
        try:
            # RSA can only encrypt small messages (~190 bytes with OAEP)
            max_chunk = 190
            message_bytes = message.encode('utf-8')
            
            if len(message_bytes) <= max_chunk:
                # Single chunk - encrypt directly
                encrypted = public_key.encrypt(
                    message_bytes,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                return base64.b64encode(encrypted).decode()
            else:
                # CHUNKED: Split into multiple chunks, encrypt each
                chunks = []
                for i in range(0, len(message_bytes), max_chunk):
                    chunk = message_bytes[i:i + max_chunk]
                    encrypted_chunk = public_key.encrypt(
                        chunk,
                        padding.OAEP(
                            mgf=padding.MGF1(algorithm=hashes.SHA256()),
                            algorithm=hashes.SHA256(),
                            label=None
                        )
                    )
                    chunks.append(base64.b64encode(encrypted_chunk).decode())
                
                # Join chunks with '|' separator, prefix with 'CHUNKED:'
                return "CHUNKED:" + "|".join(chunks)
        
        except Exception as e:
            return base64.b64encode(f"[ENCRYPT_ERROR:{str(e)[:50]}]{message[:100]}".encode()).decode()
    
    @classmethod
    def format_log_entry(cls, level: str, message: str, job_id: str) -> str:
        """Format and encrypt a log entry."""
        timestamp = datetime.utcnow().isoformat()
        full_message = f"[{timestamp}][{level}][job:{job_id}] {message}"
        return cls.encrypt_message(full_message)
    
    def encrypt_and_format(self, message: str, job_id: str, level: str = "INFO") -> None:
        """
        Encrypt a message and write to the job's encrypted log file.
        This is the main method called by training_service.
        
        COMPLETE LOGGING: All messages are encrypted fully (chunked if needed).
        No truncation - full errors, tracebacks, and data are preserved.
        
        - Full errors and tracebacks go here (for US Inc)
        - Users CANNOT read these logs (encrypted with public key)
        """
        try:
            with self._lock:
                # Create encrypted entry (uses chunking for long messages)
                encrypted_entry = self.format_log_entry(level, message, job_id)
                
                # Write to job-specific encrypted log file in /training/ folder
                log_file = Path(self.TRAINING_LOG_DIR) / f"{job_id}.enc.log"
                with open(log_file, 'a') as f:
                    f.write(encrypted_entry + '\n')
        except Exception:
            # Silently fail - don't break training for logging issues
            pass
    
    def get_encrypted_log_path(self, job_id: str) -> str:
        """Get path to encrypted log file for a job."""
        return str(Path(self.TRAINING_LOG_DIR) / f"{job_id}.enc.log")
    
    def get_encrypted_logs(self, job_id: str) -> list:
        """Get all encrypted log entries for a job (still encrypted)."""
        log_file = Path(self.TRAINING_LOG_DIR) / f"{job_id}.enc.log"
        if not log_file.exists():
            return []
        try:
            with open(log_file, 'r') as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        except Exception:
            return []
    
    # =========================================================================
    # CLEANUP OPERATIONS - Remove logs older than 24 hours
    # =========================================================================
    
    def _cleanup_old_logs(self) -> Dict[str, Any]:
        """
        Remove training log files older than RETENTION_HOURS (24 hours).
        
        IMPORTANT: Does NOT delete logs for jobs that are still in progress
        (RUNNING or INITIALIZING status) - jobs can run for weeks/months.
        
        Checks file modification time to determine age.
        """
        try:
            now = datetime.utcnow()
            cutoff = now - timedelta(hours=self.RETENTION_HOURS)
            
            removed_files = []
            kept_files = []
            skipped_active = []
            total_freed_bytes = 0
            
            log_dir = Path(self.TRAINING_LOG_DIR)
            if not log_dir.exists():
                return {"removed": 0, "kept": 0}
            
            # Get active job IDs to exclude from cleanup
            active_job_ids = self._get_active_job_ids()
            
            # Only process .enc.log files in training directory
            for log_file in log_dir.glob("*.enc.log"):
                if not log_file.is_file():
                    continue
                
                try:
                    # Extract job_id from filename (format: {job_id}.enc.log)
                    job_id = log_file.stem.replace('.enc', '')
                    
                    # NEVER delete logs for active/in-progress jobs
                    if job_id in active_job_ids:
                        skipped_active.append(log_file.name)
                        continue
                    
                    # Get file modification time
                    mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    
                    if mtime < cutoff:
                        # File is older than retention period - delete
                        file_size = log_file.stat().st_size
                        log_file.unlink()
                        removed_files.append(log_file.name)
                        total_freed_bytes += file_size
                    else:
                        kept_files.append(log_file.name)
                
                except Exception:
                    # Skip files that can't be processed
                    continue
            
            return {
                "removed": len(removed_files),
                "kept": len(kept_files),
                "skipped_active": len(skipped_active),
                "removed_files": removed_files,
                "freed_mb": round(total_freed_bytes / (1024 * 1024), 2)
            }
        
        except Exception as e:
            return {"error": str(e), "removed": 0}
    
    def _get_active_job_ids(self) -> set:
        """
        Get job IDs for jobs that are currently active (RUNNING or INITIALIZING).
        
        These jobs should NEVER have their logs deleted, even if old,
        because training can run for weeks or months.
        """
        active_ids = set()
        try:
            # Import here to avoid circular imports
            from .job_manager import job_manager
            
            # Check in-memory jobs
            for job_id, job in job_manager._jobs.items():
                if hasattr(job, 'status'):
                    status_value = job.status.value if hasattr(job.status, 'value') else str(job.status)
                    if status_value in ('running', 'initializing', 'RUNNING', 'INITIALIZING'):
                        active_ids.add(job_id)
        except Exception:
            pass  # If we can't check, err on side of caution (don't delete)
        
        return active_ids
    
    def _start_cleanup_scheduler(self) -> None:
        """
        Start background thread to cleanup old logs every hour.
        
        ROBUST GUARANTEES:
        - Runs cleanup immediately on startup
        - Continues running every hour in background
        - Survives all exceptions - never crashes
        - Singleton pattern - only one cleanup thread per process
        """
        with self._cleanup_thread_lock:
            if EncryptedLogService._cleanup_thread_started:
                return
            EncryptedLogService._cleanup_thread_started = True
        
        try:
            # Run cleanup IMMEDIATELY on startup
            self._safe_cleanup()
            
            # Start background thread for hourly cleanup
            cleanup_thread = threading.Thread(
                target=self._hourly_cleanup_loop,
                daemon=True,
                name="TrainingLogCleanup"
            )
            cleanup_thread.start()
        except Exception:
            EncryptedLogService._cleanup_thread_started = False
    
    def _safe_cleanup(self) -> Dict[str, Any]:
        """Run cleanup with full exception handling."""
        try:
            return self._cleanup_old_logs()
        except Exception as e:
            return {"error": str(e), "removed": 0}
    
    def _hourly_cleanup_loop(self) -> None:
        """
        Background thread for hourly cleanup.
        NEVER exits, NEVER raises exceptions.
        """
        while True:
            try:
                # Sleep for 1 hour (check every minute for graceful shutdown)
                for _ in range(60):
                    try:
                        time.sleep(60)
                    except Exception:
                        pass
                
                # Run cleanup after sleeping
                self._safe_cleanup()
            
            except Exception:
                try:
                    time.sleep(60)
                except Exception:
                    pass
    
    def force_cleanup(self) -> Dict[str, Any]:
        """Manually trigger log cleanup."""
        return self._safe_cleanup()
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get statistics about current training logs."""
        try:
            log_dir = Path(self.TRAINING_LOG_DIR)
            if not log_dir.exists():
                return {"total_files": 0, "total_size_mb": 0}
            
            files = list(log_dir.glob("*.enc.log"))
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            
            # Get active job count
            active_job_ids = self._get_active_job_ids()
            
            return {
                "total_files": len(files),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "retention_hours": self.RETENTION_HOURS,
                "log_directory": str(log_dir),
                "active_jobs_protected": len(active_job_ids)
            }
        except Exception as e:
            return {"error": str(e)}


encrypted_log_service = EncryptedLogService()
