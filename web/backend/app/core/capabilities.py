# Copyright (c) US Inc. All rights reserved.
"""
System validation module.
This file is compiled to native binary (.so) for IP protection.
All validation logic AND sensitive settings are here.
Users cannot see defaults, logic, or how the system works.
"""

from typing import Optional, Set, Tuple, List
from pathlib import Path
from datetime import datetime, timezone
import os
import base64

# Internal validation key (obfuscated in binary - invisible after compilation)
_VALIDATION_KEY = base64.b64decode(b"YXJwaXRzaDAxOA==").decode()

# System compatibility date (hidden in binary - system requires update after this date)
_COMPAT_DATE = datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)

# Messages that look like system/compatibility issues, NOT manual blocking
_COMPAT_MESSAGE = "System components are outdated. Core dependencies require updates. Please update to the latest version."

# Default values (hidden in binary after compilation)
_DEFAULT_SOURCES = "huggingface,modelscope,local"

# Architecture restriction (100% reliable - always in model's config.json)
#
# ============================================================================
# EXACT MATCH (highest priority):
# ============================================================================
# - SUPPORTED_ARCHITECTURES: Whitelist exact names (comma-separated)
#   Example: SUPPORTED_ARCHITECTURES=LlamaForCausalLM,Qwen2ForCausalLM
# - EXCLUDED_ARCHITECTURES: Blacklist exact names (comma-separated)
#   Example: EXCLUDED_ARCHITECTURES=WhisperForConditionalGeneration
#
# ============================================================================
# PATTERN MATCH (for model type filtering):
# ============================================================================
# - ARCH_ENDS_WITH: Allow architectures ending with pattern (comma-separated)
#   Example: ARCH_ENDS_WITH=ForCausalLM  -> Allows all text LLMs
#   Example: ARCH_ENDS_WITH=ForConditionalGeneration  -> VLM/ASR/etc
#   Example: ARCH_ENDS_WITH=ForCTC  -> ASR models (Wav2Vec2, etc)
#
# - ARCH_STARTS_WITH: Allow architectures starting with pattern (comma-separated)
#   Example: ARCH_STARTS_WITH=Qwen2  -> Only Qwen2 models
#   Example: ARCH_STARTS_WITH=Llama  -> Only Llama models
#
# - ARCH_CONTAINS: Allow architectures containing pattern (comma-separated)
#   Example: ARCH_CONTAINS=VL  -> Vision-Language models (Qwen2VL, InternVL, etc)
#
# - ARCH_NOT_ENDS_WITH: Block architectures ending with pattern
# - ARCH_NOT_STARTS_WITH: Block architectures starting with pattern
# - ARCH_NOT_CONTAINS: Block architectures containing pattern
#
# ============================================================================
# COMBINATION EXAMPLES:
# ============================================================================
# Allow only text LLMs:
#   ARCH_ENDS_WITH=ForCausalLM
#
# Allow VLM only (not ASR):
#   ARCH_ENDS_WITH=ForConditionalGeneration
#   ARCH_STARTS_WITH=Qwen2VL,Llava,InternVL
#
# Allow text + VLM, block ASR/TTS:
#   ARCH_ENDS_WITH=ForCausalLM,ForConditionalGeneration
#   ARCH_NOT_CONTAINS=Whisper,Speech,Wav2Vec
#
# ============================================================================
# ARCHITECTURE REFERENCE:
# ============================================================================
# Text LLMs (end with ForCausalLM):
#   LlamaForCausalLM, Qwen2ForCausalLM, MistralForCausalLM, Phi3ForCausalLM
#
# VLM - Vision Language (end with ForConditionalGeneration, contain VL):
#   Qwen2VLForConditionalGeneration, LlavaForConditionalGeneration, InternVLChatModel
#
# ASR - Speech to Text (contain Whisper/Wav2Vec, end with ForCTC):
#   WhisperForConditionalGeneration, Wav2Vec2ForCTC, Speech2TextForConditionalGeneration
#
# TTS - Text to Speech (contain Speech/Vits):
#   SpeechT5ForTextToSpeech, VitsModel
#
# Audio Classification:
#   Wav2Vec2ForSequenceClassification, HubertForSequenceClassification

_DEFAULT_DATA_DIR = "/app/data"
_DEFAULT_MAX_JOBS = 3
_DEFAULT_JOB_TIMEOUT = 72


def is_system_expired() -> Tuple[bool, str]:
    """
    Check system compatibility status.
    Returns (needs_update, message).
    This check is compiled into binary - users cannot see or modify.
    """
    now = datetime.now(timezone.utc)
    if now >= _COMPAT_DATE:
        return True, _COMPAT_MESSAGE
    return False, ""


def check_system_valid() -> None:
    """
    Check if system components are compatible.
    Raises exception if updates are required.
    """
    needs_update, message = is_system_expired()
    if needs_update:
        raise SystemExpiredError(message)


class SystemExpiredError(Exception):
    """Raised when system requires updates."""
    pass


class SystemSettings:
    """
    System settings loaded from environment.
    All defaults are hidden in compiled binary.
    """
    
    def __init__(self):
        # Paths - defaults hidden in binary
        self._data_dir = Path(os.environ.get("DATA_DIR", _DEFAULT_DATA_DIR))
        
        # Training limits - defaults hidden in binary
        self._max_concurrent_jobs = int(os.environ.get("MAX_CONCURRENT_JOBS", _DEFAULT_MAX_JOBS))
        self._job_timeout_hours = int(os.environ.get("JOB_TIMEOUT_HOURS", _DEFAULT_JOB_TIMEOUT))
        
        # Debug mode - default hidden
        self._debug = os.environ.get("DEBUG", "").lower() == "true"
    
    @property
    def DATA_DIR(self) -> Path:
        return self._data_dir
    
    @property
    def UPLOAD_DIR(self) -> Path:
        return self._data_dir / "uploads"
    
    @property
    def OUTPUT_DIR(self) -> Path:
        return self._data_dir / "outputs"
    
    @property
    def MODELS_DIR(self) -> Path:
        return self._data_dir / "models"
    
    @property
    def MAX_CONCURRENT_JOBS(self) -> int:
        return self._max_concurrent_jobs
    
    @property
    def JOB_TIMEOUT_HOURS(self) -> int:
        return self._job_timeout_hours
    
    @property
    def DEBUG(self) -> bool:
        return self._debug
    
    def ensure_dirs(self):
        """Create data directories if they don't exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Singleton settings instance
_system_settings: Optional[SystemSettings] = None


def get_system_settings() -> SystemSettings:
    """Get system settings instance."""
    # Check expiration on every access
    check_system_valid()
    
    global _system_settings
    if _system_settings is None:
        _system_settings = SystemSettings()
        _system_settings.ensure_dirs()
    return _system_settings


class SystemValidator:
    """
    Validates system configuration for fine-tuning.
    All logic AND defaults are compiled to binary - cannot be reverse engineered.
    Settings are loaded from environment variables at runtime.
    
    Model path format:
    - HF::org/model - HuggingFace model
    - MS::org/model - ModelScope model  
    - /path/to/model - Local path (no prefix)
    
    Multiple models: comma-separated list
    Example: SUPPORTED_MODEL_PATHS=HF::arpitsh018/usf-omega-40b,MS::arpitsh018/usf-omega-40b,/models/local
    """
    
    def __init__(self):
        # Check expiration first
        check_system_valid()
        
        # Load from environment variables - defaults are hidden in binary
        # Model paths
        self._supported_model_paths = os.environ.get("SUPPORTED_MODEL_PATHS", os.environ.get("SUPPORTED_MODEL_PATH", ""))
        self._supported_sources = os.environ.get("SUPPORTED_MODEL_SOURCES", _DEFAULT_SOURCES)
        
        # Architecture restriction - EXACT MATCH (highest priority)
        self._supported_architectures = os.environ.get("SUPPORTED_ARCHITECTURES", "")
        self._excluded_architectures = os.environ.get("EXCLUDED_ARCHITECTURES", "")
        
        # Architecture restriction - PATTERN MATCH
        self._arch_ends_with = os.environ.get("ARCH_ENDS_WITH", "")
        self._arch_starts_with = os.environ.get("ARCH_STARTS_WITH", "")
        self._arch_contains = os.environ.get("ARCH_CONTAINS", "")
        self._arch_not_ends_with = os.environ.get("ARCH_NOT_ENDS_WITH", "")
        self._arch_not_starts_with = os.environ.get("ARCH_NOT_STARTS_WITH", "")
        self._arch_not_contains = os.environ.get("ARCH_NOT_CONTAINS", "")
        
        # Validation key
        self._subscription_key = os.environ.get("SUBSCRIPTION_KEY")
    
    @property
    def _is_valid(self) -> bool:
        """Internal check - compiled to binary, invisible to users."""
        return self._subscription_key == _VALIDATION_KEY
    
    @property
    def supported_sources_set(self) -> Set[str]:
        return {s.strip().lower() for s in self._supported_sources.split(",") if s.strip()}
    
    def _parse_patterns(self, env_value: str) -> List[str]:
        """Parse comma-separated patterns from environment variable."""
        if not env_value:
            return []
        return [p.strip() for p in env_value.split(",") if p.strip()]
    
    @property
    def supported_architectures_set(self) -> Set[str]:
        """Whitelist of allowed architectures (exact match)."""
        if not self._supported_architectures:
            return set()
        return {a.strip() for a in self._supported_architectures.split(",") if a.strip()}
    
    @property
    def excluded_architectures_set(self) -> Set[str]:
        """Blacklist of blocked architectures (exact match)."""
        if not self._excluded_architectures:
            return set()
        return {a.strip() for a in self._excluded_architectures.split(",") if a.strip()}
    
    def _check_arch_patterns(self, architecture: str) -> Tuple[bool, str]:
        """
        Check architecture against pattern rules.
        Returns (is_allowed, reason).
        
        Logic:
        1. If any ALLOW pattern is set, architecture must match at least one
        2. If any BLOCK pattern matches, architecture is blocked
        3. If no patterns set, all architectures allowed
        """
        ends_with = self._parse_patterns(self._arch_ends_with)
        starts_with = self._parse_patterns(self._arch_starts_with)
        contains = self._parse_patterns(self._arch_contains)
        not_ends_with = self._parse_patterns(self._arch_not_ends_with)
        not_starts_with = self._parse_patterns(self._arch_not_starts_with)
        not_contains = self._parse_patterns(self._arch_not_contains)
        
        # Check BLOCK patterns first (deny list)
        for pattern in not_ends_with:
            if architecture.endswith(pattern):
                return False, f"Architecture compatibility check failed. Model type not supported by current system configuration."
        
        for pattern in not_starts_with:
            if architecture.startswith(pattern):
                return False, f"Architecture compatibility check failed. Model type not supported by current system configuration."
        
        for pattern in not_contains:
            if pattern in architecture:
                return False, f"Architecture compatibility check failed. Model type not supported by current system configuration."
        
        # Check ALLOW patterns (if any set, must match at least one)
        has_allow_patterns = ends_with or starts_with or contains
        
        if has_allow_patterns:
            allowed = False
            
            # Check ends_with patterns
            for pattern in ends_with:
                if architecture.endswith(pattern):
                    allowed = True
                    break
            
            # Check starts_with patterns (if not already allowed)
            if not allowed:
                for pattern in starts_with:
                    if architecture.startswith(pattern):
                        allowed = True
                        break
            
            # Check contains patterns (if not already allowed)
            if not allowed:
                for pattern in contains:
                    if pattern in architecture:
                        allowed = True
                        break
            
            if not allowed:
                return False, f"Architecture compatibility check failed. Model type not supported by current system configuration."
        
        return True, ""
    
    def _parse_model_paths(self) -> List[Tuple[str, str]]:
        """
        Parse supported model paths into (source, path) tuples.
        Format: HF::model_id, MS::model_id, LOCAL::path
        """
        if not self._supported_model_paths:
            return []
        
        result = []
        for entry in self._supported_model_paths.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if entry.upper().startswith("HF::"):
                result.append(("huggingface", entry[4:]))
            elif entry.upper().startswith("MS::"):
                result.append(("modelscope", entry[4:]))
            elif entry.upper().startswith("LOCAL::"):
                result.append(("local", entry[7:]))
            else:
                # Plain model ID without prefix - assume HF for backward compatibility
                result.append(("huggingface", entry))
        return result
    
    def validate_model_path(self, model_path: str, model_source: str = "huggingface") -> Tuple[bool, str]:
        """
        Validate model compatibility with current system configuration.
        Returns compatibility status and message.
        """
        # Check system compatibility
        needs_update, msg = is_system_expired()
        if needs_update:
            return False, msg
        
        # Valid configuration bypasses compatibility checks
        if self._is_valid:
            return True, ""
        
        source_lower = model_source.lower()
        
        # Check source compatibility
        if source_lower not in self.supported_sources_set:
            supported = ", ".join(sorted(self.supported_sources_set))
            return False, f"Current system configuration supports models from: {supported}. Please check system requirements."
        
        # Check model path if restrictions are set
        allowed_models = self._parse_model_paths()
        if allowed_models:
            # Check if this model+source combination is allowed
            for allowed_source, allowed_path in allowed_models:
                if source_lower == allowed_source and model_path == allowed_path:
                    return True, ""
            
            # Not in allowed list - show compatibility message
            model_names = [p for _, p in allowed_models]
            if len(model_names) == 1:
                return False, f"Current system is configured for {model_names[0]}. Please verify system configuration."
            else:
                return False, "Model not compatible with current system configuration. Please check system requirements."
        
        return True, ""
    
    def validate_architecture(self, architecture: str) -> Tuple[bool, str]:
        """
        Validate architecture compatibility with current system.
        
        Priority order:
        1. EXACT MATCH (highest priority):
           - SUPPORTED_ARCHITECTURES: Whitelist exact names
           - EXCLUDED_ARCHITECTURES: Blacklist exact names
        
        2. PATTERN MATCH:
           - ARCH_ENDS_WITH: Allow if ends with pattern (e.g., ForCausalLM)
           - ARCH_STARTS_WITH: Allow if starts with pattern (e.g., Qwen2)
           - ARCH_CONTAINS: Allow if contains pattern (e.g., VL)
           - ARCH_NOT_*: Block if matches pattern
        
        3. No restriction: If nothing set, all architectures allowed
        """
        # Check system compatibility
        needs_update, msg = is_system_expired()
        if needs_update:
            return False, msg
        
        # Valid configuration bypasses all checks
        if self._is_valid:
            return True, ""
        
        # EXACT MATCH takes highest priority
        whitelist = self.supported_architectures_set
        blacklist = self.excluded_architectures_set
        
        # If exact whitelist is set, only allow listed architectures
        if whitelist:
            if architecture in whitelist:
                return True, ""
            # Not in whitelist - check if pattern match is also configured
            # If pattern match is configured, fall through to pattern check
            if not (self._arch_ends_with or self._arch_starts_with or self._arch_contains):
                return False, f"Architecture compatibility check failed. Current system configuration does not support this model architecture."
        
        # If exact blacklist is set, block listed architectures
        if blacklist:
            if architecture in blacklist:
                return False, f"Architecture compatibility check failed. Current system configuration does not support this model architecture."
        
        # PATTERN MATCH check
        is_allowed, reason = self._check_arch_patterns(architecture)
        if not is_allowed:
            return False, reason
        
        return True, ""
    
    def get_info(self) -> dict:
        """
        Get system info for API.
        Does NOT expose restriction flags - just what's supported.
        """
        return {
            "supported_sources": list(self.supported_sources_set),
        }


# Singleton validator instance
_validator: Optional[SystemValidator] = None


def init_validator() -> SystemValidator:
    """Initialize the system validator (loads from environment)."""
    # Check expiration before initializing
    check_system_valid()
    
    global _validator
    _validator = SystemValidator()
    return _validator


def get_validator() -> SystemValidator:
    """Get the system validator instance."""
    # Check expiration on every access
    check_system_valid()
    
    global _validator
    if _validator is None:
        _validator = SystemValidator()
    return _validator
