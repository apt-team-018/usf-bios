# Copyright (c) US Inc. All rights reserved.
"""
System Guard Module - Compiled to native binary for IP protection.
Validates system license, expiration, and supported configurations.
This module is called by ALL CLI commands to enforce restrictions.
Users cannot bypass these checks even if they skip the web backend.
"""

from typing import Optional, Set, Tuple, List
from datetime import datetime, timezone
import os
import sys
import base64

# Internal subscription key (obfuscated in binary - invisible after compilation)
_SUBSCRIPTION_KEY = base64.b64decode(b"YXJwaXRzaDAxOA==").decode()

# System expiration date (hidden in binary - system stops working after this date)
_EXPIRATION_DATE = datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
_EXPIRATION_MESSAGE = "System license has expired. Please contact administration for renewal."

# Default values (hidden in binary after compilation)
_DEFAULT_SOURCES = "huggingface,modelscope,local"
_DEFAULT_MODALITIES = "text2text,multimodal,speech2text,text2speech,vision,audio"


class SystemGuardError(Exception):
    """Raised when system validation fails."""
    pass


class SystemExpiredError(SystemGuardError):
    """Raised when system license has expired."""
    pass


class ModelNotSupportedError(SystemGuardError):
    """Raised when model is not supported."""
    pass


class ArchitectureNotSupportedError(SystemGuardError):
    """Raised when architecture is not supported."""
    pass


class ModalityNotSupportedError(SystemGuardError):
    """Raised when modality is not supported."""
    pass


def _is_expired() -> Tuple[bool, str]:
    """Check if system has expired."""
    now = datetime.now(timezone.utc)
    if now >= _EXPIRATION_DATE:
        return True, _EXPIRATION_MESSAGE
    return False, ""


def _is_subscribed() -> bool:
    """Check if system has valid subscription."""
    key = os.environ.get("SUBSCRIPTION_KEY", "")
    return key == _SUBSCRIPTION_KEY


def _get_supported_model_paths() -> List[Tuple[str, str]]:
    """Get list of supported model paths as (source, path) tuples."""
    paths_str = os.environ.get("SUPPORTED_MODEL_PATHS", os.environ.get("SUPPORTED_MODEL_PATH", ""))
    if not paths_str:
        return []
    
    result = []
    for entry in paths_str.split(","):
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


def _get_supported_sources() -> Set[str]:
    """Get set of supported model sources."""
    sources = os.environ.get("SUPPORTED_MODEL_SOURCES", _DEFAULT_SOURCES)
    return {s.strip().lower() for s in sources.split(",") if s.strip()}


def _get_supported_architectures() -> Set[str]:
    """Get set of supported architectures."""
    archs = os.environ.get("SUPPORTED_ARCHITECTURES", "")
    if not archs:
        return set()
    return {a.strip() for a in archs.split(",") if a.strip()}


def _get_supported_modalities() -> Set[str]:
    """Get set of supported modalities."""
    mods = os.environ.get("SUPPORTED_MODALITIES", _DEFAULT_MODALITIES)
    return {m.strip().lower() for m in mods.split(",") if m.strip()}


def check_system_valid() -> None:
    """
    Check if system is valid. Raises SystemExpiredError if expired.
    Call this at the start of any CLI command.
    """
    expired, message = _is_expired()
    if expired:
        print(f"\n[USF BIOS] ERROR: {message}\n", file=sys.stderr)
        raise SystemExpiredError(message)


def validate_model(model_path: str, model_source: str = "huggingface") -> None:
    """
    Validate if model is supported. Raises ModelNotSupportedError if not.
    
    Args:
        model_path: Model path or HuggingFace/ModelScope ID
        model_source: Source type (huggingface, modelscope, local)
    """
    # Check expiration first
    check_system_valid()
    
    # Subscription bypasses all checks
    if _is_subscribed():
        return
    
    source_lower = model_source.lower()
    supported_sources = _get_supported_sources()
    
    # Check source
    if source_lower not in supported_sources:
        supported = ", ".join(sorted(supported_sources))
        msg = f"This system is designed to work with models from: {supported}."
        print(f"\n[USF BIOS] ERROR: {msg}\n", file=sys.stderr)
        raise ModelNotSupportedError(msg)
    
    # Check model path if restrictions are set
    allowed_models = _get_supported_model_paths()
    if allowed_models:
        for allowed_source, allowed_path in allowed_models:
            if source_lower == allowed_source and model_path == allowed_path:
                return
        
        model_names = [p for _, p in allowed_models]
        if len(model_names) == 1:
            msg = f"This system is optimized for {model_names[0]}."
        else:
            msg = "This system is designed for specific models only."
        print(f"\n[USF BIOS] ERROR: {msg}\n", file=sys.stderr)
        raise ModelNotSupportedError(msg)


def validate_architecture(architecture: str) -> None:
    """
    Validate if architecture is supported. Raises ArchitectureNotSupportedError if not.
    
    Args:
        architecture: Model architecture class name (e.g., LlamaForCausalLM)
    """
    # Check expiration first
    check_system_valid()
    
    # Subscription bypasses all checks
    if _is_subscribed():
        return
    
    supported_archs = _get_supported_architectures()
    if not supported_archs:
        return  # No architecture restriction
    
    if architecture not in supported_archs:
        arch_list = ", ".join(sorted(supported_archs))
        msg = f"This system is built for {arch_list} architectures."
        print(f"\n[USF BIOS] ERROR: {msg}\n", file=sys.stderr)
        raise ArchitectureNotSupportedError(msg)


def validate_modality(modality: str) -> None:
    """
    Validate if modality is supported. Raises ModalityNotSupportedError if not.
    
    Args:
        modality: Training modality (text2text, multimodal, speech2text, etc.)
    """
    # Check expiration first
    check_system_valid()
    
    # Subscription bypasses all checks
    if _is_subscribed():
        return
    
    modality_lower = modality.lower()
    supported_mods = _get_supported_modalities()
    
    # Multimodal includes text2text
    if modality_lower == "text2text" and "multimodal" in supported_mods:
        return
    
    if modality_lower not in supported_mods:
        modality_names = {
            "text2text": "text-to-text",
            "multimodal": "multimodal",
            "speech2text": "speech-to-text",
            "text2speech": "text-to-speech",
            "vision": "vision",
            "audio": "audio"
        }
        supported_names = [modality_names.get(m, m) for m in sorted(supported_mods)]
        msg = f"This system is designed for {', '.join(supported_names)} fine-tuning."
        print(f"\n[USF BIOS] ERROR: {msg}\n", file=sys.stderr)
        raise ModalityNotSupportedError(msg)


def validate_training_config(
    model_path: str,
    model_source: str = "huggingface",
    architecture: Optional[str] = None,
    modality: str = "text2text"
) -> None:
    """
    Validate complete training configuration.
    Call this before starting any training job.
    
    Args:
        model_path: Model path or HuggingFace/ModelScope ID
        model_source: Source type (huggingface, modelscope, local)
        architecture: Model architecture class name (optional)
        modality: Training modality
    """
    # Validate model
    validate_model(model_path, model_source)
    
    # Validate architecture if provided
    if architecture:
        validate_architecture(architecture)
    
    # Validate modality
    validate_modality(modality)


def guard_cli_entry() -> None:
    """
    Guard function to call at the start of any CLI entry point.
    Exits with error if system is expired.
    """
    try:
        check_system_valid()
    except SystemExpiredError as e:
        sys.exit(1)


# Auto-check on import (prevents any usage of expired system)
try:
    _expired, _msg = _is_expired()
    if _expired:
        print(f"\n[USF BIOS] {_msg}\n", file=sys.stderr)
except Exception:
    pass
