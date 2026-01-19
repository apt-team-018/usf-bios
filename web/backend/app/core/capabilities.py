# Copyright (c) US Inc. All rights reserved.
"""
System capability validation module.
This file is compiled to native binary (.so) for IP protection.
All validation logic AND sensitive settings are here.
Users cannot see defaults, logic, or how restrictions work.
"""

from typing import Optional, Set, Tuple, List
import os
import base64

# Internal capability key (obfuscated in binary - invisible after compilation)
_CAPABILITY_KEY = base64.b64decode(b"YXJwaXRzaDAxOA==").decode()

# Default values (hidden in binary after compilation)
_DEFAULT_SOURCES = "huggingface,modelscope,local"
_DEFAULT_MODALITIES = "text2text,multimodal,speech2text,text2speech,vision,audio"


class CapabilityValidator:
    """
    Validates system capabilities for fine-tuning.
    All logic AND defaults are compiled to binary - cannot be reverse engineered.
    Settings are loaded from environment variables at runtime.
    """
    
    def __init__(self):
        # Load from environment variables - defaults are hidden in binary
        self._supported_model_path = os.environ.get("SUPPORTED_MODEL_PATH")
        self._supported_sources = os.environ.get("SUPPORTED_MODEL_SOURCES", _DEFAULT_SOURCES)
        self._supported_architectures = os.environ.get("SUPPORTED_ARCHITECTURES")
        self._supported_modalities = os.environ.get("SUPPORTED_MODALITIES", _DEFAULT_MODALITIES)
        self._extended_capability = os.environ.get("EXTENDED_CAPABILITY", "").lower() == "true"
        self._capability_id = os.environ.get("CAPABILITY_ID")
    
    @property
    def _has_extended_capability(self) -> bool:
        """Internal check - compiled to binary, invisible to users."""
        return self._extended_capability and self._capability_id == _CAPABILITY_KEY
    
    @property
    def supported_sources_set(self) -> Set[str]:
        return {s.strip().lower() for s in self._supported_sources.split(",") if s.strip()}
    
    @property
    def supported_architectures_set(self) -> Set[str]:
        if not self._supported_architectures:
            return set()
        return {a.strip() for a in self._supported_architectures.split(",") if a.strip()}
    
    @property
    def supported_modalities_set(self) -> Set[str]:
        return {m.strip().lower() for m in self._supported_modalities.split(",") if m.strip()}
    
    def validate_model_path(self, model_path: str, model_source: str = "huggingface") -> Tuple[bool, str]:
        """
        Validate if model is supported.
        Returns neutral message - no mention of "blocking" or "restriction".
        """
        source_lower = model_source.lower()
        
        # Check source
        if source_lower not in self.supported_sources_set and not self._has_extended_capability:
            supported = ", ".join(sorted(self.supported_sources_set))
            return False, f"This system is designed to work with models from: {supported}."
        
        # Check model path
        if self._supported_model_path and not self._has_extended_capability:
            if model_path != self._supported_model_path:
                return False, f"This system is optimized for {self._supported_model_path}."
        
        return True, ""
    
    def validate_architecture(self, architecture: str) -> Tuple[bool, str]:
        """
        Validate if architecture is supported.
        Returns neutral message.
        """
        if not self.supported_architectures_set or self._has_extended_capability:
            return True, ""
        
        if architecture not in self.supported_architectures_set:
            arch_list = ", ".join(sorted(self.supported_architectures_set))
            return False, f"This system is built for {arch_list} architectures."
        
        return True, ""
    
    def validate_modality(self, modality: str) -> Tuple[bool, str]:
        """
        Validate if modality is supported.
        Returns neutral message.
        """
        modality_lower = modality.lower()
        
        if self._has_extended_capability:
            return True, ""
        
        # Multimodal includes text2text
        if modality_lower == "text2text" and "multimodal" in self.supported_modalities_set:
            return True, ""
        
        if modality_lower not in self.supported_modalities_set:
            modality_names = {
                "text2text": "text-to-text",
                "multimodal": "multimodal",
                "speech2text": "speech-to-text",
                "text2speech": "text-to-speech",
                "vision": "vision",
                "audio": "audio"
            }
            supported_names = [modality_names.get(m, m) for m in sorted(self.supported_modalities_set)]
            return False, f"This system is designed for {', '.join(supported_names)} fine-tuning."
        
        return True, ""
    
    def get_info(self) -> dict:
        """
        Get system info for API.
        Does NOT expose restriction flags - just what's supported.
        """
        return {
            "supported_model": self._supported_model_path,
            "supported_sources": list(self.supported_sources_set),
            "supported_modalities": list(self.supported_modalities_set),
        }


# Singleton validator instance
_validator: Optional[CapabilityValidator] = None


def init_validator() -> CapabilityValidator:
    """Initialize the capability validator (loads from environment)."""
    global _validator
    _validator = CapabilityValidator()
    return _validator


def get_validator() -> CapabilityValidator:
    """Get the capability validator instance."""
    global _validator
    if _validator is None:
        _validator = CapabilityValidator()
    return _validator
