# Copyright (c) US Inc. All rights reserved.
"""Model-related endpoints"""

import os
from pathlib import Path
from typing import Optional, Literal, List

from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...models.schemas import ModelSource, ModelValidation
from ...core.database import get_db
from ...core.capabilities import get_validator
from ...services.model_registry_service import ModelRegistryService

router = APIRouter()


# Request/Response models for Model Registry
class ModelRegistration(BaseModel):
    """Request model for registering a model"""
    name: str
    source: str = "local"  # Model source
    source_id: str  # Local path to model directory
    description: Optional[str] = None
    model_type: Optional[str] = None  # llm, vlm, etc.
    model_size: Optional[str] = None  # 7B, 14B, 40B, etc.


class RegisteredModelInfo(BaseModel):
    """Registered model info"""
    id: str
    name: str
    source: str
    source_id: str
    description: Optional[str]
    model_type: Optional[str]
    model_size: Optional[str]
    times_used: int
    last_used_at: Optional[str]
    created_at: Optional[str]
    trainings_count: int


class SupportedModels(BaseModel):
    usf_models: list
    popular_models: list


@router.get("/supported", response_model=SupportedModels, include_in_schema=False)
async def get_supported_models():
    """Get list of supported models (hidden - returns empty)"""
    # Do not expose model suggestions - only local models supported
    return SupportedModels(
        usf_models=[],
        popular_models=[]
    )


@router.post("/validate", response_model=ModelValidation)
async def validate_model(
    model_path: str = Query(..., description="Path to model directory"),
    source: ModelSource = Query(ModelSource.LOCAL, description="Model source")
):
    """Validate if a model exists and is accessible"""
    try:
        # First validate source is allowed
        validator = get_validator()
        source_str = source.value if hasattr(source, 'value') else str(source)
        is_valid, msg = validator.validate_model_path(model_path, source_str)
        if not is_valid:
            return ModelValidation(valid=False, error=msg)
        
        if source == ModelSource.LOCAL:
            path = Path(model_path)
            
            # Check if path exists
            if not path.exists():
                return ModelValidation(valid=False, error="Model path does not exist. Please verify the path is correct.")
            
            # Check read permissions
            if not os.access(model_path, os.R_OK):
                return ModelValidation(valid=False, error="Cannot access model path. Please check read permissions.")
            
            # For directories, check for model files
            if path.is_dir():
                config_path = path / "config.json"
                if config_path.exists():
                    # Try to read config and get architecture
                    import json
                    try:
                        with open(config_path) as f:
                            config = json.load(f)
                        
                        model_type = config.get("model_type", "unknown")
                        architectures = config.get("architectures", [])
                        
                        # Validate architecture if available
                        if architectures and len(architectures) > 0:
                            arch_valid, arch_msg = validator.validate_architecture(architectures[0])
                            if not arch_valid:
                                return ModelValidation(valid=False, error=arch_msg)
                        
                        return ModelValidation(valid=True, model_type=model_type)
                    except (json.JSONDecodeError, IOError):
                        return ModelValidation(valid=False, error="Could not read model configuration.")
                else:
                    # Check for model files without config.json
                    model_files = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
                    if model_files:
                        return ModelValidation(valid=True, model_type="unknown")
                    return ModelValidation(valid=False, error="No model files found in directory.")
            else:
                # Single file - check extension
                if path.suffix.lower() in [".safetensors", ".bin", ".pt", ".pth", ".gguf"]:
                    return ModelValidation(valid=True, model_type="file")
                return ModelValidation(valid=False, error="Unsupported model file format.")
        
        else:
            # For remote models - validate source is allowed
            source_key = "huggingface" if source == ModelSource.HUGGINGFACE else "modelscope"
            is_source_valid, source_msg = validator.validate_model_path(model_path, source_key)
            if not is_source_valid:
                return ModelValidation(valid=False, error=source_msg)
            
            if "/" not in model_path:
                return ModelValidation(valid=False, error="Invalid model ID format. Use 'organization/model-name'")
            
            return ModelValidation(valid=True, model_type="remote")
    
    except Exception as e:
        return ModelValidation(valid=False, error="Failed to validate model")


# ============================================================================
# Model Registry Endpoints
# ============================================================================

# Valid source values (internal - not exposed in schema)
_VALID_MODEL_SOURCES = {"local", "huggingface", "modelscope"}


@router.post("/registry/register")
async def register_model(registration: ModelRegistration, db: Session = Depends(get_db)):
    """Register a model in the global registry"""
    try:
        # Validate source value is valid
        if registration.source not in _VALID_MODEL_SOURCES:
            raise HTTPException(status_code=400, detail="Invalid source type")
        
        # Validate source is allowed by system configuration
        validator = get_validator()
        is_valid, msg = validator.validate_model_path(registration.source_id, registration.source)
        if not is_valid:
            raise HTTPException(status_code=403, detail=msg)
        
        # For local models, validate path exists and is accessible
        if registration.source == "local":
            path = Path(registration.source_id)
            if not path.exists():
                raise HTTPException(status_code=400, detail="Model path does not exist. Please verify the path is correct.")
            if not os.access(registration.source_id, os.R_OK):
                raise HTTPException(status_code=400, detail="Cannot access model path. Please check read permissions.")
            
            # Validate architecture if config.json exists
            if path.is_dir():
                config_path = path / "config.json"
                if config_path.exists():
                    try:
                        import json
                        with open(config_path) as f:
                            config = json.load(f)
                        architectures = config.get("architectures", [])
                        if architectures and len(architectures) > 0:
                            arch_valid, arch_msg = validator.validate_architecture(architectures[0])
                            if not arch_valid:
                                raise HTTPException(status_code=403, detail=arch_msg)
                    except (json.JSONDecodeError, IOError):
                        pass  # Continue if config can't be read
        
        service = ModelRegistryService(db)
        result = service.register_model(
            name=registration.name,
            source=registration.source,
            source_id=registration.source_id,
            description=registration.description,
            model_type=registration.model_type,
            model_size=registration.model_size
        )
        return {"success": True, "model": result}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to register model")


@router.get("/registry/list")
async def list_registered_models(
    source: Optional[str] = Query(None, description="Filter by source"),
    db: Session = Depends(get_db)
):
    """List all registered models"""
    try:
        service = ModelRegistryService(db)
        models = service.list_models(source=source)
        return {"models": models, "total": len(models)}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to list models")


@router.get("/registry/{model_id}")
async def get_registered_model(model_id: str, db: Session = Depends(get_db)):
    """Get a specific registered model by ID"""
    try:
        service = ModelRegistryService(db)
        model = service.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
        return model
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get model")


@router.get("/registry/{model_id}/delete-info")
async def get_model_delete_info(model_id: str, db: Session = Depends(get_db)):
    """Get information needed for delete confirmation (returns model name)"""
    try:
        service = ModelRegistryService(db)
        model = service.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
        
        # Check if model is used by any trainings
        trainings_count = len(model.training_jobs) if hasattr(model, 'training_jobs') else 0
        
        return {
            "model_id": model_id,
            "model_name": model.name,
            "source": model.source,
            "trainings_count": trainings_count,
            "can_delete": trainings_count == 0,
            "confirm_text": model.name  # User must type this to confirm deletion
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get model info")


@router.delete("/registry/{model_id}")
async def unregister_model(
    model_id: str,
    confirm: str = Query(..., description="Type the model NAME to confirm"),
    force: bool = Query(False, description="Force delete even if used by trainings"),
    db: Session = Depends(get_db)
):
    """Unregister a model from the registry. User must type the model name to confirm."""
    try:
        service = ModelRegistryService(db)
        model = service.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
        
        # Validate confirmation matches the model name
        if confirm != model.name:
            raise HTTPException(
                status_code=400, 
                detail=f"Confirmation failed. You must type '{model.name}' to delete this model."
            )
        
        result = service.delete_model(model_id, force=force)
        result["deleted_name"] = model.name
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to unregister model")


@router.get("/registry/popular")
async def get_popular_models(limit: int = Query(10, ge=1, le=50), db: Session = Depends(get_db)):
    """Get most frequently used models"""
    try:
        service = ModelRegistryService(db)
        models = service.get_popular_models(limit=limit)
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get popular models")
