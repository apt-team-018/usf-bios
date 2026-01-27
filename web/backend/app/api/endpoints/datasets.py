# Copyright (c) US Inc. All rights reserved.
"""Dataset-related endpoints"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from ...core.config import settings
from ...core.capabilities import get_system_settings, get_validator
from ...models.schemas import DatasetValidation
from ...services.dataset_type_service import dataset_type_service, DatasetType, ModelType, FileFormatConfig

router = APIRouter()

# In-memory registry for datasets (in production, use a database)
_dataset_registry: dict = {}


class DatasetRegistration(BaseModel):
    """Request model for registering a dataset"""
    name: str
    source: str = "local_path"  # Dataset source
    dataset_id: str  # Local path to dataset directory or file
    subset: Optional[str] = None  # Dataset subset name
    split: Optional[str] = "train"
    max_samples: Optional[int] = None  # None or 0 = use all samples


class RegisteredDataset(BaseModel):
    """Registered dataset info"""
    id: str
    name: str
    source: str  # Dataset source
    path: str  # Actual path or dataset ID
    subset: Optional[str] = None
    split: Optional[str] = None
    total_samples: int = 0
    size_human: str = "Unknown"
    format: str = "unknown"
    created_at: float
    selected: bool = True
    max_samples: Optional[int] = None  # None or 0 = use all samples
    dataset_type: Optional[str] = None  # sft, rlhf_offline, rlhf_online, pt, kto, unknown
    dataset_type_display: Optional[str] = None  # Human-readable type name
    compatible_training_methods: Optional[List[str]] = None


@router.post("/validate", response_model=DatasetValidation)
async def validate_dataset(dataset_path: str = Query(..., description="Path to dataset file")):
    """Validate dataset format and structure"""
    try:
        path = Path(dataset_path)
        
        if not path.exists():
            return DatasetValidation(
                valid=False,
                errors=["Dataset path does not exist"]
            )
        
        # Detect format by extension
        suffix = path.suffix.lower()
        
        if suffix == ".jsonl":
            return await _validate_jsonl(path)
        elif suffix == ".json":
            return await _validate_json(path)
        elif suffix == ".csv":
            return await _validate_csv(path)
        else:
            return DatasetValidation(
                valid=False,
                format_detected="unknown",
                errors=[f"Unsupported format: {suffix}. Use .jsonl, .json, or .csv"]
            )
    
    except Exception as e:
        return DatasetValidation(
            valid=False,
            errors=["Failed to validate dataset"]
        )


async def _validate_jsonl(path: Path) -> DatasetValidation:
    """Validate JSONL dataset"""
    samples = []
    total = 0
    errors = []
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total += 1
                if i < 5:  # Preview first 5
                    try:
                        sample = json.loads(line.strip())
                        samples.append(sample)
                    except json.JSONDecodeError:
                        errors.append(f"Invalid JSON on line {i+1}")
        
        if not samples:
            return DatasetValidation(
                valid=False,
                total_samples=0,
                format_detected="jsonl",
                errors=["Dataset is empty"]
            )
        
        columns = list(samples[0].keys())
        
        # Check for required fields
        has_messages = "messages" in columns
        has_instruction = "instruction" in columns
        has_query_response = "query" in columns and "response" in columns
        
        if not (has_messages or has_instruction or has_query_response):
            errors.append("Dataset must have one of: 'messages', 'instruction', or 'query'/'response' fields")
        
        # Validate messages format if present
        if has_messages and samples:
            messages = samples[0].get("messages", [])
            if not isinstance(messages, list):
                errors.append("'messages' field must be a list")
            elif messages:
                for msg in messages:
                    if not isinstance(msg, dict):
                        errors.append("Each message must be a dict with 'role' and 'content'")
                        break
                    if "role" not in msg or "content" not in msg:
                        errors.append("Each message must have 'role' and 'content' keys")
                        break
        
        return DatasetValidation(
            valid=len(errors) == 0,
            total_samples=total,
            format_detected="jsonl",
            columns=columns,
            sample_preview=samples,
            errors=errors
        )
    
    except Exception as e:
        return DatasetValidation(
            valid=False,
            format_detected="jsonl",
            errors=["Failed to read JSONL file"]
        )


async def _validate_json(path: Path) -> DatasetValidation:
    """Validate JSON dataset"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            samples = data[:5]
            total = len(data)
            columns = list(samples[0].keys()) if samples else []
        else:
            return DatasetValidation(
                valid=False,
                format_detected="json",
                errors=["JSON file must contain a list of samples"]
            )
        
        return DatasetValidation(
            valid=True,
            total_samples=total,
            format_detected="json",
            columns=columns,
            sample_preview=samples,
            errors=[]
        )
    
    except Exception as e:
        return DatasetValidation(
            valid=False,
            format_detected="json",
            errors=["Failed to read JSON file"]
        )


async def _validate_csv(path: Path) -> DatasetValidation:
    """Validate CSV dataset"""
    try:
        import csv
        
        samples = []
        total = 0
        
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            
            for i, row in enumerate(reader):
                total += 1
                if i < 5:
                    samples.append(dict(row))
        
        return DatasetValidation(
            valid=True,
            total_samples=total,
            format_detected="csv",
            columns=list(columns),
            sample_preview=samples,
            errors=[]
        )
    
    except Exception as e:
        return DatasetValidation(
            valid=False,
            format_detected="csv",
            errors=["Failed to read CSV file"]
        )


def _normalize_name(name: str) -> str:
    """Normalize dataset name for comparison"""
    safe = "".join(c for c in name if c.isalnum() or c in "._- ")
    return safe.strip().replace(" ", "_").lower()


def _is_name_taken(name: str) -> bool:
    """Check if normalized name already exists in uploads or registry"""
    normalized = _normalize_name(name)
    
    # Check uploaded files
    get_system_settings().UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for f in get_system_settings().UPLOAD_DIR.glob("*"):
        if f.suffix.lower() in [".jsonl", ".json", ".csv"]:
            if _normalize_name(f.stem) == normalized:
                return True
    
    # Check registered datasets
    for ds in _dataset_registry.values():
        if _normalize_name(ds.get("name", "")) == normalized:
            return True
    
    return False


@router.get("/check-name")
async def check_name_available(name: str = Query(..., description="Dataset name to check")):
    """Check if a dataset name is available"""
    normalized = _normalize_name(name)
    if not normalized:
        return {"available": False, "error": "Invalid name"}
    
    taken = _is_name_taken(name)
    return {"available": not taken, "normalized": normalized}


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_name: str = Query(..., description="Name for the dataset")
):
    """
    Upload a dataset file with a custom name.
    
    Validates:
    1. File format is supported by USF BIOS (jsonl, json, csv, txt)
    2. File size is within limits for the format
    3. Dataset type is allowed by system feature flags
    
    Supported formats:
    - JSONL: Unlimited size, recommended for large datasets
    - JSON: Max 2GB (must fit in memory)
    - CSV: Unlimited size, streaming supported
    - TXT: Unlimited size, for pre-training only
    
    NOT supported (will be rejected):
    - TSV, Parquet, Excel, Arrow
    """
    try:
        # === VALIDATION 1: Filename provided ===
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        suffix = Path(file.filename).suffix.lower()
        
        # === VALIDATION 2: Check for explicitly unsupported formats ===
        if FileFormatConfig.is_unsupported(suffix):
            error_msg = FileFormatConfig.get_unsupported_error(suffix)
            raise HTTPException(
                status_code=400, 
                detail=f"Format not supported: {error_msg}"
            )
        
        # === VALIDATION 3: Check if format is supported by USF BIOS ===
        file_format = FileFormatConfig.get_format_by_extension(suffix)
        if not file_format:
            supported = ", ".join(FileFormatConfig.get_all_extensions())
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported format: {suffix}. USF BIOS only supports: {supported}"
            )
        
        # Sanitize dataset name (remove special characters)
        safe_name = "".join(c for c in dataset_name if c.isalnum() or c in "._- ")
        safe_name = safe_name.strip().replace(" ", "_")
        if not safe_name:
            raise HTTPException(status_code=400, detail="Invalid dataset name")
        
        # Check if name is already taken
        if _is_name_taken(dataset_name):
            raise HTTPException(status_code=409, detail=f"Dataset name '{dataset_name}' is already in use. Please choose a different name.")
        
        # Create filename with user's name
        final_filename = f"{safe_name}{suffix}"
        upload_path = get_system_settings().UPLOAD_DIR / final_filename
        
        # Ensure upload directory exists
        get_system_settings().UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save file
        content = await file.read()
        with open(upload_path, "wb") as f:
            f.write(content)
        
        # Validate the uploaded file
        validation = await validate_dataset(str(upload_path))
        
        # Detect dataset type
        type_info = dataset_type_service.detect_dataset_type(str(upload_path))
        
        # Get feature flags and validate if dataset type is allowed
        try:
            from ...core.capabilities import get_validator
            validator = get_validator()
            feature_flags = validator.get_feature_flags() if hasattr(validator, 'get_feature_flags') else {}
        except Exception:
            feature_flags = {}
        
        # Validate dataset type against feature flags
        is_valid, error_msg, _ = dataset_type_service.validate_dataset_for_upload(
            str(upload_path), feature_flags
        )
        
        if not is_valid:
            # Delete the uploaded file since it's not allowed
            try:
                upload_path.unlink()
            except Exception:
                pass
            raise HTTPException(status_code=403, detail=error_msg)
        
        return {
            "success": True,
            "id": final_filename,
            "name": safe_name,
            "filename": final_filename,
            "path": str(upload_path),
            "size": len(content),
            "format": suffix[1:],  # Remove the dot
            "valid": validation.valid,
            "total_samples": validation.total_samples,
            "errors": validation.errors,
            "dataset_type": type_info.dataset_type.value,
            "dataset_type_display": type_info.display_name,
            "compatible_training_methods": type_info.compatible_training_methods,
            "compatible_rlhf_types": type_info.compatible_rlhf_types,
            "format_warning": type_info.format_warning,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to upload dataset")


@router.get("/list")
async def list_datasets():
    """List all uploaded datasets with metadata"""
    try:
        # Ensure upload directory exists
        get_system_settings().UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        datasets = []
        # Get all supported extensions from FileFormatConfig
        supported_extensions = FileFormatConfig.get_all_extensions()
        for f in get_system_settings().UPLOAD_DIR.glob("*"):
            if f.suffix.lower() in supported_extensions:
                # Get basic info
                stat = f.stat()
                
                # Try to get sample count based on format
                total_samples = 0
                try:
                    ext = f.suffix.lower()
                    if ext == ".jsonl":
                        with open(f, 'r', encoding='utf-8') as fp:
                            total_samples = sum(1 for _ in fp)
                    elif ext == ".json":
                        import json
                        with open(f, 'r', encoding='utf-8') as fp:
                            data = json.load(fp)
                            if isinstance(data, list):
                                total_samples = len(data)
                    elif ext == ".csv":
                        with open(f, 'r', encoding='utf-8') as fp:
                            total_samples = sum(1 for _ in fp) - 1  # Minus header
                    elif ext == ".txt":
                        with open(f, 'r', encoding='utf-8') as fp:
                            total_samples = sum(1 for _ in fp)  # Line count for text
                except Exception:
                    pass
                
                datasets.append({
                    "id": f.name,
                    "name": f.stem,  # Name without extension
                    "filename": f.name,
                    "path": str(f),
                    "size": stat.st_size,
                    "size_human": _format_size(stat.st_size),
                    "format": f.suffix[1:].lower(),
                    "total_samples": total_samples,
                    "created_at": stat.st_ctime,
                })
        
        # Sort by creation time (newest first)
        datasets.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {"datasets": datasets, "total": len(datasets)}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to list datasets")


@router.get("/delete-info/{dataset_id}")
async def get_dataset_delete_info(dataset_id: str):
    """Get information needed for delete confirmation (returns dataset name)"""
    # Check uploaded datasets
    dataset_path = get_system_settings().UPLOAD_DIR / dataset_id
    if dataset_path.exists() and dataset_path.is_file():
        dataset_name = dataset_path.stem  # Name without extension
        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "source": "upload",
            "confirm_text": dataset_name  # User must type this to confirm deletion
        }
    
    # Check registered datasets
    if dataset_id in _dataset_registry:
        ds = _dataset_registry[dataset_id]
        dataset_name = ds.get("name", dataset_id)
        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "source": ds.get("source", "unknown"),
            "confirm_text": dataset_name  # User must type this to confirm deletion
        }
    
    raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")


@router.delete("/delete/{dataset_id}")
async def delete_dataset(dataset_id: str, confirm: str = Query(..., description="Type the dataset NAME to confirm")):
    """Delete an uploaded dataset (requires typing the dataset name to confirm)"""
    try:
        # Find the dataset
        dataset_path = get_system_settings().UPLOAD_DIR / dataset_id
        
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
        
        if not dataset_path.is_file():
            raise HTTPException(status_code=400, detail="Invalid dataset path")
        
        # Get the dataset name for confirmation
        dataset_name = dataset_path.stem  # Name without extension
        
        # Validate confirmation matches the dataset name
        if confirm != dataset_name:
            raise HTTPException(
                status_code=400, 
                detail=f"Confirmation failed. You must type '{dataset_name}' to delete this dataset."
            )
        
        # Verify it's in the upload directory (security check)
        try:
            dataset_path.resolve().relative_to(get_system_settings().UPLOAD_DIR.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Delete the file
        dataset_path.unlink()
        
        return {
            "success": True,
            "message": f"Dataset '{dataset_name}' deleted successfully",
            "deleted_id": dataset_id,
            "deleted_name": dataset_name
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to delete dataset")


def _format_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# Valid source values (internal - not exposed in schema)
_VALID_DATASET_SOURCES = {"local_path", "local", "huggingface", "modelscope", "upload"}


@router.post("/register")
async def register_dataset(registration: DatasetRegistration):
    """Register a dataset from local path"""
    try:
        # Validate source value is valid
        if registration.source not in _VALID_DATASET_SOURCES:
            raise HTTPException(status_code=400, detail="Invalid source type")
        
        # Validate dataset is supported by this system configuration
        validator = get_validator()
        source_key = registration.source
        # Convert frontend source names to capability keys
        if source_key in ("local_path", "upload"):
            source_key = "local"
        
        is_valid, message = validator.validate_dataset_source(source_key)
        if not is_valid:
            raise HTTPException(status_code=403, detail=message)
        
        # Check if name is already taken
        if _is_name_taken(registration.name):
            raise HTTPException(status_code=409, detail=f"Dataset name '{registration.name}' is already in use. Please choose a different name.")
        
        dataset_id = str(uuid.uuid4())[:8]
        
        # Validate based on source
        if registration.source == "local_path":
            path = Path(registration.dataset_id)
            if not path.exists():
                raise HTTPException(status_code=400, detail=f"Local path does not exist: {registration.dataset_id}")
            
            # Get file info
            if path.is_file():
                stat = path.stat()
                size_human = _format_size(stat.st_size)
                fmt = path.suffix[1:].lower() if path.suffix else "unknown"
            else:
                size_human = "Directory"
                fmt = "directory"
        else:
            # For remote sources, we just register the ID
            size_human = "Remote"
            fmt = "hub"
        
        # Create registered dataset entry
        registered = RegisteredDataset(
            id=dataset_id,
            name=registration.name,
            source=registration.source,
            path=registration.dataset_id,
            subset=registration.subset,
            split=registration.split,
            total_samples=0,  # Will be determined during training
            size_human=size_human,
            format=fmt,
            created_at=datetime.now().timestamp(),
            selected=True,
            max_samples=registration.max_samples if registration.max_samples and registration.max_samples > 0 else None
        )
        
        # Store in registry
        _dataset_registry[dataset_id] = registered.model_dump()
        
        return {
            "success": True,
            "dataset": registered.model_dump()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to register dataset")


@router.get("/list-all")
async def list_all_datasets():
    """List all datasets (uploaded + registered) with dataset type labels"""
    try:
        get_system_settings().UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        all_datasets = []
        
        # 1. Get uploaded files
        for f in get_system_settings().UPLOAD_DIR.glob("*"):
            if f.suffix.lower() in [".jsonl", ".json", ".csv"]:
                stat = f.stat()
                total_samples = 0
                
                # Detect dataset type
                try:
                    type_info = dataset_type_service.detect_dataset_type(str(f))
                    dataset_type = type_info.dataset_type.value
                    dataset_type_display = type_info.display_name
                    compatible_methods = type_info.compatible_training_methods
                    total_samples = type_info.sample_count
                except Exception:
                    dataset_type = "unknown"
                    dataset_type_display = "Unknown"
                    compatible_methods = ["sft", "pt", "rlhf"]
                    # Fallback sample counting
                    try:
                        if f.suffix.lower() == ".jsonl":
                            with open(f, 'r', encoding='utf-8') as fp:
                                total_samples = sum(1 for _ in fp)
                        elif f.suffix.lower() == ".json":
                            with open(f, 'r', encoding='utf-8') as fp:
                                data = json.load(fp)
                                if isinstance(data, list):
                                    total_samples = len(data)
                        elif f.suffix.lower() == ".csv":
                            with open(f, 'r', encoding='utf-8') as fp:
                                total_samples = sum(1 for _ in fp) - 1
                    except Exception:
                        pass
                
                all_datasets.append({
                    "id": f"upload_{f.name}",
                    "name": f.stem,
                    "source": "upload",
                    "path": str(f),
                    "subset": None,
                    "split": None,
                    "total_samples": total_samples,
                    "size_human": _format_size(stat.st_size),
                    "format": f.suffix[1:].lower(),
                    "created_at": stat.st_ctime,
                    "selected": True,
                    "dataset_type": dataset_type,
                    "dataset_type_display": dataset_type_display,
                    "compatible_training_methods": compatible_methods,
                })
        
        # 2. Get registered datasets
        for ds in _dataset_registry.values():
            # Add dataset type if not already present
            if "dataset_type" not in ds:
                try:
                    type_info = dataset_type_service.detect_dataset_type(ds.get("path", ""))
                    ds["dataset_type"] = type_info.dataset_type.value
                    ds["dataset_type_display"] = type_info.display_name
                    ds["compatible_training_methods"] = type_info.compatible_training_methods
                except Exception:
                    ds["dataset_type"] = "unknown"
                    ds["dataset_type_display"] = "Unknown"
                    ds["compatible_training_methods"] = ["sft", "pt", "rlhf"]
            all_datasets.append(ds)
        
        # Sort by creation time (newest first)
        all_datasets.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        
        return {"datasets": all_datasets, "total": len(all_datasets)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to list datasets")


@router.delete("/unregister/{dataset_id}")
async def unregister_dataset(dataset_id: str, confirm: str = Query(..., description="Type the dataset NAME to confirm")):
    """Unregister a dataset. User must type the dataset name to confirm."""
    try:
        dataset_name = None
        
        # Check if it's an uploaded file
        if dataset_id.startswith("upload_"):
            filename = dataset_id[7:]  # Remove "upload_" prefix
            file_path = get_system_settings().UPLOAD_DIR / filename
            if file_path.exists():
                dataset_name = file_path.stem
                # Validate confirmation matches the dataset name
                if confirm != dataset_name:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Confirmation failed. You must type '{dataset_name}' to delete this dataset."
                    )
                file_path.unlink()
                return {"success": True, "message": f"Deleted uploaded file: {filename}", "deleted_name": dataset_name}
        
        # Check registry
        if dataset_id in _dataset_registry:
            ds = _dataset_registry[dataset_id]
            dataset_name = ds.get("name", dataset_id)
            # Validate confirmation matches the dataset name
            if confirm != dataset_name:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Confirmation failed. You must type '{dataset_name}' to delete this dataset."
                )
            del _dataset_registry[dataset_id]
            return {"success": True, "message": f"Unregistered dataset: {dataset_name}", "deleted_name": dataset_name}
        
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to unregister dataset")


# ============================================================================
# Dataset Type Detection Endpoints
# ============================================================================

class DatasetTypeResponse(BaseModel):
    """Response model for dataset type detection"""
    dataset_type: str
    confidence: float
    detected_fields: List[str]
    sample_count: int
    compatible_training_methods: List[str]
    incompatible_training_methods: List[str]
    compatible_rlhf_types: List[str] = []
    display_name: str = ""
    message: str
    file_size_bytes: int = 0
    is_large_file: bool = False
    format_warning: Optional[str] = None
    file_format: Optional[str] = None
    supports_streaming: bool = False
    estimated_samples: bool = False
    validation_errors: List[str] = []
    validation_warnings: List[str] = []


class ModelTypeResponse(BaseModel):
    """Response model for model type detection"""
    model_type: str
    is_adapter: bool
    base_model_path: Optional[str]
    can_do_lora: bool
    can_do_qlora: bool
    can_do_full: bool
    can_do_rlhf: bool
    warnings: List[str]


class TrainingValidationRequest(BaseModel):
    """Request for validating training configuration"""
    dataset_path: str
    training_method: str  # sft, pt, rlhf
    train_type: str  # lora, qlora, full, adalora
    model_path: Optional[str] = None
    rlhf_type: Optional[str] = None  # dpo, orpo, etc.


class TrainingValidationResponse(BaseModel):
    """Response for training configuration validation"""
    valid: bool
    message: str
    dataset_type: str
    model_type: Optional[str] = None
    compatible_methods: List[str]
    incompatible_methods: List[str]
    warnings: List[str]


@router.get("/detect-type", response_model=DatasetTypeResponse)
async def detect_dataset_type(dataset_path: str = Query(..., description="Path to dataset file")):
    """
    Detect the type of dataset (SFT, RLHF-Offline, RLHF-Online, PT).
    
    This is used by the frontend to:
    1. Restrict available training methods based on dataset type
    2. Auto-reset training method when dataset type changes
    3. Show appropriate warnings/info to users
    4. Label datasets with their type (SFT, RLHF-Online, RLHF-Offline, PT)
    """
    try:
        result = dataset_type_service.detect_dataset_type(dataset_path)
        return DatasetTypeResponse(
            dataset_type=result.dataset_type.value,
            confidence=result.confidence,
            detected_fields=result.detected_fields,
            sample_count=result.sample_count,
            compatible_training_methods=result.compatible_training_methods,
            incompatible_training_methods=result.incompatible_training_methods,
            compatible_rlhf_types=result.compatible_rlhf_types,
            display_name=result.display_name,
            message=result.message,
            file_size_bytes=result.file_size_bytes,
            is_large_file=result.is_large_file,
            format_warning=result.format_warning,
            file_format=result.file_format,
            supports_streaming=result.supports_streaming,
            estimated_samples=result.estimated_samples,
            validation_errors=result.validation_errors,
            validation_warnings=result.validation_warnings,
        )
    except Exception as e:
        return DatasetTypeResponse(
            dataset_type="unknown",
            confidence=0.0,
            detected_fields=[],
            sample_count=0,
            compatible_training_methods=["sft", "pt", "rlhf"],
            incompatible_training_methods=[],
            compatible_rlhf_types=[],
            display_name="Unknown",
            message=f"Error detecting dataset type: {str(e)}",
            validation_errors=[str(e)],
        )


@router.get("/detect-model-type", response_model=ModelTypeResponse)
async def detect_model_type(model_path: str = Query(..., description="Path to model directory")):
    """
    Detect if a model is a full model or LoRA adapter.
    
    This is used to:
    1. Warn users if they try to do full fine-tuning on a LoRA adapter
    2. Ensure proper base model is available for LoRA adapter training
    3. Validate RLHF compatibility
    """
    try:
        result = dataset_type_service.detect_model_type(model_path)
        return ModelTypeResponse(
            model_type=result.model_type.value,
            is_adapter=result.is_adapter,
            base_model_path=result.base_model_path,
            can_do_lora=result.can_do_lora,
            can_do_qlora=result.can_do_qlora,
            can_do_full=result.can_do_full,
            can_do_rlhf=result.can_do_rlhf,
            warnings=result.warnings,
        )
    except Exception as e:
        return ModelTypeResponse(
            model_type="unknown",
            is_adapter=False,
            base_model_path=None,
            can_do_lora=True,
            can_do_qlora=True,
            can_do_full=True,
            can_do_rlhf=True,
            warnings=[f"Error detecting model type: {str(e)}"],
        )


@router.post("/validate-training-config", response_model=TrainingValidationResponse)
async def validate_training_config(request: TrainingValidationRequest):
    """
    Validate if the training configuration is compatible with the dataset and model.
    
    This provides a final validation before training starts, ensuring:
    1. Dataset type matches training method
    2. Model type supports the selected train type
    3. All compatibility requirements are met
    """
    try:
        # Detect dataset type
        dataset_result = dataset_type_service.detect_dataset_type(request.dataset_path)
        
        # Detect model type if path provided
        model_result = None
        model_type_str = None
        warnings = []
        
        if request.model_path:
            model_result = dataset_type_service.detect_model_type(request.model_path)
            model_type_str = model_result.model_type.value
            warnings.extend(model_result.warnings)
        
        # Validate configuration
        is_valid, message = dataset_type_service.validate_training_config(
            dataset_type=dataset_result.dataset_type,
            training_method=request.training_method,
            train_type=request.train_type,
            model_type=model_result.model_type if model_result else ModelType.FULL,
            rlhf_type=request.rlhf_type
        )
        
        return TrainingValidationResponse(
            valid=is_valid,
            message=message,
            dataset_type=dataset_result.dataset_type.value,
            model_type=model_type_str,
            compatible_methods=dataset_result.compatible_training_methods,
            incompatible_methods=dataset_result.incompatible_training_methods,
            warnings=warnings,
        )
    
    except Exception as e:
        return TrainingValidationResponse(
            valid=False,
            message=f"Validation error: {str(e)}",
            dataset_type="unknown",
            model_type=None,
            compatible_methods=["sft", "pt", "rlhf"],
            incompatible_methods=[],
            warnings=[],
        )


@router.get("/detect-type-bulk")
async def detect_dataset_types_bulk(
    paths: str = Query(..., description="Comma-separated list of dataset paths")
):
    """
    Detect types for multiple datasets at once.
    
    Returns the combined compatible training methods (intersection of all datasets).
    This is useful when multiple datasets are selected for training.
    """
    try:
        path_list = [p.strip() for p in paths.split(",") if p.strip()]
        
        if not path_list:
            return {
                "datasets": [],
                "combined_compatible_methods": ["sft", "pt", "rlhf"],
                "combined_incompatible_methods": [],
                "all_same_type": True,
                "detected_type": "unknown",
                "message": "No datasets provided"
            }
        
        results = []
        types_found = set()
        compatible_sets = []
        
        for path in path_list:
            result = dataset_type_service.detect_dataset_type(path)
            results.append({
                "path": path,
                "type": result.dataset_type.value,
                "confidence": result.confidence,
                "compatible_methods": result.compatible_training_methods,
            })
            types_found.add(result.dataset_type.value)
            compatible_sets.append(set(result.compatible_training_methods))
        
        # Find intersection of compatible methods
        if compatible_sets:
            combined_compatible = set.intersection(*compatible_sets)
        else:
            combined_compatible = {"sft", "pt", "rlhf"}
        
        # Find union of incompatible methods
        all_incompatible = set()
        for result in results:
            ds_result = dataset_type_service.detect_dataset_type(result["path"])
            all_incompatible.update(ds_result.incompatible_training_methods)
        
        all_same_type = len(types_found) == 1
        detected_type = list(types_found)[0] if all_same_type else "mixed"
        
        message = ""
        if not all_same_type:
            message = f"Mixed dataset types detected: {', '.join(types_found)}. Only methods compatible with all datasets are available."
        elif detected_type != "unknown":
            type_display = {"sft": "SFT", "rlhf": "RLHF preference", "pt": "Pre-training", "kto": "KTO"}.get(detected_type, detected_type)
            message = f"All datasets are {type_display} format."
        
        return {
            "datasets": results,
            "combined_compatible_methods": list(combined_compatible),
            "combined_incompatible_methods": list(all_incompatible),
            "all_same_type": all_same_type,
            "detected_type": detected_type,
            "message": message
        }
    
    except Exception as e:
        return {
            "datasets": [],
            "combined_compatible_methods": ["sft", "pt", "rlhf"],
            "combined_incompatible_methods": [],
            "all_same_type": True,
            "detected_type": "unknown",
            "message": f"Error: {str(e)}"
        }


@router.get("/supported-formats")
async def get_supported_formats():
    """
    Get information about supported dataset file formats.
    
    Returns format limits, streaming support, and recommendations.
    This endpoint helps the frontend display format requirements to users.
    
    Format Summary:
    - JSONL: UNLIMITED size, streaming, recommended for large datasets
    - JSON: 2GB max, no streaming, for smaller datasets only
    - CSV/TSV: UNLIMITED size, streaming, good for tabular data
    - TXT: UNLIMITED size, streaming, for pre-training text
    - Parquet: UNLIMITED size, chunked reading, efficient columnar format
    """
    return dataset_type_service.get_supported_formats()
