# Copyright (c) US Inc. All rights reserved.
"""
Iterative Self-Training API Endpoints

Provides REST API for managing iterative self-training jobs (ReST/STaR/Expert Iteration).
Supports create, start, pause, cancel, status, history, and delete operations.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

from ...services.iterative_training_service import (
    iterative_training_service,
    IterativeTrainingConfig,
    IterativeTrainingJob,
    IterativeTrainingStatus,
    IterativePhase,
    FilterStrategy,
    RewardModelType,
    RewardModelConfig,
    DatasetSelectionStrategy,
    DatasetConfig,
    DifficultyLevel,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class RewardConfigRequest(BaseModel):
    """Configuration for reward/judge model"""
    type: str = Field(default="local", description="Reward model type: local, api, script, rule_based")
    
    # For LOCAL type
    model_path: str = Field(default="", description="HuggingFace model path for local reward model")
    
    # For API type
    api_endpoint: str = Field(default="", description="API URL for external scoring service")
    api_key: str = Field(default="", description="API key if needed")
    api_timeout: float = Field(default=30.0, description="Request timeout in seconds")
    api_batch_size: int = Field(default=10, description="Batch size for API calls")
    
    # For SCRIPT type
    script_path: str = Field(default="", description="Path to Python scoring script")
    script_function: str = Field(default="score", description="Function name to call in script")
    
    # For RULE_BASED type
    rule_type: str = Field(default="math", description="Rule type: math, code, json, regex")


class DatasetConfigRequest(BaseModel):
    """Configuration for a dataset in curriculum"""
    path: str = Field(..., description="Path to dataset file")
    name: str = Field(default="", description="Human-readable name")
    difficulty: str = Field(default="medium", description="Difficulty level: easy, medium, hard, expert")
    weight: float = Field(default=1.0, description="Weight for random selection")
    samples_per_round: Optional[int] = Field(default=None, description="Max samples per round from this dataset")


class CreateIterativeJobRequest(BaseModel):
    """Request to create a new iterative training job"""
    name: str = Field(..., description="Human-readable job name")
    base_model_path: str = Field(..., description="Path to base model for training")
    
    # =========================================================================
    # REWARD MODEL CONFIGURATION
    # =========================================================================
    # Simple mode: just provide path (uses LOCAL type)
    reward_model_path: str = Field(default="", description="Path to reward model (simple mode)")
    # Advanced mode: full reward config
    reward_config: Optional[RewardConfigRequest] = Field(default=None, description="Advanced reward model configuration")
    
    # =========================================================================
    # DATASET CONFIGURATION
    # =========================================================================
    # Simple mode: single dataset
    prompts_dataset_path: str = Field(default="", description="Path to prompts dataset (simple mode)")
    # Advanced mode: multiple datasets with curriculum
    datasets: List[DatasetConfigRequest] = Field(default=[], description="Multiple datasets with curriculum settings")
    dataset_selection_strategy: str = Field(default="sequential", description="How to select samples: sequential, round_robin, difficulty_curriculum, random_weighted, random_uniform")
    samples_per_round: int = Field(default=1000, ge=1, description="How many prompts to use per round")
    
    # =========================================================================
    # DIFFICULTY/CURRICULUM SETTINGS
    # =========================================================================
    enable_difficulty_curriculum: bool = Field(default=False, description="Enable difficulty-based curriculum learning")
    difficulty_progression: List[str] = Field(default=["easy", "easy", "medium", "medium", "hard"], description="Difficulty per round")
    
    # =========================================================================
    # ROUND SETTINGS
    # =========================================================================
    num_rounds: int = Field(default=5, ge=1, le=1000, description="Number of iterative rounds (can be millions via restarts)")
    
    # =========================================================================
    # GENERATION SETTINGS
    # =========================================================================
    num_generations_per_prompt: int = Field(default=8, ge=1, le=64, description="Completions per prompt")
    max_new_tokens: int = Field(default=512, ge=32, le=4096, description="Max tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
    
    # =========================================================================
    # FILTERING SETTINGS
    # =========================================================================
    filter_strategy: str = Field(default="top_k_percent", description="Filter strategy: top_k_percent, threshold, top_n, best_of_n")
    filter_top_k_percent: float = Field(default=20.0, ge=1.0, le=100.0, description="Keep top K%")
    filter_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Min reward threshold")
    filter_top_n: int = Field(default=1000, ge=1, description="Max samples to keep")
    
    # =========================================================================
    # TRAINING SETTINGS
    # =========================================================================
    training_method: str = Field(default="lora", description="lora, qlora, or full")
    learning_rate: float = Field(default=1e-5, gt=0, description="Learning rate")
    num_train_epochs: int = Field(default=1, ge=1, le=10, description="Epochs per round")
    batch_size: int = Field(default=4, ge=1, le=64, description="Training batch size")
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=64, description="Gradient accumulation")
    
    # =========================================================================
    # ADVANCED SETTINGS
    # =========================================================================
    use_previous_round_model: bool = Field(default=True, description="Train from previous round output")
    enable_encrypted_logging: bool = Field(default=True, description="Enable encrypted logging")
    log_generated_samples: bool = Field(default=True, description="Save all generated samples")
    log_scored_samples: bool = Field(default=True, description="Save all scored samples")


class IterativeJobResponse(BaseModel):
    """Response containing job details"""
    id: str
    name: str
    status: str
    current_phase: str
    current_round: int
    total_rounds: int
    progress_percent: float
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    final_model_path: Optional[str] = None
    round_history: List[Dict[str, Any]] = []


class IterativeJobListResponse(BaseModel):
    """Response containing list of jobs"""
    jobs: List[IterativeJobResponse]
    total: int


class RoundHistoryResponse(BaseModel):
    """Response containing round history and metrics"""
    job_id: str
    job_name: str
    total_rounds: int
    completed_rounds: int
    rounds: List[Dict[str, Any]]


# =============================================================================
# API ENDPOINTS
# =============================================================================

@router.post("/jobs", response_model=IterativeJobResponse)
async def create_iterative_job(request: CreateIterativeJobRequest):
    """
    Create a new iterative self-training job.
    
    This creates the job configuration but does not start training.
    Use POST /jobs/{job_id}/start to begin training.
    """
    try:
        # Convert filter strategy string to enum
        try:
            filter_strat = FilterStrategy(request.filter_strategy)
        except ValueError:
            filter_strat = FilterStrategy.TOP_K_PERCENT
        
        # Create config
        import uuid
        job_id = str(uuid.uuid4())
        
        config = IterativeTrainingConfig(
            job_id=job_id,
            job_name=request.name,
            base_model_path=request.base_model_path,
            reward_model_path=request.reward_model_path,
            prompts_dataset_path=request.prompts_dataset_path,
            output_dir="",  # Will be set by service
            num_rounds=request.num_rounds,
            num_generations_per_prompt=request.num_generations_per_prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            filter_strategy=filter_strat,
            filter_top_k_percent=request.filter_top_k_percent,
            filter_threshold=request.filter_threshold,
            filter_top_n=request.filter_top_n,
            training_method=request.training_method,
            learning_rate=request.learning_rate,
            num_train_epochs=request.num_train_epochs,
            batch_size=request.batch_size,
            gradient_accumulation_steps=request.gradient_accumulation_steps,
            use_previous_round_model=request.use_previous_round_model,
        )
        
        job = iterative_training_service.create_job(config)
        
        logger.info(f"Created iterative training job: {job.id} ({job.name})")
        
        return IterativeJobResponse(
            id=job.id,
            name=job.name,
            status=job.status.value if hasattr(job.status, 'value') else str(job.status),
            current_phase=job.current_phase.value if hasattr(job.current_phase, 'value') else str(job.current_phase),
            current_round=job.current_round,
            total_rounds=job.total_rounds,
            progress_percent=job.progress_percent,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
            final_model_path=job.final_model_path,
            round_history=job.round_history,
        )
        
    except Exception as e:
        logger.error(f"Failed to create iterative job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs", response_model=IterativeJobListResponse)
async def list_iterative_jobs():
    """List all iterative training jobs"""
    try:
        jobs = iterative_training_service.list_jobs()
        
        return IterativeJobListResponse(
            jobs=[
                IterativeJobResponse(
                    id=job.id,
                    name=job.name,
                    status=job.status.value if hasattr(job.status, 'value') else str(job.status),
                    current_phase=job.current_phase.value if hasattr(job.current_phase, 'value') else str(job.current_phase),
                    current_round=job.current_round,
                    total_rounds=job.total_rounds,
                    progress_percent=job.progress_percent,
                    created_at=job.created_at,
                    started_at=job.started_at,
                    completed_at=job.completed_at,
                    error_message=job.error_message,
                    final_model_path=job.final_model_path,
                    round_history=job.round_history,
                )
                for job in jobs
            ],
            total=len(jobs),
        )
        
    except Exception as e:
        logger.error(f"Failed to list iterative jobs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}", response_model=IterativeJobResponse)
async def get_iterative_job(job_id: str):
    """Get details of a specific iterative training job"""
    try:
        job = iterative_training_service.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        
        return IterativeJobResponse(
            id=job.id,
            name=job.name,
            status=job.status.value if hasattr(job.status, 'value') else str(job.status),
            current_phase=job.current_phase.value if hasattr(job.current_phase, 'value') else str(job.current_phase),
            current_round=job.current_round,
            total_rounds=job.total_rounds,
            progress_percent=job.progress_percent,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
            final_model_path=job.final_model_path,
            round_history=job.round_history,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get iterative job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/start")
async def start_iterative_job(job_id: str, background_tasks: BackgroundTasks):
    """
    Start an iterative training job.
    
    The job runs in the background. Use GET /jobs/{job_id} to monitor progress.
    """
    try:
        job = iterative_training_service.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        
        # Start in background
        background_tasks.add_task(iterative_training_service.start_job, job_id)
        
        return {
            "status": "started",
            "job_id": job_id,
            "message": f"Iterative training job '{job.name}' started. Monitor progress at GET /iterative/jobs/{job_id}"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start iterative job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/pause")
async def pause_iterative_job(job_id: str):
    """
    Pause a running iterative training job.
    
    The job will pause after the current phase completes.
    Use POST /jobs/{job_id}/start to resume.
    """
    try:
        success = await iterative_training_service.pause_job(job_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Job is not running or cannot be paused")
        
        return {
            "status": "pausing",
            "job_id": job_id,
            "message": "Job will pause after current phase completes"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to pause iterative job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/cancel")
async def cancel_iterative_job(job_id: str):
    """
    Cancel a running iterative training job.
    
    The job will be cancelled after the current phase completes.
    """
    try:
        success = await iterative_training_service.cancel_job(job_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Job is not running or cannot be cancelled")
        
        return {
            "status": "cancelling",
            "job_id": job_id,
            "message": "Job will be cancelled after current phase completes"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel iterative job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/jobs/{job_id}")
async def delete_iterative_job(job_id: str):
    """
    Delete an iterative training job and all its data.
    
    Cannot delete a running job - cancel it first.
    """
    try:
        success = iterative_training_service.delete_job(job_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        
        return {
            "status": "deleted",
            "job_id": job_id,
            "message": "Job and all associated data deleted"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete iterative job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/history", response_model=RoundHistoryResponse)
async def get_iterative_job_history(job_id: str):
    """
    Get detailed round history for an iterative training job.
    
    Returns metrics for each completed round including:
    - Generation stats (samples, time)
    - Judging stats (scores, time)
    - Filtering stats (samples kept)
    - Training stats (loss, time, checkpoint path)
    """
    try:
        job = iterative_training_service.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        
        return RoundHistoryResponse(
            job_id=job.id,
            job_name=job.name,
            total_rounds=job.total_rounds,
            completed_rounds=len(job.round_history),
            rounds=job.round_history,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get history for job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/rounds/{round_num}")
async def get_round_details(job_id: str, round_num: int):
    """
    Get detailed information about a specific round.
    
    Returns the round metrics plus paths to generated/filtered data files.
    """
    try:
        job = iterative_training_service.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        
        if round_num < 0 or round_num >= len(job.round_history):
            raise HTTPException(status_code=404, detail=f"Round {round_num} not found")
        
        round_metrics = job.round_history[round_num]
        
        # Add file paths
        import os
        round_dir = os.path.join(job.output_dir, "rounds", f"round_{round_num}")
        
        return {
            "round_number": round_num,
            "metrics": round_metrics,
            "files": {
                "generated": os.path.join(round_dir, "generated.jsonl"),
                "scored": os.path.join(round_dir, "scored.jsonl"),
                "filtered": os.path.join(round_dir, "filtered_training_data.jsonl"),
                "metrics": os.path.join(round_dir, "metrics.json"),
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get round {round_num} for job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================

@router.get("/filter-strategies")
async def get_filter_strategies():
    """Get available filter strategies"""
    return {
        "strategies": [
            {
                "value": "top_k_percent",
                "name": "Top K Percent",
                "description": "Keep the top K% of samples by reward score",
            },
            {
                "value": "threshold",
                "name": "Threshold",
                "description": "Keep all samples above a minimum reward threshold",
            },
            {
                "value": "top_n",
                "name": "Top N",
                "description": "Keep the top N samples by reward score",
            },
            {
                "value": "best_of_n",
                "name": "Best of N",
                "description": "Keep only the best response for each prompt",
            },
        ]
    }


@router.get("/training-methods")
async def get_training_methods():
    """Get available training methods for iterative training"""
    return {
        "methods": [
            {
                "value": "lora",
                "name": "LoRA",
                "description": "Low-Rank Adaptation - efficient fine-tuning with small adapters",
            },
            {
                "value": "qlora",
                "name": "QLoRA",
                "description": "Quantized LoRA - even more memory efficient with 4-bit quantization",
            },
            {
                "value": "full",
                "name": "Full Fine-Tuning",
                "description": "Update all model parameters - requires more VRAM",
            },
        ]
    }
