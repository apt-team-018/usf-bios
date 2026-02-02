from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, Enum, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.core.database import Base


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CLEANUP = "cleanup"


class DatasetStatus(str, enum.Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"
    DELETED = "deleted"


class DatasetSource(str, enum.Enum):
    UPLOAD = "upload"           # User uploaded file
    HUGGINGFACE = "huggingface" # HuggingFace Hub
    MODELSCOPE = "modelscope"   # ModelScope
    LOCAL = "local"             # Local file path


class DatasetTypeEnum(str, enum.Enum):
    """Dataset types based on format - determines compatible training methods"""
    SFT = "sft"                                          # Supervised Fine-Tuning (messages format)
    PT = "pt"                                            # Pre-Training (raw text)
    RLHF_OFFLINE_PREFERENCE = "rlhf_offline_preference"  # Offline RLHF preference (prompt/chosen/rejected)
    RLHF_OFFLINE_BINARY = "rlhf_offline_binary"          # Offline RLHF binary feedback (prompt/completion/label) - KTO
    RLHF_ONLINE = "rlhf_online"                          # Online RLHF (prompt only) - PPO/GRPO/GKD
    UNKNOWN = "unknown"                                  # Could not determine type


class ModelSource(str, enum.Enum):
    HUGGINGFACE = "huggingface" # HuggingFace Hub
    MODELSCOPE = "modelscope"   # ModelScope
    LOCAL = "local"             # Local model path


class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Source tracking - where this dataset came from
    source = Column(String(20), default=DatasetSource.UPLOAD.value, nullable=False, index=True)
    source_id = Column(String(512), nullable=True)  # HF/MS dataset ID or local path
    source_subset = Column(String(255), nullable=True)  # For HF datasets with subsets
    source_split = Column(String(50), nullable=True)  # train/test/validation
    
    # File information (for uploaded/local datasets)
    file_path = Column(String(512), nullable=True)  # Made nullable for HF/MS datasets
    file_name = Column(String(255), nullable=True)  # Made nullable for HF/MS datasets
    file_size = Column(Integer, nullable=False, default=0)
    file_format = Column(String(50), nullable=True)  # Made nullable for HF/MS datasets
    
    # Dataset type detection - determines compatible training methods
    dataset_type = Column(String(50), default=DatasetTypeEnum.UNKNOWN.value, nullable=False, index=True)
    dataset_type_confidence = Column(Float, default=0.0)  # 0.0 to 1.0 confidence score
    compatible_training_methods = Column(JSON, nullable=True)  # ['sft'], ['rlhf'], ['pt'], etc.
    compatible_rlhf_algorithms = Column(JSON, nullable=True)  # ['dpo', 'orpo'], ['kto'], ['ppo', 'grpo'], etc.
    detected_fields = Column(JSON, nullable=True)  # Fields found in dataset samples
    
    num_samples = Column(Integer, nullable=True)
    num_columns = Column(Integer, nullable=True)
    column_info = Column(JSON, nullable=True)
    
    status = Column(String(20), default=DatasetStatus.UPLOADING.value, index=True)
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    training_jobs = relationship("TrainingJob", back_populates="dataset")
    
    __table_args__ = (
        Index("idx_dataset_status_created", "status", "created_at"),
    )


class RegisteredModel(Base):
    """Global model registry - models can be used across multiple trainings"""
    __tablename__ = "registered_models"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Source tracking
    source = Column(String(20), default=ModelSource.HUGGINGFACE.value, nullable=False, index=True)
    source_id = Column(String(512), nullable=False)  # HF/MS model ID or local path
    
    # Model info
    model_type = Column(String(50), nullable=True)  # llm, vlm, etc.
    model_size = Column(String(20), nullable=True)  # 7B, 14B, 40B, etc.
    
    # Usage tracking
    times_used = Column(Integer, default=0)
    last_used_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    training_jobs = relationship("TrainingJob", back_populates="registered_model")
    
    __table_args__ = (
        Index("idx_model_source", "source"),
        UniqueConstraint("source", "source_id", name="uq_model_source_id"),
    )


class TrainingJob(Base):
    __tablename__ = "training_jobs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, index=True)
    
    # Reference to global dataset (optional - can also use inline dataset path)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=True)
    dataset = relationship("Dataset", back_populates="training_jobs")
    
    # Reference to global model registry (optional - can also use inline model path)
    registered_model_id = Column(String(36), ForeignKey("registered_models.id"), nullable=True)
    registered_model = relationship("RegisteredModel", back_populates="training_jobs")
    
    # Inline model info (for quick setup without registration)
    model_source = Column(String(50), nullable=False)
    model_path = Column(String(512), nullable=False)
    model_name = Column(String(255), nullable=True)
    
    training_config = Column(JSON, nullable=False, default=dict)
    config_finalized = Column(Boolean, default=False)
    
    status = Column(String(20), default=JobStatus.PENDING.value, index=True)
    error_message = Column(Text, nullable=True)
    
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    
    output_dir = Column(String(512), nullable=True)
    checkpoint_dir = Column(String(512), nullable=True)
    log_file_path = Column(String(512), nullable=True)
    final_model_path = Column(String(512), nullable=True)
    
    resume_from_checkpoint_id = Column(String(36), nullable=True)
    resume_from_step = Column(Integer, nullable=True)
    resume_count = Column(Integer, default=0)
    
    original_config_hash = Column(String(64), nullable=True)
    
    gpu_memory_used = Column(Float, nullable=True)
    gpu_utilization = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    metrics = relationship("TrainingMetric", back_populates="job", cascade="all, delete-orphan")
    checkpoints = relationship("Checkpoint", back_populates="job", cascade="all, delete-orphan")
    logs = relationship("TrainingLog", back_populates="job", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_job_status_created", "status", "created_at"),
    )


class TrainingMetric(Base):
    """
    Training metrics table - supports ALL training algorithms:
    - SFT (Supervised Fine-Tuning): loss, learning_rate, grad_norm
    - RLHF: reward, kl_divergence, policy_loss, value_loss
    - PPO: policy_loss, value_loss, entropy, approx_kl, clip_fraction
    - DPO: chosen_rewards, rejected_rewards, reward_margin
    - GRPO: reward, kl_penalty, policy_gradient_loss
    - GKD: distillation_loss, student_loss, teacher_loss
    - Pre-training: loss, perplexity, tokens_per_second
    - ASR: wer, cer, loss
    - TTS: mel_loss, duration_loss, pitch_loss
    - Multimodal: image_loss, text_loss, contrastive_loss
    
    Common columns handle standard metrics.
    extra_metrics JSON column handles algorithm-specific metrics dynamically.
    """
    __tablename__ = "training_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    job = relationship("TrainingJob", back_populates="metrics")
    
    # Training type for filtering (sft, rlhf, ppo, dpo, grpo, gkd, pretrain, asr, tts, multimodal)
    train_type = Column(String(50), nullable=True, index=True)
    
    step = Column(Integer, nullable=False, index=True)
    epoch = Column(Float, nullable=True)
    
    # Common metrics (all training types)
    loss = Column(Float, nullable=True)
    learning_rate = Column(Float, nullable=True)
    grad_norm = Column(Float, nullable=True)
    
    # Evaluation metrics
    eval_loss = Column(Float, nullable=True)
    eval_accuracy = Column(Float, nullable=True)
    eval_perplexity = Column(Float, nullable=True)
    
    # RLHF/PPO specific
    reward = Column(Float, nullable=True)
    kl_divergence = Column(Float, nullable=True)
    policy_loss = Column(Float, nullable=True)
    value_loss = Column(Float, nullable=True)
    entropy = Column(Float, nullable=True)
    
    # DPO specific
    chosen_rewards = Column(Float, nullable=True)
    rejected_rewards = Column(Float, nullable=True)
    reward_margin = Column(Float, nullable=True)
    
    # Dynamic extra metrics (for any algorithm-specific metrics)
    # This JSON column can store ANY additional metrics not covered above
    extra_metrics = Column(JSON, nullable=True)
    
    # System metrics
    gpu_memory_mb = Column(Float, nullable=True)
    gpu_utilization_pct = Column(Float, nullable=True)
    throughput_samples_sec = Column(Float, nullable=True)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index("idx_metric_job_step", "job_id", "step"),
        Index("idx_metric_train_type", "train_type"),
        UniqueConstraint("job_id", "step", name="uq_job_step"),
    )


class Checkpoint(Base):
    __tablename__ = "checkpoints"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String(36), ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    job = relationship("TrainingJob", back_populates="checkpoints")
    
    step = Column(Integer, nullable=False)
    epoch = Column(Float, nullable=True)
    
    path = Column(String(512), nullable=False)
    size_mb = Column(Float, nullable=True)
    
    is_best = Column(Boolean, default=False)
    is_final = Column(Boolean, default=False)
    
    metrics_snapshot = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_checkpoint_job_step", "job_id", "step"),
    )


class TrainingLog(Base):
    __tablename__ = "training_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    job = relationship("TrainingJob", back_populates="logs")
    
    level = Column(String(20), default="INFO")
    message_encrypted = Column(Text, nullable=False)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index("idx_log_job_timestamp", "job_id", "timestamp"),
    )


class SystemState(Base):
    __tablename__ = "system_state"
    
    id = Column(Integer, primary_key=True, default=1)
    
    current_job_id = Column(String(36), ForeignKey("training_jobs.id"), nullable=True)
    is_training_active = Column(Boolean, default=False)
    
    last_cleanup_at = Column(DateTime, nullable=True)
    
    gpu_info = Column(JSON, nullable=True)
    
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
