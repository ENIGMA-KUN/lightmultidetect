import uuid
from sqlalchemy import Column, String, Float, DateTime, JSON, Boolean
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime

from app.db.session import Base


class DetectionResult(Base):
    __tablename__ = "detection_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_hash = Column(String, unique=True, index=True)
    file_path = Column(String)
    media_type = Column(String)  # image, audio, video
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Results
    is_fake = Column(Boolean, default=False)
    confidence_score = Column(Float)
    detection_details = Column(JSON)  # Store detailed detection results
    
    # Models used
    models_used = Column(JSON)  # Store information about which models were used
    
    # Visualization data
    heatmap_path = Column(String, nullable=True)  # Path to heatmap visualization if available
    temporal_analysis_path = Column(String, nullable=True)  # Path to temporal analysis if available