import uuid
from sqlalchemy import Column, String, Float, DateTime, JSON, Boolean, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.session import Base


class DetectionResult(Base):
    """
    Model for storing detection results.
    
    This table stores the results of deepfake detection analysis on various media types.
    It includes confidence scores, detailed analysis results, and paths to visualizations.
    """
    __tablename__ = "detection_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_hash = Column(String(64), unique=True, index=True, nullable=False, 
                      comment="SHA-256 hash of the file")
    file_path = Column(String(255), nullable=False,
                      comment="Path to the uploaded file")
    media_type = Column(String(10), nullable=False, index=True,
                       comment="Type of media: image, audio, video")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False,
                       comment="When the analysis was performed")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False,
                       comment="When the record was last updated")
    
    # Results
    is_fake = Column(Boolean, default=False, nullable=False,
                    comment="Whether the media is detected as fake")
    confidence_score = Column(Float, nullable=False,
                             comment="Confidence score of the detection (0-1)")
    
    # Using JSONB for better performance with complex JSON querying
    detection_details = Column(JSONB, nullable=True,
                              comment="Detailed detection results as JSON")
    
    # Models used
    models_used = Column(JSONB, nullable=True,
                        comment="Information about which models were used")
    
    # Visualization data
    heatmap_path = Column(String(255), nullable=True,
                         comment="Path to heatmap visualization if available")
    temporal_analysis_path = Column(String(255), nullable=True,
                                   comment="Path to temporal analysis if available")
    
    def __repr__(self):
        """String representation of the result."""
        return f"<DetectionResult id={self.id} media_type={self.media_type} is_fake={self.is_fake}>"


# You can add more models as needed for your application, such as:

class User(Base):
    """
    User model for authentication (if needed in the future).
    """
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Define this relationship if you implement this feature in the future
    # results = relationship("DetectionResult", back_populates="user")
    
    def __repr__(self):
        return f"<User {self.username}>"