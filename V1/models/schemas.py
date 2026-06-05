"""
Pydantic schemas for data validation.
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime


class EntityType(str, Enum):
    """Entity type enumeration."""
    CONCEPT = "概念"
    METHOD = "方法"
    TOOL = "工具"
    MODEL = "模型"
    PRINCIPLE = "原则"
    PATTERN = "模式"
    LANGUAGE = "语言"
    FRAMEWORK = "框架"


class Entity(BaseModel):
    """Entity model with validation."""
    name: str = Field(..., min_length=1, max_length=100, description="Entity name")
    type: Optional[EntityType] = Field(None, description="Entity type")
    
    @field_validator('name')
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        """Validate name is not empty."""
        if not v or not v.strip():
            raise ValueError('实体名称不能为空')
        return v.strip()


class QuestionInput(BaseModel):
    """User question input validation."""
    text: str = Field(..., min_length=1, max_length=1000, description="Question text")
    session_id: Optional[str] = Field(None, description="Session identifier")
    
    @field_validator('text')
    @classmethod
    def text_valid(cls, v: str) -> str:
        """Validate question text."""
        if not v or not v.strip():
            raise ValueError('问题不能为空')
        if len(v) > 1000:
            raise ValueError('问题过长（最大1000字符）')
        return v.strip()


class SearchResult(BaseModel):
    """Document search result model."""
    source: str = Field(..., description="Source file name")
    text: str = Field(..., max_length=1000, description="Text snippet")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score")


class FileUploadResult(BaseModel):
    """File upload result model."""
    success: bool = Field(..., description="Whether upload succeeded")
    message: str = Field(..., description="Result message")
    files_processed: int = Field(..., ge=0, description="Number of files processed")
    chunks_created: Optional[int] = Field(None, ge=0, description="Number of chunks created")
    cached_files: Optional[int] = Field(None, ge=0, description="Number of cached files")


class AnswerOutput(BaseModel):
    """Answer output model."""
    answer: str = Field(..., description="Generated answer")
    agent_name: str = Field(..., description="Agent that generated the answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    has_kg_context: bool = Field(..., description="Whether knowledge graph context was used")
    doc_results_count: int = Field(..., ge=0, description="Number of document search results")
    execution_time: float = Field(..., ge=0.0, description="Execution time in seconds")
    timestamp: str = Field(..., description="Timestamp of the answer")
    performance: Optional[Dict[str, str]] = Field(None, description="Performance metrics")
    knowledge_graph: Optional[List[Dict[str, Any]]] = Field(None, description="KG context")
    search_results: Optional[List[SearchResult]] = Field(None, description="Document results")


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Overall status: healthy, degraded, down")
    timestamp: str = Field(..., description="Timestamp of health check")
    services: Dict[str, bool] = Field(..., description="Individual service statuses")
    stats: Optional[Dict[str, Any]] = Field(None, description="Additional statistics")
