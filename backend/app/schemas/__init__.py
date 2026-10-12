"""Pydantic request/response models (TRD §7.2)."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320, pattern=r"^[^@\s]+@[^@\s]+$")
    password: str = Field(min_length=1, max_length=200)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105 - OAuth2 token type, not a secret
    expires_in: int
    user: "UserOut"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    email: str
    full_name: str | None
    role: str
    org_id: str


class SessionCreate(BaseModel):
    mode: Literal["live", "upload"]
    title: str | None = Field(default=None, max_length=200)
    consent: bool

    @field_validator("consent")
    @classmethod
    def consent_required(cls, v: bool) -> bool:
        if not v:
            raise ValueError("Consent is required before any capture or upload (FR-04)")
        return v


class ModalityInfo(BaseModel):
    available: bool
    reliability: float
    weight: float
    emotion: str | None
    confidence: float | None


class SegmentOut(BaseModel):
    id: str
    index: int
    start_ms: int
    end_ms: int
    transcript: str | None
    emotion: str
    confidence: float
    abstained: bool
    valence: float | None
    arousal: float | None
    conflict: bool
    probabilities: dict[str, float]
    modalities: dict[str, ModalityInfo]
    explanation: list[str]


class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    rule: str
    severity: str
    start_ms: int
    end_ms: int | None
    message: str
    acknowledged: bool


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    mode: str
    title: str | None
    status: str
    progress: int
    duration_ms: int | None
    modalities_used: list[str]
    model_version: str | None
    error_message: str | None
    created_at: datetime
    completed_at: datetime | None
    segment_count: int = 0
    alert_count: int = 0
    dominant_emotion: str | None = None
    owner_email: str | None = None


class SessionDetail(SessionOut):
    summary: dict[str, Any] | None


class Page(BaseModel):
    items: list[SessionOut]
    total: int


class DashboardStats(BaseModel):
    sessions_total: int
    sessions_completed: int
    hours_analyzed: float
    alerts_total: int
    critical_alerts: int
    emotion_distribution: dict[str, float]
    mean_reliability: dict[str, float]
    recent_alerts: list[dict[str, Any]]


TokenResponse.model_rebuild()
