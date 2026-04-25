// Centralised error type for all API handlers.

use std::sync::Arc;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

use crate::model_registry::{EngineEntry, ModelRegistry};

/// All failure modes that HTTP handlers can produce.
#[allow(dead_code)]
pub enum AppError {
    BadRequest(String),
    ModelNotFound(String),
    ModelLoadFailed(String),
    EmbeddingFailed(String),
    InternalError(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, msg, err_type, code) = match self {
            AppError::BadRequest(m) => (
                StatusCode::BAD_REQUEST,
                m,
                "invalid_request_error",
                "invalid_request",
            ),
            AppError::ModelNotFound(m) => (
                StatusCode::NOT_FOUND,
                m,
                "invalid_request_error",
                "model_not_found",
            ),
            AppError::ModelLoadFailed(m) => (
                StatusCode::SERVICE_UNAVAILABLE,
                m,
                "server_error",
                "model_load_failed",
            ),
            AppError::EmbeddingFailed(m) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                m,
                "server_error",
                "embedding_failed",
            ),
            AppError::InternalError(m) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                m,
                "server_error",
                "internal_error",
            ),
        };
        (
            status,
            Json(json!({
                "error": {
                    "message": msg,
                    "type": err_type,
                    "param": null,
                    "code": code
                }
            })),
        )
            .into_response()
    }
}

/// Resolve and load a model, logging the right message depending on whether the
/// failure is "model not on disk" (404) or "model failed to load" (e.g. OOM → 503).
///
/// Returns `Ok(entry)` or an `Err(Response)` ready to return from a handler.
pub async fn load_model_or_respond(
    registry: &ModelRegistry,
    model: &str,
) -> Result<Arc<EngineEntry>, Response> {
    // Try to load (get_or_load checks the engine cache first, then resolves on disk).
    match registry.get_or_load(model).await {
        Ok(entry) => Ok(entry),
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("not found") {
                tracing::warn!(model = %model, error = %e, "model not found");
                Err(AppError::ModelNotFound(msg).into_response())
            } else {
                tracing::error!(model = %model, error = %e, "failed to load model");
                Err(AppError::ModelLoadFailed(msg).into_response())
            }
        }
    }
}
