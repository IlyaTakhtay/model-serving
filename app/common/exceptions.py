from __future__ import annotations


class ServingError(Exception):
    code = "SERVING_ERROR"
    status_code = 400
    layer = "common"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ModelNotFoundError(ServingError):
    code = "MODEL_NOT_FOUND"
    status_code = 404
    layer = "registry"


class VersionNotFoundError(ServingError):
    code = "VERSION_NOT_FOUND"
    status_code = 404
    layer = "registry"


class ModelNotReadyError(ServingError):
    code = "MODEL_NOT_READY"
    status_code = 503
    layer = "runtime"


class ResourceBudgetExceededError(ServingError):
    code = "RESOURCE_BUDGET_EXCEEDED"
    status_code = 409
    layer = "runtime"


class InvalidTensorError(ServingError):
    code = "INVALID_TENSOR"
    status_code = 422
    layer = "inference"


class InvalidUploadError(ServingError):
    code = "INVALID_UPLOAD"
    status_code = 422
    layer = "inference"


class InvalidRequestError(ServingError):
    code = "INVALID_REQUEST"
    status_code = 400
    layer = "api"


class InvalidInferenceProtocolError(InvalidRequestError):
    code = "INVALID_INFERENCE_PROTOCOL"


class ActiveModelStateError(ServingError):
    code = "ACTIVE_MODEL_STATE_ERROR"
    status_code = 500
    layer = "control_plane"


class ModelStorageError(ServingError):
    code = "MODEL_STORAGE_ERROR"
    status_code = 500
    layer = "control_plane"


class WorkerStartupError(ServingError):
    code = "WORKER_STARTUP_FAILED"
    status_code = 503
    layer = "runtime"


class WorkerCommunicationError(ServingError):
    code = "WORKER_COMMUNICATION_FAILED"
    status_code = 503
    layer = "runtime"


class WorkerCrashedError(ServingError):
    code = "WORKER_CRASHED"
    status_code = 503
    layer = "runtime"


class RuntimeInferenceError(ServingError):
    code = "RUNTIME_INFERENCE_FAILED"
    status_code = 422
    layer = "runtime"


class ModelHealthCheckError(ServingError):
    code = "MODEL_HEALTHCHECK_FAILED"
    status_code = 503
    layer = "control_plane"
