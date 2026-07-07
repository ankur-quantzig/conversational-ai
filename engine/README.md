# Engine

Retrieval, RAG answer generation, model clients, document/video processing pipelines, schemas, and shared utilities.

This folder is imported by the backend as part of the same `app.*` namespace package. Keep engine code free of UI concerns and only expose behavior through Python modules used by the backend or pipeline scripts.
