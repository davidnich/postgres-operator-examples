-- =============================================================================
-- Qdrant Vector Search Infrastructure — Schema Migration
-- Idempotent: safe to run multiple times
-- Requires: pgvector extension
-- =============================================================================

BEGIN;

-- Verify pgvector
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        RAISE EXCEPTION 'pgvector extension not installed. Run: CREATE EXTENSION vector;';
    END IF;
END $$;

-- ─── Source Documents ────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS qdrant_documents (
    document_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID NOT NULL,
    source_type     TEXT NOT NULL,
    source_ref      TEXT NOT NULL,
    title           TEXT,
    raw_content     TEXT NOT NULL,
    content_hash    BYTEA NOT NULL,
    content_size    INTEGER NOT NULL,
    mime_type       TEXT,
    language        TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_tenant ON qdrant_documents(tenant_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_source_dedup
    ON qdrant_documents(tenant_id, source_type, source_ref);
CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON qdrant_documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_created ON qdrant_documents(created_at DESC);

-- ─── Document Chunks ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS qdrant_chunks (
    chunk_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL
                    REFERENCES qdrant_documents(document_id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT NOT NULL,
    chunk_hash      BYTEA NOT NULL,
    token_count     INTEGER,
    char_start      INTEGER,
    char_end        INTEGER,
    overlap_prev    INTEGER DEFAULT 0,
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON qdrant_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON qdrant_chunks(chunk_hash);

-- ─── Embeddings (pgvector) ───────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS qdrant_embeddings (
    chunk_id            UUID NOT NULL
                        REFERENCES qdrant_chunks(chunk_id) ON DELETE CASCADE,
    model_name          TEXT NOT NULL DEFAULT 'bge-m3',
    model_version       TEXT NOT NULL,
    dense_vector        vector(1024) NOT NULL,
    sparse_vector       sparsevec,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (chunk_id, model_name)
);

-- HNSW index for fallback dense vector search
CREATE INDEX IF NOT EXISTS idx_embeddings_dense_hnsw
    ON qdrant_embeddings USING hnsw (dense_vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

-- Sparse HNSW — may fail on older pgvector, that's OK
DO $$ BEGIN
    CREATE INDEX IF NOT EXISTS idx_embeddings_sparse_hnsw
        ON qdrant_embeddings USING hnsw (sparse_vector sparsevec_cosine_ops)
        WITH (m = 16, ef_construction = 128);
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Sparse HNSW index skipped (pgvector may not support it): %', SQLERRM;
END $$;

CREATE INDEX IF NOT EXISTS idx_embeddings_model_version
    ON qdrant_embeddings(model_name, model_version);

-- ─── Qdrant Sync State ───────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS qdrant_sync_state (
    chunk_id            UUID NOT NULL,
    model_name          TEXT NOT NULL,
    qdrant_collection   TEXT NOT NULL,
    qdrant_point_id     TEXT NOT NULL,
    synced_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding_created   TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (chunk_id, model_name, qdrant_collection),
    FOREIGN KEY (chunk_id, model_name)
        REFERENCES qdrant_embeddings(chunk_id, model_name) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sync_collection ON qdrant_sync_state(qdrant_collection);
CREATE INDEX IF NOT EXISTS idx_sync_synced_at ON qdrant_sync_state(synced_at DESC);

-- ─── Ingestion Log ───────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS qdrant_ingestion_log (
    ingestion_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL
                    REFERENCES qdrant_documents(document_id) ON DELETE CASCADE,
    tenant_id       UUID NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    error_message   TEXT,
    error_detail    TEXT,
    retry_count     INTEGER NOT NULL DEFAULT 0,
    chunks_total    INTEGER,
    chunks_embedded INTEGER NOT NULL DEFAULT 0,
    chunks_synced   INTEGER NOT NULL DEFAULT 0,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ,
    chunk_config    JSONB,
    embed_config    JSONB
);

CREATE INDEX IF NOT EXISTS idx_ingestion_active
    ON qdrant_ingestion_log(status) WHERE status NOT IN ('done', 'error');
CREATE INDEX IF NOT EXISTS idx_ingestion_document ON qdrant_ingestion_log(document_id);
CREATE INDEX IF NOT EXISTS idx_ingestion_tenant_status ON qdrant_ingestion_log(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_ingestion_retry
    ON qdrant_ingestion_log(status, retry_count) WHERE status = 'error' AND retry_count < 3;

-- ─── Collection Config ───────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS qdrant_collection_config (
    collection_name     TEXT PRIMARY KEY,
    tenant_id           UUID,
    description         TEXT,
    vector_config       JSONB NOT NULL,
    sparse_vector_config JSONB,
    hnsw_config         JSONB,
    optimizers_config   JSONB,
    payload_indexes     JSONB NOT NULL DEFAULT '[]',
    model_name          TEXT NOT NULL DEFAULT 'bge-m3',
    model_version       TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ─── Helper Functions ────────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION qdrant_point_id(
    p_chunk_id UUID, p_model_name TEXT DEFAULT 'bge-m3'
) RETURNS TEXT AS $$
BEGIN
    IF p_model_name = 'bge-m3' THEN RETURN p_chunk_id::TEXT;
    ELSE RETURN p_model_name || ':' || p_chunk_id::TEXT;
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE FUNCTION qdrant_unsynced_embeddings(
    p_collection TEXT, p_model_name TEXT DEFAULT 'bge-m3', p_limit INTEGER DEFAULT 256
) RETURNS TABLE (
    chunk_id UUID, document_id UUID, chunk_text TEXT, chunk_index INTEGER,
    chunk_metadata JSONB, doc_title TEXT, doc_source_ref TEXT, doc_source_type TEXT,
    doc_metadata JSONB, tenant_id UUID, dense_vector vector, sparse_vector sparsevec,
    model_version TEXT, embedding_created TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT e.chunk_id, c.document_id, c.chunk_text, c.chunk_index,
        c.metadata, d.title, d.source_ref, d.source_type, d.metadata,
        d.tenant_id, e.dense_vector, e.sparse_vector, e.model_version, e.created_at
    FROM qdrant_embeddings e
    JOIN qdrant_chunks c ON c.chunk_id = e.chunk_id
    JOIN qdrant_documents d ON d.document_id = c.document_id
    LEFT JOIN qdrant_sync_state s ON s.chunk_id = e.chunk_id
        AND s.model_name = e.model_name AND s.qdrant_collection = p_collection
    WHERE e.model_name = p_model_name AND s.chunk_id IS NULL
    ORDER BY e.created_at ASC LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

CREATE OR REPLACE FUNCTION qdrant_fallback_search(
    p_query_vector vector, p_tenant_id UUID, p_model_name TEXT DEFAULT 'bge-m3',
    p_limit INTEGER DEFAULT 10, p_score_threshold FLOAT DEFAULT 0.0
) RETURNS TABLE (
    chunk_id UUID, document_id UUID, chunk_text TEXT, chunk_index INTEGER,
    chunk_metadata JSONB, doc_title TEXT, doc_source_ref TEXT, score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT e.chunk_id, c.document_id, c.chunk_text, c.chunk_index,
        c.metadata, d.title, d.source_ref,
        (1 - (e.dense_vector <=> p_query_vector))::FLOAT AS score
    FROM qdrant_embeddings e
    JOIN qdrant_chunks c ON c.chunk_id = e.chunk_id
    JOIN qdrant_documents d ON d.document_id = c.document_id
    WHERE e.model_name = p_model_name AND d.tenant_id = p_tenant_id
      AND (1 - (e.dense_vector <=> p_query_vector)) >= p_score_threshold
    ORDER BY e.dense_vector <=> p_query_vector LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- ─── Triggers ────────────────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION qdrant_update_timestamp() RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = now(); RETURN NEW; END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_documents_updated ON qdrant_documents;
CREATE TRIGGER trg_documents_updated BEFORE UPDATE ON qdrant_documents
    FOR EACH ROW EXECUTE FUNCTION qdrant_update_timestamp();

DROP TRIGGER IF EXISTS trg_collection_config_updated ON qdrant_collection_config;
CREATE TRIGGER trg_collection_config_updated BEFORE UPDATE ON qdrant_collection_config
    FOR EACH ROW EXECUTE FUNCTION qdrant_update_timestamp();

-- ─── Seed Default Collection Config ──────────────────────────────────────────

INSERT INTO qdrant_collection_config (
    collection_name, description, vector_config, sparse_vector_config,
    optimizers_config, payload_indexes, model_name, model_version
) VALUES (
    'documents',
    'Default BGE-M3 dense+sparse vector search collection',
    '{"dense": {"size": 1024, "distance": "Cosine"}}',
    '{"sparse": {}}',
    '{"indexing_threshold": 20000}',
    '[{"field": "tenant_id", "schema": "keyword"},
      {"field": "document_id", "schema": "keyword"},
      {"field": "source_type", "schema": "keyword"},
      {"field": "chunk_index", "schema": "integer"}]',
    'bge-m3', '1.0.0'
) ON CONFLICT (collection_name) DO NOTHING;

-- ─── Verification ────────────────────────────────────────────────────────────

DO $$
DECLARE
    missing TEXT := '';
    tbl TEXT;
BEGIN
    FOREACH tbl IN ARRAY ARRAY[
        'qdrant_documents', 'qdrant_chunks', 'qdrant_embeddings',
        'qdrant_sync_state', 'qdrant_ingestion_log', 'qdrant_collection_config'
    ] LOOP
        IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = tbl) THEN
            missing := missing || tbl || ', ';
        END IF;
    END LOOP;
    IF missing != '' THEN
        RAISE EXCEPTION 'Missing tables: %', rtrim(missing, ', ');
    ELSE
        RAISE NOTICE 'All 6 qdrant_* tables created successfully';
    END IF;
END $$;

DO $$ BEGIN
    PERFORM '[1,2,3]'::vector;
    RAISE NOTICE 'pgvector vector type works';
EXCEPTION WHEN OTHERS THEN
    RAISE EXCEPTION 'pgvector vector type not working: %', SQLERRM;
END $$;

DO $$ BEGIN
    PERFORM qdrant_point_id(gen_random_uuid());
    RAISE NOTICE 'qdrant_point_id function works';
END $$;

DO $$ BEGIN
    PERFORM * FROM qdrant_collection_config WHERE collection_name = 'documents';
    RAISE NOTICE 'Default collection config seeded';
END $$;

COMMIT;
