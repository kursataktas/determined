CREATE SEQUENCE IF NOT EXISTS stream_experiment_seq START 1;

ALTER TABLE experiments ADD COLUMN IF NOT EXISTS seq bigint DEFAULT 0;