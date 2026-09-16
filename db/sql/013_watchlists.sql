/*
============================================================
AlphaLens - Watchlists
============================================================

Purpose:
    Store named single-user ticker groups that later scheduled ingestion,
    event alerts, and automatic briefs can use as monitoring scope.

Behavior:
    Membership is idempotent, company deletion removes stale membership,
    and deleting a watchlist cascades only to its own membership rows.
============================================================
*/


CREATE TABLE IF NOT EXISTS watchlists (

    watchlist_id BIGSERIAL PRIMARY KEY,

    name VARCHAR(80) NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


CREATE UNIQUE INDEX IF NOT EXISTS idx_watchlists_name_unique
ON watchlists (LOWER(name));


CREATE TABLE IF NOT EXISTS watchlist_items (

    watchlist_id BIGINT NOT NULL
        REFERENCES watchlists(watchlist_id)
        ON DELETE CASCADE,

    ticker VARCHAR(20) NOT NULL
        REFERENCES companies(ticker)
        ON DELETE CASCADE,

    added_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (watchlist_id, ticker)
);


CREATE INDEX IF NOT EXISTS idx_watchlist_items_ticker
ON watchlist_items (ticker, watchlist_id);


-- A first-run default makes the feature usable immediately while retaining
-- normal named-list CRUD for users who want separate research themes.
INSERT INTO watchlists (name)
SELECT 'Core Watchlist'
WHERE NOT EXISTS (SELECT 1 FROM watchlists);
