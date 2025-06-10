import sqlite3
import json
from pathlib import Path


HERE = Path(__file__).parent
DB_PATH = HERE / "chunks.db"
_con = sqlite3.connect(DB_PATH, check_same_thread=False)


def connect(con=_con):
    con.executescript(
        """
        PRAGMA foreign_keys = ON;
        
        DROP TABLE IF EXISTS chunksets;
        DROP TABLE IF EXISTS chunks;

        CREATE TABLE IF NOT EXISTS chunks(
            doc_id TEXT, chunk_index INT, depth INT, content TEXT,
            PRIMARY KEY(doc_id, chunk_index));
        CREATE TABLE IF NOT EXISTS chunksets(
            doc_id TEXT, chunkset_index INT, chunks TEXT, contents TEXT,
            PRIMARY KEY(doc_id, chunkset_index));
    """
    )


def _save_many(sql, data, con=_con):
    con.executemany(sql, data)
    con.commit()


def save_chunks_and_chunksets(doc_id, raw_chunks, chunksets, con=_con):
    _save_many(
        "INSERT OR IGNORE INTO chunksets VALUES (?, ?, ?, ?)",
        [
            (doc_id, cs["chunkset_index"], json.dumps(cs["chunks"]), cs["contents"])
            for cs in chunksets
        ],
    )
    _save_many(
        "INSERT OR IGNORE INTO chunks VALUES (?, ?, ?, ?)",
        [(doc_id, c["chunk_index"], c["depth"], c["content"]) for c in raw_chunks],
    )


def fetch_chunks(doc_id, con=_con):
    rows = con.execute(
        "SELECT chunk_index, depth, content FROM chunks WHERE doc_id=? ORDER BY chunk_index",
        (doc_id,),
    ).fetchall()
    return [{"chunk_index": r[0], "depth": r[1], "content": r[2]} for r in rows]


def fetch_chunkset(doc_id, chunkset_index, con=_con):
    row = con.execute(
        "SELECT chunkset_index, chunks, contents FROM chunksets WHERE doc_id=? AND chunkset_index=?",
        (doc_id, chunkset_index),
    ).fetchone()
    if row:
        return {
            "chunkset_index": row[0],
            "chunks": json.loads(row[1]),
            "contents": row[2],
        }
    return {}
