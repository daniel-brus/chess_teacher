import sqlite3

from config.config import configuration


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(configuration.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS games (
                id          TEXT PRIMARY KEY,  -- chess.com game uuid
                pgn         TEXT NOT NULL,
                white       TEXT NOT NULL,
                black       TEXT NOT NULL,
                result      TEXT NOT NULL,
                time_class  TEXT,              -- bullet/blitz/rapid
                played_at   INTEGER,           -- unix timestamp
                analyzed    INTEGER DEFAULT 0  -- boolean flag
            )
        """)


def save_games(games: list[dict]) -> int:
    """Sla nieuwe potjes op, skip duplicaten. Returnt aantal nieuwe."""
    saved = 0
    with get_connection() as conn:
        for game in games:
            try:
                conn.execute(
                    """
                    INSERT INTO games (id, pgn, white, black, result, time_class, played_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        str(game["url"].split("/")[-1]),
                        game.get("pgn", ""),
                        game["white"]["username"],
                        game["black"]["username"],
                        game["white"]["result"],
                        game.get("time_class"),
                        game.get("end_time"),
                    ),
                )
                saved += 1
            except sqlite3.IntegrityError:
                pass  # duplicate, skip
    return saved


def get_unanalyzed_games() -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute("SELECT * FROM games WHERE analyzed = 0").fetchall()


def mark_analyzed(game_id: str) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE games SET analyzed = 1 WHERE id = ?", (game_id,))


def get_last_fetched() -> int | None:
    """Returnt unix timestamp van het laatste opgeslagen potje."""
    with get_connection() as conn:
        row = conn.execute("SELECT MAX(played_at) FROM games").fetchone()
        return row[0] if row else None
