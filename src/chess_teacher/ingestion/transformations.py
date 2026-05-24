from __future__ import annotations

import re
from datetime import date
from pathlib import PurePosixPath
from typing import ClassVar

import polars as pl

from chess_teacher.pipelines.transformations import DataFrameTransformation
from chess_teacher.platform.account import AccountPlatform
from chess_teacher.utils.chess_utils import Color, Reason, Result
from chess_teacher.utils.exception_utils import DataError, TransformationError
from chess_teacher.utils.logging_utils import get_logger

logger = get_logger()


class ExtractFileMetadataTransformation(DataFrameTransformation):
    """
    Extract ingestion file metadata from ``_source_file`` paths.

    Expected layout:
        .../ingested/{account_id}/{YYYY}/{MM}/{DD}/{platform}_{batch_id}.jsonl
    """

    SOURCE_FILE_COLUMN = "_source_file"
    INGESTED_FOLDER = "ingested"
    _SOURCE_FILE_PATH_RE = re.compile(
        rf"(?:^|.*/){re.escape(INGESTED_FOLDER)}"
        r"/(?P<account_id>[^/]+)/(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2})/(?P<file_name>[^/]+)$"
    )

    @staticmethod
    def _empty_metadata() -> dict[str, str | date | None]:
        return {
            "account_id": None,
            "ingestion_date": None,
            "file_name": None,
        }

    @classmethod
    def _parse_source_file_path(cls, source_file: str) -> dict[str, str | date | None]:
        """Parse account_id, ingestion_date, file_name from a _source_file path."""
        if not source_file:
            return cls._empty_metadata()

        normalized = source_file.replace("\\", "/")
        match = cls._SOURCE_FILE_PATH_RE.search(normalized)
        if match is None:
            return cls._parse_source_file_path_fallback(normalized)

        file_name = match.group("file_name")
        return {
            "account_id": match.group("account_id"),
            "ingestion_date": date(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
            ),
            "file_name": file_name,
        }

    @classmethod
    def _parse_source_file_path_fallback(cls, normalized: str) -> dict[str, str | date | None]:
        """Fallback parser using path parts when the regex does not match."""
        parts = PurePosixPath(normalized).parts
        try:
            ingested_idx = parts.index(cls.INGESTED_FOLDER)
        except ValueError:
            return cls._empty_metadata()

        tail = parts[ingested_idx + 1 :]
        if len(tail) < 5:
            file_name = tail[-1] if tail else None
            return {
                "account_id": tail[0] if tail else None,
                "ingestion_date": None,
                "file_name": file_name,
            }

        account_id, year, month, day, file_name = tail[0], tail[1], tail[2], tail[3], tail[4]
        ingestion_date: date | None = None
        if len(year) == 4 and year.isdigit() and month.isdigit() and day.isdigit():
            ingestion_date = date(int(year), int(month), int(day))

        return {
            "account_id": account_id,
            "ingestion_date": ingestion_date,
            "file_name": file_name,
        }

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.SOURCE_FILE_COLUMN not in df.columns:
            logger.log_and_raise(
                TransformationError(
                    f"Column {self.SOURCE_FILE_COLUMN!r} is required for file metadata extraction."
                )
            )

        try:
            result = df.with_columns(
                pl.col(self.SOURCE_FILE_COLUMN)
                .map_elements(
                    self._parse_source_file_path,
                    return_dtype=pl.Struct({
                        "account_id": pl.Utf8,
                        "ingestion_date": pl.Date,
                        "file_name": pl.Utf8,
                    }),
                )
                .alias("_file_metadata")
            ).unnest("_file_metadata")
        except Exception as e:
            logger.log_and_raise(TransformationError(f"Failed to extract file metadata: {e}"))

        unparsed = result.filter(pl.col("account_id").is_null()).height
        if unparsed:
            logger.warning(
                "ExtractFileMetadataTransformation: %s row(s) could not be parsed from %s.",
                unparsed,
                self.SOURCE_FILE_COLUMN,
            )

        return result


class ExtractPlayersAndResultTransformation(DataFrameTransformation):
    """
    Extract the color of the user from the PGN,
    the result (win/draw/loss) and reason for the result.
    Also derive the user's and opponent's ELO rating at the start of the game.
    Requires:
    - the input DataFrame to contain the 'username' column.
    - the input DataFrame to contain the 'platform' column.
    Returns the input DataFrame with only these columns added (or updated if already present):
    - color (Color.value: which color the user is playing as)
    - result (Result.value: the result of the game from the user's perspective)
    - reason (Reason.value: the reason for the result)
    - user_elo (int: the user's ELO rating at the start of the game)
    - opponent_elo (int: the opponent's ELO rating at the start of the game)
    """

    _OUTPUT_COLUMNS = ("color", "result", "reason", "user_elo", "opponent_elo")

    CHESS_COM = AccountPlatform.CHESS_COM.value
    LICHESS = AccountPlatform.LICHESS.value

    _PGN_TERMINATION_RE = re.compile(r'\[Termination\s+"([^"]+)"\]', re.IGNORECASE)
    _PGN_RESULT_RE = re.compile(r'\[Result\s+"([^"]+)"\]', re.IGNORECASE)

    _CHESS_COM_REASON_MAP: ClassVar[dict[str, Reason]] = {
        "checkmated": Reason.CHECKMATE,
        "resigned": Reason.RESIGNATION,
        "timeout": Reason.TIMEOUT,
        "stalemate": Reason.STALEMATE,
        "insufficient": Reason.INSUFFICIENT_MATERIAL,
        "timevsinsufficient": Reason.TIMEOUT_INSUFFICIENT_MATERIAL,
        "repetition": Reason.THREEFOLD_REPETITION,
        "agreed": Reason.AGREED_DRAW,
        "50move": Reason.FIFTY_MOVE_RULE,
        "abandoned": Reason.ABANDONED,
        "kingofthehill": Reason.OTHER,
        "threecheck": Reason.OTHER,
        "bughousepartnerlose": Reason.OTHER,
    }

    _DRAW_REASONS = frozenset({
        Reason.STALEMATE,
        Reason.INSUFFICIENT_MATERIAL,
        Reason.TIMEOUT_INSUFFICIENT_MATERIAL,
        Reason.THREEFOLD_REPETITION,
        Reason.AGREED_DRAW,
        Reason.FIFTY_MOVE_RULE,
    })

    # Covers all values from Lichess GameStatusName; unlisted future values fall back to OTHER.
    _LICHESS_STATUS_REASON_MAP: ClassVar[dict[str, Reason]] = {
        "mate": Reason.CHECKMATE,
        "resign": Reason.RESIGNATION,
        "outoftime": Reason.TIMEOUT,
        "stalemate": Reason.STALEMATE,
        "draw": Reason.OTHER,  # disambiguated from PGN Termination when possible
        "insufficientMaterialClaim": Reason.INSUFFICIENT_MATERIAL,
        "timeout": Reason.ABANDONED,
        "aborted": Reason.ABANDONED,
        "noStart": Reason.ABANDONED,
        "cheat": Reason.OTHER,
        "variantEnd": Reason.OTHER,
        "unknownFinish": Reason.OTHER,
        "created": Reason.OTHER,
        "started": Reason.OTHER,
    }

    _LICHESS_DRAW_TERMINATION_MAP: ClassVar[dict[str, Reason]] = {
        "Draw by mutual agreement": Reason.AGREED_DRAW,
        "Draw by repetition": Reason.THREEFOLD_REPETITION,
        "Draw by insufficient material": Reason.INSUFFICIENT_MATERIAL,
        "Draw by the 50-move rule": Reason.FIFTY_MOVE_RULE,
        "Draw by stalemate": Reason.STALEMATE,
    }

    # Always a draw; only ``draw`` needs PGN Termination disambiguation.
    _LICHESS_DRAW_STATUSES = frozenset({"draw", "stalemate", "insufficientMaterialClaim"})
    # In-progress exports that should not appear in finished-game ingestion.
    _LICHESS_INCOMPLETE_STATUSES = frozenset({"created", "started"})
    # Finished games where winner is normally set (fallback: PGN Result tag).
    _LICHESS_DECISIVE_STATUSES = frozenset({"mate", "resign", "outoftime"})

    @classmethod
    def _is_chess_com(cls) -> pl.Expr:
        """True when the row is from Chess.com."""
        return pl.col("platform") == cls.CHESS_COM

    @classmethod
    def _is_lichess(cls) -> pl.Expr:
        """True when the row is from Lichess."""
        return pl.col("platform") == cls.LICHESS

    @classmethod
    def _chain_when(cls, branches: list[tuple[pl.Expr, pl.Expr]], *, default: pl.Expr) -> pl.Expr:
        """Fold ``(condition, value)`` pairs into a nested ``pl.when`` chain."""
        expr = default
        for condition, value in reversed(branches):
            expr = pl.when(condition).then(value).otherwise(expr)
        return expr

    @classmethod
    def _side_username_expr(cls, columns: set[str], side: Color) -> pl.Expr:
        """Username for ``side`` (white or black), per platform schema."""
        side_value = side.value
        branches: list[tuple[pl.Expr, pl.Expr]] = []
        if side_value in columns:
            branches.append((cls._is_chess_com(), pl.col(side_value).struct.field("username")))
        if "players" in columns:
            branches.append((
                cls._is_lichess(),
                pl.col("players")
                .struct.field(side_value)
                .struct.field("user")
                .struct.field("name"),
            ))
        return cls._chain_when(branches, default=pl.lit(None).cast(pl.Utf8))

    @classmethod
    def _side_rating_expr(cls, columns: set[str], side: Color) -> pl.Expr:
        """Pre-game rating for ``side`` (white or black), per platform schema. Returns None if the rating is not present."""
        side_value = side.value
        branches: list[tuple[pl.Expr, pl.Expr]] = []
        if side_value in columns:
            branches.append((cls._is_chess_com(), pl.col(side_value).struct.field("rating")))
        if "players" in columns:
            branches.append((
                cls._is_lichess(),
                pl.col("players").struct.field(side_value).struct.field("rating"),
            ))
        return cls._chain_when(branches, default=pl.lit(None))

    @classmethod
    def _parse_pgn_termination(cls, pgn: str | None) -> str | None:
        """Extract the Termination tag value from a PGN string."""
        if not pgn:
            return None
        match = cls._PGN_TERMINATION_RE.search(pgn)
        return match.group(1) if match else None

    @classmethod
    def _parse_pgn_result(cls, pgn: str | None) -> str | None:
        """Extract the Result tag value from a PGN string."""
        if not pgn:
            return None
        match = cls._PGN_RESULT_RE.search(pgn)
        return match.group(1) if match else None

    @classmethod
    def _result_from_pgn_tag(cls, color: str, pgn: str | None) -> Result | None:
        """Map a PGN Result tag to the user's result from their color."""
        result_tag = cls._parse_pgn_result(pgn)
        if result_tag == "1-0":
            return Result.WIN if color == Color.WHITE.value else Result.LOSS
        if result_tag == "0-1":
            return Result.LOSS if color == Color.WHITE.value else Result.WIN
        if result_tag == "1/2-1/2":
            return Result.DRAW
        if result_tag == "*":
            return Result.NO_RESULT
        return None

    @classmethod
    def _result_from_lichess_winner(cls, color: str, winner: str) -> Result:
        """Map Lichess ``winner`` to the user's win or loss."""
        if winner not in (Color.WHITE.value, Color.BLACK.value):
            raise DataError(f"Unknown Lichess winner value: {winner!r}.")
        return Result.WIN if color == winner else Result.LOSS

    @classmethod
    def _lichess_draw_reason_from_pgn(cls, pgn: str | None) -> Reason:
        """Refine a generic Lichess draw using the PGN Termination tag."""
        termination = cls._parse_pgn_termination(pgn)
        if termination is None:
            return Reason.OTHER
        return cls._LICHESS_DRAW_TERMINATION_MAP.get(termination, Reason.OTHER)

    @classmethod
    def _result_from_reason(cls, reason: Reason, *, user_won: bool) -> Result:
        """Derive win/loss/draw from a termination reason and whether the user won."""
        if reason in cls._DRAW_REASONS:
            return Result.DRAW
        if user_won:
            return Result.WIN
        return Result.LOSS

    @classmethod
    def _map_chess_com_reason(cls, result_code: str) -> Reason:
        """Map a Chess.com player result code to a ``Reason``."""
        reason = cls._CHESS_COM_REASON_MAP.get(result_code)
        if reason is None:
            raise DataError(f"Unknown Chess.com player result code: {result_code!r}.")
        return reason

    @classmethod
    def _extract_chess_com_result_reason(
        cls, color: str, white: dict | None, black: dict | None
    ) -> tuple[Result, Reason]:
        """Derive result and reason from Chess.com white/black player structs."""
        if not white or not black:
            raise DataError("Chess.com game is missing white or black player data.")

        if color == Color.WHITE.value:
            user_side = white.get("result", None)
            opponent_side = black.get("result", None)
        elif color == Color.BLACK.value:
            user_side = black.get("result", None)
            opponent_side = white.get("result", None)
        else:
            raise TransformationError(f"Unknown color: {color!r}.")

        if user_side is None or opponent_side is None:
            raise DataError("Chess.com game is missing a player result code.")

        if user_side == "win":
            reason = cls._map_chess_com_reason(opponent_side)
            return Result.WIN, reason

        reason = cls._map_chess_com_reason(user_side)
        result = cls._result_from_reason(reason, user_won=False)
        return result, reason

    @classmethod
    def _finalize_lichess_result_reason(
        cls, result: Result, reason: Reason, *, pgn: str | None
    ) -> tuple[Result, Reason]:
        """Apply PGN-based draw reason refinement to a resolved Lichess outcome."""
        if result == Result.DRAW:
            reason = cls._lichess_draw_reason_from_pgn(pgn)
        return result, reason

    @classmethod
    def _resolve_and_finalize_lichess(
        cls,
        color: str,
        reason: Reason,
        *,
        winner: str | None,
        pgn: str | None,
        allow_no_result: bool,
    ) -> tuple[Result, Reason]:
        """Resolve Lichess outcome from winner/PGN, then finalize draw reason if needed."""
        result = cls._resolve_lichess_outcome(
            color, winner=winner, pgn=pgn, allow_no_result=allow_no_result
        )
        return cls._finalize_lichess_result_reason(result, reason, pgn=pgn)

    @classmethod
    def _resolve_lichess_outcome(
        cls,
        color: str,
        *,
        winner: str | None,
        pgn: str | None,
        allow_no_result: bool,
    ) -> Result:
        """Resolve result from Lichess winner, PGN Result tag, or no-result fallback."""
        if winner is not None:
            return cls._result_from_lichess_winner(color, winner)

        pgn_result = cls._result_from_pgn_tag(color, pgn)
        if pgn_result is not None:
            return pgn_result

        if allow_no_result:
            return Result.NO_RESULT

        raise DataError(
            "Lichess game has no winner and no decisive PGN Result (expected 1-0, 0-1, or 1/2-1/2)."
        )

    @classmethod
    def _extract_lichess_result_reason(
        cls, color: str, status: str | None, winner: str | None, pgn: str | None
    ) -> tuple[Result, Reason]:
        """Derive result and reason from Lichess status, winner, and PGN."""
        if not status:
            raise DataError("Lichess game is missing status.")

        reason = cls._LICHESS_STATUS_REASON_MAP.get(status, Reason.OTHER)

        if status in cls._LICHESS_INCOMPLETE_STATUSES:
            return Result.NO_RESULT, reason

        if status in cls._LICHESS_DRAW_STATUSES:
            if status == "draw":
                reason = cls._lichess_draw_reason_from_pgn(pgn)
            return Result.DRAW, reason

        if status in cls._LICHESS_DECISIVE_STATUSES:
            return cls._resolve_and_finalize_lichess(
                color, reason, winner=winner, pgn=pgn, allow_no_result=False
            )

        # Optional-outcome statuses and any future API value: same resolve path.
        return cls._resolve_and_finalize_lichess(
            color, reason, winner=winner, pgn=pgn, allow_no_result=True
        )

    @classmethod
    def _extract_result_reason_row(cls, row: dict) -> dict[str, str]:
        """Per-row handler: extract ``result`` and ``reason`` for one game."""
        platform = row.get("platform")
        color = row.get("color")
        if not color:
            raise DataError("Missing player color for result extraction.")

        try:
            if platform == cls.CHESS_COM:
                result_reason = cls._extract_chess_com_result_reason(
                    color, row.get("white"), row.get("black")
                )
            elif platform == cls.LICHESS:
                result_reason = cls._extract_lichess_result_reason(
                    color, row.get("status"), row.get("winner"), row.get("pgn")
                )
            else:
                raise DataError(f"Unsupported platform for result extraction: {platform!r}.")
        except DataError as e:
            logger.log_and_raise(e)
        except Exception as e:
            logger.log_and_raise(TransformationError(f"Failed to extract result and reason: {e}"))

        result, reason = result_reason
        return {"result": result.value, "reason": reason.value}

    @classmethod
    def _raise_non_unique_player_match(cls, df: pl.DataFrame) -> None:
        """Raise if ``username`` matches zero or both sides in any row."""
        bad_rows = df.filter(pl.col("_match_count") != 1)
        if not bad_rows.height:
            return

        sample = bad_rows.select("username", "platform", "_match_count").row(0, named=True)
        count = sample["_match_count"]
        if count == 0:
            detail = "matched neither white nor black"
        else:
            detail = "matched both white and black"

        raise DataError(
            f"Account username {sample["username"]!r} on {sample["platform"]!r} "
            f"does not uniquely identify a player color ({detail}). "
            f"{bad_rows.height} row(s) affected."
        )

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add color, ELOs, result, and reason columns."""
        input_columns = list(df.columns)

        if "username" not in df.columns or "platform" not in df.columns:
            logger.log_and_raise(
                TransformationError(
                    "Columns 'username' and 'platform' are required for player extraction."
                )
            )

        unknown_platform = df.filter(~self._is_chess_com() & ~self._is_lichess())
        if unknown_platform.height:
            platforms = unknown_platform.get_column("platform").unique().to_list()
            logger.log_and_raise(
                TransformationError(
                    f"Unsupported platform value(s) for player extraction: {platforms!r}."
                )
            )

        try:
            # Determine the color of the user and the opponent
            column_names = set(df.columns)
            account_username = pl.col("username").str.to_lowercase()
            white_username = self._side_username_expr(column_names, Color.WHITE).str.to_lowercase()
            black_username = self._side_username_expr(column_names, Color.BLACK).str.to_lowercase()

            white_match = account_username == white_username
            black_match = account_username == black_username
            match_count = white_match.cast(pl.Int8) + black_match.cast(pl.Int8)

            working = df.with_columns(
                _match_count=match_count,
                _white_match=white_match,
            )
            self._raise_non_unique_player_match(working)

            # Add the color, user_elo, and opponent_elo columns
            df = working.with_columns(
                color=pl.when(pl.col("_white_match"))
                .then(pl.lit(Color.WHITE.value))
                .otherwise(pl.lit(Color.BLACK.value)),
                user_elo=pl.when(pl.col("_white_match"))
                .then(self._side_rating_expr(column_names, Color.WHITE))
                .otherwise(self._side_rating_expr(column_names, Color.BLACK)),
                opponent_elo=pl.when(pl.col("_white_match"))
                .then(self._side_rating_expr(column_names, Color.BLACK))
                .otherwise(self._side_rating_expr(column_names, Color.WHITE)),
            ).drop("_match_count", "_white_match")
        except (TransformationError, DataError) as e:
            logger.log_and_raise(e)
        except Exception as e:
            logger.log_and_raise(
                TransformationError(f"Failed to extract player and result information: {e}")
            )

        # Extract the result and reason, for which the following columns are required:
        required_columns = ["platform", "color", "white", "black", "status", "winner", "pgn"]
        try:
            df = df.with_columns(
                pl.struct([pl.col(column) for column in required_columns if column in df.columns])
                .map_elements(
                    self._extract_result_reason_row,
                    return_dtype=pl.Struct({
                        "result": pl.Utf8,
                        "reason": pl.Utf8,
                    }),
                )
                .alias("_result_reason")
            ).unnest("_result_reason")
        except DataError as e:
            logger.log_and_raise(e)
        except Exception as e:
            logger.log_and_raise(
                TransformationError(f"Failed to extract game result and reason details: {e}")
            )

        # Return the input DataFrame with only the original columns and the output columns
        base_columns = [column for column in input_columns if column not in self._OUTPUT_COLUMNS]
        return df.select(base_columns + list(self._OUTPUT_COLUMNS))
