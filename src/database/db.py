from pathlib import Path

from src.utils.env_utils import get_env_variable
from src.utils.pipeline_utils import Pipeline


class DatabaseLoader(Pipeline):
    # HAAL DEZE MISSCHIEN NAAR utils! en splits dan op in
    # logging pipeline + ingested->db pipeline child classes
    def __init__(self):
        super().__init__()
        try:
            raw_dir = get_env_variable("RAW_DIR")
        except ValueError as e:
            self.logger.error("RAW_DIR environment variable is not set.")
            raise e
        self.raw_dir = Path(raw_dir)

    def _run(self) -> None:
        NotImplementedError("IF YOU READ THIS, DO NOT REPLACE YET")
        self.logger.warning(
            "This pipeline is not implemented yet. It will be added in a future PR."
        )


# for file in self.raw_dir.glob("*.json"):
#     with open(file) as f:
#         data = json.load(f)

#     game_id = data["url"]

#     with self.engine.begin() as conn:
#         conn.execute(
#             text("""
#             INSERT INTO core.games (
#                 external_game_id,
#                 played_at,
#                 white_username,
#                 black_username,
#                 white_rating,
#                 black_rating,
#                 result,
#                 time_control,
#                 pgn,
#                 source_file
#             )
#             VALUES (
#                 :game_id,
#                 :played_at,
#                 :white_username,
#                 :black_username,
#                 :white_rating,
#                 :black_rating,
#                 :result,
#                 :time_control,
#                 :pgn,
#                 :source_file
#             )
#             ON CONFLICT (external_game_id) DO NOTHING
#         """),
#             {
#                 "game_id": game_id,
#                 "played_at": data.get("end_time"),
#                 "white_username": data["white"]["username"],
#                 "black_username": data["black"]["username"],
#                 "white_rating": data["white"]["rating"],
#                 "black_rating": data["black"]["rating"],
#                 "result": data["white"]["result"],
#                 "time_control": data.get("time_control"),
#                 "pgn": data.get("pgn"),
#                 "source_file": file.name,
#             },
#         )
