from pathlib import Path

from src.utils.pipeline_utils import Pipeline


class DatabaseLoader(Pipeline):
    def __init__(self):
        super().__init__()
        self.raw_dir = Path("/app/storage/raw/input")  # TODO: make configurabel based on .env

    def _run(self) -> None:
        NotImplementedError("IF YOU READ THIS, DO NOT REPLACE YET")


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
