"""User-facing pipeline orchestration.

Keep this package init empty: importing a submodule (e.g. ``preprocessing.games``)
must not pull ``runner`` → neural_network and create an import cycle.
Import ``run_pipeline`` from ``chess_teacher.pipelines.runner`` directly.
"""
