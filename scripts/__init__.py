"""Scripts package for chess_teacher.

Layout:
  entrypoints/  Prod container commands (CronJob, Job, Streamlit subprocess)
  ops/          Manual prod mutations (typically via run-script-job)
  tools/        Laptop / agent CLIs
  utils/        Shared helpers and the Job launcher
  dev/          Local bootstrap / sync helpers
"""
