#!/usr/bin/env bash
# Start Postgres, Redis, and MinIO and return once they answer. Idempotent.
# Does not drop or recreate an existing chess_teacher database.
set -euo pipefail

wait_for() {
  local name="$1"
  shift
  local i
  for i in $(seq 1 40); do
    if "$@"; then
      return 0
    fi
    sleep 0.5
  done
  echo "timed out waiting for ${name}" >&2
  return 1
}

if ! pg_lsclusters --no-header | awk '$1==17 && $2=="main" && $4=="online" {ok=1} END {exit !ok}'; then
  sudo pg_ctlcluster 17 main start
fi
wait_for postgres pg_isready -q -h 127.0.0.1 -p 5432

sudo -u postgres psql -v ON_ERROR_STOP=1 -d postgres << 'SQL'
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'chess_teacher') THEN
    CREATE ROLE chess_teacher LOGIN PASSWORD 'change-me';
  ELSE
    ALTER ROLE chess_teacher WITH LOGIN PASSWORD 'change-me';
  END IF;
END
$$;
SQL
if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname = 'chess_teacher'" | grep -q 1; then
  sudo -u postgres createdb -O chess_teacher chess_teacher
fi

if ! redis-cli -h 127.0.0.1 ping 2>/dev/null | grep -q PONG; then
  sudo install -d -o redis -g redis /var/lib/redis /var/run/redis
  sudo -u redis redis-server \
    --daemonize yes \
    --bind 127.0.0.1 \
    --port 6379 \
    --protected-mode yes \
    --dir /var/lib/redis \
    --dbfilename dump.rdb \
    --pidfile /var/run/redis/redis-server.pid
fi
wait_for redis redis-cli -h 127.0.0.1 ping

if ! curl -sf -o /dev/null http://127.0.0.1:9000/minio/health/live; then
  sudo install -d -o ubuntu -g ubuntu /var/lib/minio /var/log/chess-teacher
  sudo -u ubuntu env \
    MINIO_ROOT_USER=minioadmin \
    MINIO_ROOT_PASSWORD=minioadmin \
    nohup /usr/local/bin/minio server /var/lib/minio \
      --address 127.0.0.1:9000 \
      --console-address 127.0.0.1:9001 \
      > /var/log/chess-teacher/minio.log 2>&1 &
fi
wait_for minio curl -sf -o /dev/null http://127.0.0.1:9000/minio/health/live
/workspace/.venv/bin/python - << 'PY'
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

client = boto3.client(
    "s3",
    endpoint_url="http://127.0.0.1:9000",
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin",
    region_name="us-east-1",
    config=Config(signature_version="s3v4"),
)
try:
    client.create_bucket(Bucket="chess-teacher")
except ClientError as exc:
    code = exc.response.get("Error", {}).get("Code", "")
    if code not in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
        raise
PY

echo "postgres redis minio ready"
