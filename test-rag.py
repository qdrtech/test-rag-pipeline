import os

# Redis value running in docker container
## docker run -d --name redis-stack -p 6379:6379 -e REDIS_ARGS="--requirepass p1234" redis/redis-stack-server:latest
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "p1234")

REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"

