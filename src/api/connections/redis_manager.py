import redis
import redis.asyncio as aioredis
from config import DatabaseConfig


class RedisConnectionManager:
    """Handles connection to redis cache"""

    def __init__(self, config: DatabaseConfig):
        self.pool = redis.ConnectionPool.from_url(
            config.get_redis_url(),
            max_connections=50,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            decode_responses=True,
        )

    def initalize_connection(self):
        self.client = aioredis.Redis(connection_pool=self.pool)

    async def close_connection(self):
        await self.client.close()
