import asyncpg
from config import DatabaseConfig


class PostgresConnectionManager:
    """Manages connection to Postgres database"""

    def __init__(self, config: DatabaseConfig):
        self.dsn = config.get_postgres_url()
        self.min_conns = config.POSTGRES_MIN_CONNECTIONS
        self.max_conns = config.POSTGRES_MAX_CONNECTIONS

    async def initialize_connection(self):
        self.client = await asyncpg.create_pool(
            dsn=self.dsn, min_size=self.min_conns, max_size=self.max_conns
        )

    async def healthcheck(self) -> dict:
        try:
            async with self.client.acquire() as conn:
                await conn.fetchval("select 1")
            return {"Postgres": "Healthy!"}
        except asyncpg.exceptions.PostgresError as e:
            return {"Postgres": f"Error: {e}"}

    async def close_connection(self):
        await self.client.close()
