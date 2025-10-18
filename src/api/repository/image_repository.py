from asyncpg import NoDataFoundError
from managers import PostgresConnectionManager
from mapper import image_metadata_db_to_model
from models import ImageMetadataModel


class ImageRepository:  # TODO: implement caching
    """
    Encapsulates PostgreSQL data access logic

    The PostgreSQL database has the following table schemas:

    TABLE images {
        uuid UUID
        filename VARCHAR(255)
        source_url text
        source_domain text
        file_size integer
        dimensions varchar(20) e.g. "1920x1080"
        created_at timestamp
        indexed_at timestamp
    }

    Entries returned from this database will be of the following form:

    ImageMetadataModel {
        id: str
        filename: str
        source_url: str
        source_domain: str
        file_size: int
        dimensions: str
    }
    """

    """Class-related constants"""
    INSERT_STMT = (
        "INSERT INTO images (uuid, filename, source_url, source_domain, "
        "file_size, dimensions) VALUES ($1,$2,$3,$4,$5,$6)"
    )
    GET_JOIN_STMT = (
        "SELECT uuid, filename, source_url, source_domain, file_size, dimensions FROM "
        "images WHERE uuid = $1 ORDER BY indexed_at"
    )

    def __init__(self, conn: PostgresConnectionManager):
        self.conn = conn.client

    async def insert(self, model: ImageMetadataModel):
        """Inserts model fields into PostgreSQL database. This method should be wrapped
        in a try/except block.

        Args:
            model (ImageMetadataModel): The model object containing row data to be inserted.
        """
        async with self.conn.acquire() as conn:
            await conn.execute(
                self.INSERT_STMT,
                model.id,
                model.filename,
                str(model.source_url),
                str(model.source_domain),
                model.file_size,
                model.dimensions,
            )

    async def batch_insert(self, models: list[ImageMetadataModel]):
        """Batch inserts model fields for every model into PostgreSQL database. This
        method will rollback all models on one failure, and should be wrapped in try/except
        block.

        Args:
            models (list[ImageMetadataModel]): The list of models to be inserted
        """
        models_list = [model.to_tuple() for model in models]
        async with self.conn.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(self.INSERT_STMT, models_list)

    async def get_image_metadata(self, id: str) -> ImageMetadataModel:
        """Queries the Postgres database for metadata associated with the given id and all tags
        associated with the id.

        Args:
            id (str): The uuid to check against in the database.
        Returns:
            ImageMetadataModel: The model containing returned data.
        Throws:
            NoDataFoundError: If `None` is returned from the query.
        """
        async with self.conn.acquire() as conn:
            record = await conn.fetchrow(self.GET_JOIN_STMT, id)
            if record is None:
                raise NoDataFoundError(f"No data with id {id}")
            return image_metadata_db_to_model(record)

    async def batch_get_image_metadata(
        self, ids: list[str]
    ) -> list[ImageMetadataModel]:
        async with self.conn.acquire() as conn:
            records = await conn.fetchmany(self.GET_JOIN_STMT, [[id] for id in ids])
            return [image_metadata_db_to_model(record) for record in records]
