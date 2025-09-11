from connections import PostgresConnectionManager
from mapper import image_metdata_db_to_model
from models import ImageMetadataModel, ImageTagModel


class ImageRepository: # TODO: implement caching
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

    TABLE image_tags {
        id serial
        image_uuid UUID fk -> images.uuid
        tag varchar(63) e.g. "Cat", "Dog"
        confidence float (model-generated tag confidence)
    }
    idx_image_tags_uuid on image_tags.uuid
    idx_image_tags_tag on image_tags.tag

    In the image_tags table, the only field of interest to the program is the `tag`,
    the rest are for relationship and enabling ordered retrieval

    Note the one-to-many relationship shared between images and their tags. Entries
    returned from this database will be of the following form:

    ImageMetadataModel {
        id: str
        filename: str
        source_url: str
        source_domain: str
        file_size: int
        dimensions: str
        tags: list[ImageTagModel]
    }
    """

    """Class-related constants"""
    INSERT_STMT = (
        "insert into `images` (uuid, filename, source_url, source_domain, "
        "file_size, dimensions) values ($1,$2,$3,$4,$5,$6)"
    )
    TAG_INSERT_STMT = "insert into `image_tags` (id, image_uuid, tag, confidence) values ($1,$2,$3,$4)"
    GET_JOIN_STMT = (
        "SELECT i.uuid, i.filename, i.source_url, i.source_domain, i.file_size, i.dimensions, "
        "STRING_AGG(it.tag, ',' ORDER BY it.confidence DESC) as tags FROM images i inner JOIN "
        "image_tags it ON i.uuid = it.image_uuid WHERE i.uuid = $1 GROUP BY i.uuid, i.filename, "
        "i.source_url, i.source_domain, i.file_size, i.dimensions ORDER BY i.indexed_at"
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
            async with conn.transaction():
                await conn.execute(
                    self.INSERT_STMT,
                    model.id,
                    model.filename,
                    model.source_url,
                    model.source_domain,
                    model.file_size,
                    model.dimensions,
                )

                await conn.executemany(self.TAG_INSERT_STMT, [tag.to_tuple() for tag in model.tags])
    
    async def batch_insert(self, models: list[ImageMetadataModel]):
        """Batch inserts model fields for every model into PostgreSQL database. This
        method will rollback all models on one failure, and should be wrapped in try/except
        block.
        
        Args:
            models (list[ImageMetadataModel]): The list of models to be inserted
        """
        models_list = [model.to_tuple() for model in models]
        tags_list = [[tag.to_tuple() for tag in model.tag] for model in models]
        async with self.conn.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(self.INSERT_STMT, models_list)
                for tags in tags_list:
                    await conn.executemany(self.TAG_INSERT_STMT, tags)

    async def get_image_metadata(self, id: str) -> ImageMetadataModel | None:
        async with self.conn.acquire() as conn:
            record = await conn.fetch(self.GET_JOIN_STMT, id)
            return image_metdata_db_to_model(record)
