import chromadb
from config import DatabaseConfig


class ChromaConnectionManager:
    """Handles connection to ChromaDB vector database"""

    COLLECTION_NAME = "images"

    def __init__(self, config: DatabaseConfig):
        """
        Initializes the connection parameters need to connect to ChromaDB image embeddings database.

        Args:
            config (DatabaseConfig): The connection configuration loaded through `docker/.env`.
        """
        self.host = config.CHROMA_HOST
        self.port = config.CHROMA_PORT
        self.headers = {"CHROMA_TOKEN": config.CHROMA_SERVER_AUTH_TOKEN}

    def initialize_connection(self) -> None:
        """
        Creates an HTTP connection to the ChromaDB docker container running from docker-compose and retrieves the
        corresponding image embedding collection or creates it.
        """
        self.client = chromadb.HttpClient(
            host=self.host, port=self.port, headers=self.headers
        )

        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def healthcheck(self) -> dict:
        try:
            is_alive = self.client.heartbeat()
            return {"ChromaDB": "Healthy!"}
        except Exception as e:
            return {"ChromaDB": f"Error: {e}"}
