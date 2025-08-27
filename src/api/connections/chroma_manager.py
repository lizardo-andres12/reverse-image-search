import chromadb
from config import DatabaseConfig


class ChromaConnectionManager:
    """Handles connection to ChromaDB vector database"""

    COLLECTION_NAME = "images"

    def __init__(self, config: DatabaseConfig):
        self.host = config.CHROMA_HOST
        self.port = config.CHROMA_PORT
        self.headers = {"CHROMA_TOKEN": config.CHROMA_SERVER_AUTH_TOKEN}

    def initialize_connection(self):
        print(self.host, self.port, self.headers)
        self.client = chromadb.HttpClient(
            host=self.host, port=self.port, headers=self.headers
        )

        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
