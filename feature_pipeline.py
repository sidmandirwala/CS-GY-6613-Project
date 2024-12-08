# feature_pipeline.py

import os
import uuid
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Type, TypeVar, Any, Union
from loguru import logger
from pymongo import MongoClient, errors
from pymongo.errors import ConnectionFailure
from pydantic import BaseModel, Field
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import chromedriver_autoinstaller
import openai

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

# -------------------------
# Custom Exceptions
# -------------------------

class LLMTwinException(Exception):
    """Base exception for LLMTwin."""
    pass

class ImproperlyConfigured(LLMTwinException):
    """Exception raised for improperly configured settings."""
    pass

# -------------------------
# Settings Class
# -------------------------

class Settings:
    # MongoDB Configuration
    DATABASE_HOST: str = "mongodb://127.0.0.1:27017"
    DATABASE_NAME: str = "twin"

    # Qdrant Configuration
    USE_QDRANT_CLOUD: bool = False
    QDRANT_DATABASE_HOST: str = "localhost"
    QDRANT_DATABASE_PORT: int = 6333
    QDRANT_CLOUD_URL: str = "str"  # Placeholder
    QDRANT_APIKEY: Optional[str] = None  # Set to your Qdrant API key if required

    # OpenAI Configuration
    OPENAI_MODEL_ID: str = "gpt-4o-mini"
    OPENAI_API_KEY: Optional[str] = None  # Set to your OpenAI API key if available

    # Data Configuration
    DATABASES: List[str] = ["github_scraper", "medium_scraper", "linkedin_scraper"]
    COLLECTIONS: List[str] = ["articles", "repositories", "profiles"]

    @property
    def OPENAI_MAX_TOKEN_WINDOW(self) -> int:
        official_max_token_window = {
            "gpt-3.5-turbo": 16385,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
        }.get(self.OPENAI_MODEL_ID, 128000)

        max_token_window = int(official_max_token_window * 0.90)
        return max_token_window

    def export(self) -> None:
        """
        Placeholder method for exporting settings.
        """
        # Since we are not using ZenML or .env, this can be left empty or implemented if needed.
        pass

    @classmethod
    def load_settings(cls) -> "Settings":
        """
        Loads settings. Since we are not using ZenML or .env, simply return an instance.
        """
        return cls()

# -------------------------
# MongoDatabaseConnector Class
# -------------------------

class MongoDatabaseConnector:
    _instance: Optional[MongoClient] = None

    def __new__(cls, settings: Settings) -> MongoClient:
        if cls._instance is None:
            try:
                cls._instance = MongoClient(settings.DATABASE_HOST)
                # The following line checks the connection
                cls._instance.admin.command('ping')
                logger.info(f"Connected to MongoDB at {settings.DATABASE_HOST}")
            except ConnectionFailure as e:
                logger.error(f"Couldn't connect to the database: {e!s}")
                raise
        return cls._instance

    def get_database(self, db_name: str):
        return self._instance[db_name]

# -------------------------
# Define NoSQLBaseDocument
# -------------------------

T = TypeVar("T", bound="NoSQLBaseDocument")

class NoSQLBaseDocument(BaseModel, ABC):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, self.__class__):
            return False
        return self.id == value.id

    def __hash__(self) -> int:
        return hash(self.id)

    @classmethod
    def from_mongo(cls: Type[T], data: dict) -> T:
        if not data:
            raise ValueError("Data is empty.")
        id_str = data.pop("_id")
        return cls(**dict(data, id=uuid.UUID(id_str)))

    def to_mongo(self: T, **kwargs) -> dict:
        exclude_unset = kwargs.pop("exclude_unset", False)
        by_alias = kwargs.pop("by_alias", True)
        parsed = self.dict(exclude_unset=exclude_unset, by_alias=by_alias, **kwargs)
        if "_id" not in parsed and "id" in parsed:
            parsed["_id"] = str(parsed.pop("id"))
        for key, value in parsed.items():
            if isinstance(value, uuid.UUID):
                parsed[key] = str(value)
        return parsed

    def save(self: T, database, **kwargs) -> Optional[T]:
        collection = database[self.get_collection_name()]
        try:
            collection.insert_one(self.to_mongo(**kwargs))
            logger.info(f"Document saved to {self.get_collection_name()} collection.")
            return self
        except errors.WriteError:
            logger.exception("Failed to insert document.")
            return None

    @classmethod
    def get_or_create(cls: Type[T], database, **filter_options) -> Optional[T]:
        collection = database[cls.get_collection_name()]
        try:
            instance = collection.find_one(filter_options)
            if instance:
                logger.info("Document found in the database.")
                return cls.from_mongo(instance)
            new_instance = cls(**filter_options)
            new_instance = new_instance.save(database, **filter_options)
            return new_instance
        except errors.OperationFailure:
            logger.exception(f"Failed to retrieve document with filter options: {filter_options}")
            raise

    @classmethod
    def bulk_insert(cls: Type[T], database, documents: List[T], **kwargs) -> bool:
        collection = database[cls.get_collection_name()]
        try:
            collection.insert_many(doc.to_mongo(**kwargs) for doc in documents)
            logger.info(f"Bulk insert successful for {cls.__name__}.")
            return True
        except (errors.WriteError, errors.BulkWriteError):
            logger.error(f"Failed to insert documents of type {cls.__name__}")
            return False

    @classmethod
    def find(cls: Type[T], database, **filter_options) -> Optional[T]:
        collection = database[cls.get_collection_name()]
        try:
            instance = collection.find_one(filter_options)
            if instance:
                return cls.from_mongo(instance)
            return None
        except errors.OperationFailure:
            logger.error("Failed to retrieve document")
            return None

    @classmethod
    def bulk_find(cls: Type[T], database, **filter_options) -> List[T]:
        collection = database[cls.get_collection_name()]
        try:
            instances = collection.find(filter_options)
            documents = [cls.from_mongo(instance) for instance in instances if instance]
            logger.info(f"Bulk find retrieved {len(documents)} documents.")
            return documents
        except errors.OperationFailure:
            logger.error("Failed to retrieve documents")
            return []

    @classmethod
    def get_collection_name(cls) -> str:
        if not hasattr(cls, "Settings") or not hasattr(cls.Settings, "name"):
            raise ImproperlyConfigured(
                "Document should define a Settings configuration class with the name of the collection."
            )
        return cls.Settings.name

    @classmethod
    def get_collection(cls, database) -> Any:
        return database[cls.get_collection_name()]

# -------------------------
# Define VectorBaseDocument
# -------------------------

class VectorBaseDocument(NoSQLBaseDocument, ABC):
    """
    Abstract base class for documents stored in the Vector DB.
    """
    embedding: List[float] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def bulk_insert(cls: Type[T], client: QdrantClient, collection_name: str, documents: List[T]) -> bool:
        """
        Inserts multiple documents into the Vector DB.

        Args:
            client (QdrantClient): The Qdrant client instance.
            collection_name (str): The name of the collection in Qdrant.
            documents (List[T]): A list of document instances to insert.

        Returns:
            bool: True if insertion is successful, False otherwise.
        """
        try:
            logger.info(f"Inserting {len(documents)} documents into Vector DB collection '{collection_name}'.")
            points = [
                PointStruct(id=str(doc.id), vector=doc.embedding, payload=doc.metadata)
                for doc in documents
            ]
            client.upsert(collection_name=collection_name, points=points)
            logger.info("Bulk insert into Vector DB successful.")
            return True
        except Exception as e:
            logger.error(f"Failed to insert documents into Vector DB: {e}")
            return False

# -------------------------
# Define EmbeddedDocument Classes
# -------------------------

class EmbeddedArticleChunk(NoSQLBaseDocument):
    text: str
    source: str

    class Settings:
        name = "embedded_articles"

class EmbeddedPostChunk(NoSQLBaseDocument):
    text: str
    source: str

    class Settings:
        name = "embedded_posts"

class EmbeddedRepositoryChunk(NoSQLBaseDocument):
    text: str
    source: str

    class Settings:
        name = "embedded_repositories"

class EmbeddedProfileChunk(NoSQLBaseDocument):
    text: str
    source: str

    class Settings:
        name = "embedded_profiles"

class EmbeddedQuery(NoSQLBaseDocument):
    query_text: str
    embedding: List[float]

    class Settings:
        name = "embedded_queries"

class DataCategory:
    ARTICLES = "articles"
    POSTS = "posts"
    REPOSITORIES = "repositories"
    PROFILES = "profiles"
    QUERIES = "queries"

# -------------------------
# Define Document and RepositoryDocument
# -------------------------

class Document(NoSQLBaseDocument, ABC):
    content: Dict[str, Any]
    platform: str
    author_id: uuid.UUID = Field(alias="author_id")
    author_full_name: str = Field(alias="author_full_name")

class RepositoryDocument(Document):
    name: str
    link: str

    class Settings:
        name = "repositories"

class ProfileDocument(Document):
    profile_name: str
    link: str

    class Settings:
        name = "profiles"

# -------------------------
# Define Cleaning, Chunking, and Embedding Handlers
# -------------------------

class CleaningHandler:
    def clean_content(self, content: Union[str, dict]) -> str:
        """
        Cleans the input content.

        Args:
            content (Union[str, dict]): Raw content as a string or dictionary.

        Returns:
            str: Cleaned content.
        """
        if isinstance(content, dict):
            # Combine all string values into one string
            text = ' '.join([str(v) for v in content.values() if isinstance(v, str)])
        elif isinstance(content, str):
            text = content
        else:
            text = ''
        cleaned = text.strip()
        return cleaned if cleaned else ""

class ChunkingHandler:
    def chunk(self, content: str, chunk_size: int = 500) -> List[str]:
        """
        Splits the content into chunks of specified size.

        Args:
            content (str): Cleaned content.
            chunk_size (int): Maximum size of each chunk.

        Returns:
            List[str]: List of content chunks.
        """
        return [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

class EmbeddingHandler:
    def embed_chunks(self, chunks: List[str]) -> List[Dict]:
        """
        Generates embeddings for each chunk.

        Args:
            chunks (List[str]): List of content chunks.

        Returns:
            List[Dict]: List of embeddings with metadata.
        """
        # Placeholder for actual embedding logic
        # Replace with actual embedding generation using a model or API
        embeddings = [{"id": str(uuid.uuid4()), "embedding": [0.0]*768, "metadata": {"text": chunk}} for chunk in chunks]
        return embeddings

    def embed_qas(self, qas: List[Dict[str, str]]) -> List[Dict]:
        """
        Generates embeddings for each Q&A pair.

        Args:
            qas (List[Dict[str, str]]): List of Q&A pairs.

        Returns:
            List[Dict]: List of embeddings with metadata.
        """
        # Placeholder for actual embedding logic
        # Replace with actual embedding generation using a model or API
        embeddings = [{"id": str(uuid.uuid4()), "embedding": [0.0]*768, "metadata": qa} for qa in qas]
        return embeddings

def convert_numpy_to_list(embeddings: List[Dict]) -> List[Dict]:
    """
    Converts NumPy arrays to Python lists in embeddings.

    Args:
        embeddings (List[Dict]): List of embeddings.

    Returns:
        List[Dict]: Converted embeddings.
    """
    # Placeholder if embeddings contain NumPy arrays
    return embeddings

# -------------------------
# Q&A Handler to generate question-answer pairs using OpenAI
# -------------------------

class QandAHandler:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def generate_qas(self, content: str, max_qas: int = 5) -> List[Dict[str, str]]:
        """
        Generate question-answer pairs from the given content using OpenAI's API.

        Args:
            content (str): The content to generate Q&A pairs from.
            max_qas (int): Maximum number of Q&A pairs to generate.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing 'question' and 'answer'.
        """
        prompt = (
            "Extract key information from the following text and present it as "
            f"up to {max_qas} question and answer pairs.\n\nContent:\n{content}\n\nQ&A Pairs:"
        )

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts key information into Q&A pairs."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                n=1,
                stop=None,
                temperature=0.7,
            )

            qas_text = response.choices[0].message['content'].strip()
            qas = self.parse_qas(qas_text)
            return qas

        except Exception as e:
            logger.error(f"Failed to generate Q&A pairs: {e}")
            return []

    @staticmethod
    def parse_qas(qas_text: str) -> List[Dict[str, str]]:
        """
        Parses the Q&A text output from OpenAI into a list of dictionaries.

        Args:
            qas_text (str): The raw Q&A text.

        Returns:
            List[Dict[str, str]]: Parsed Q&A pairs.
        """
        qas = []
        lines = qas_text.split('\n')
        current_q = None
        current_a = None

        for line in lines:
            if line.lower().startswith("question") or line.lower().startswith("q:"):
                if current_q and current_a:
                    qas.append({"question": current_q, "answer": current_a})
                    current_q = None
                    current_a = None
                parts = line.split(":", 1)
                if len(parts) > 1:
                    current_q = parts[1].strip()
            elif line.lower().startswith("answer") or line.lower().startswith("a:"):
                parts = line.split(":", 1)
                if len(parts) > 1:
                    current_a = parts[1].strip()

        if current_q and current_a:
            qas.append({"question": current_q, "answer": current_a})

        return qas

# -------------------------
# Define BaseCrawler and GithubCrawler
# -------------------------

class BaseCrawler(ABC):
    model: Type[NoSQLBaseDocument]

    @abstractmethod
    def extract(self, link: str, user: Dict) -> None:
        pass

class GithubCrawler(BaseCrawler):
    model = RepositoryDocument

    def __init__(self, settings: Settings, ignore=(".git", ".toml", ".lock", ".png")) -> None:
        super().__init__()
        self.settings = settings
        self._ignore = ignore
        self.driver = self.initialize_driver()

    def initialize_driver(self) -> webdriver.Chrome:
        chromedriver_autoinstaller.install()
        options = Options()
        options.add_argument("--no-sandbox")
        options.add_argument("--headless=new")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--log-level=3")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-background-networking")
        options.add_argument("--ignore-certificate-errors")
        options.add_argument(f"--user-data-dir={tempfile.mkdtemp()}")
        options.add_argument(f"--data-path={tempfile.mkdtemp()}")
        options.add_argument(f"--disk-cache-dir={tempfile.mkdtemp()}")
        options.add_argument("--remote-debugging-port=9226")

        self.set_extra_driver_options(options)

        driver = webdriver.Chrome(options=options)
        return driver

    def set_extra_driver_options(self, options: Options) -> None:
        """Override this method to set additional driver options."""
        pass

    def login(self) -> None:
        """Override this method to implement login functionality if required."""
        pass

    def extract(self, link: str, user: Dict) -> None:
        db = self.settings.DATABASE_NAME
        database = MongoDatabaseConnector(settings=self.settings).get_database(db)
        old_model = self.model.find(database=database, link=link)
        if old_model is not None:
            logger.info(f"Repository already exists in the database: {link}")
            return

        logger.info(f"Starting scraping GitHub repository: {link}")
        repo_name = link.rstrip("/").split("/")[-1]
        local_temp = tempfile.mkdtemp()

        try:
            os.chdir(local_temp)
            subprocess.run(["git", "clone", link], check=True)
            repo_path = os.path.join(local_temp, os.listdir(local_temp)[0])

            tree = {}
            for root, _, files in os.walk(repo_path):
                dir = root.replace(repo_path, "").lstrip("/")
                if any(dir.startswith(ignore) for ignore in self._ignore):
                    continue

                for file in files:
                    if any(file.endswith(ext) for ext in self._ignore):
                        continue
                    file_path = os.path.join(dir, file)
                    with open(os.path.join(root, file), "r", errors="ignore") as f:
                        tree[file_path] = f.read().replace(" ", "")

            instance = self.model(
                content=tree,
                name=repo_name,
                link=link,
                platform="github",
                author_id=user.get("id", uuid.uuid4()),
                author_full_name=user.get("full_name", "Unknown"),
            )
            instance.save(database=database)

        except subprocess.CalledProcessError as e:
            logger.exception(f"Failed to clone repository {link}: {e}")
            raise
        except Exception as e:
            logger.exception(f"Failed to scrape GitHub repository {link}: {e}")
            raise
        finally:
            shutil.rmtree(local_temp)

        logger.info(f"Finished scraping GitHub repository: {link}")

# -------------------------
# Define FeaturePipeline Class
# -------------------------

class FeaturePipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        # Initialize MongoDB connection
        try:
            self.mongo_connector = MongoDatabaseConnector(settings=settings)
            self.mongo_db_names = settings.DATABASES
            logger.info("Connected to MongoDB.")
        except errors.ConnectionFailure as e:
            logger.error(f"Could not connect to MongoDB: {e}")
            raise

        # Initialize cleaning, chunking, and embedding handlers
        self.cleaning_handler = CleaningHandler()
        self.chunking_handler = ChunkingHandler()
        self.embedding_handler = EmbeddingHandler()

        # Initialize Q&A handler if API key is provided
        if settings.OPENAI_API_KEY:
            self.qanda_handler = QandAHandler(api_key=settings.OPENAI_API_KEY)
            logger.info("QandAHandler initialized.")
        else:
            self.qanda_handler = None
            logger.warning("OpenAI API key not provided. Q&A generation will be skipped.")

        # Initialize Vector DB connection (assuming Qdrant)
        try:
            self.vector_client = QdrantClient(
                host=settings.QDRANT_DATABASE_HOST,
                port=settings.QDRANT_DATABASE_PORT,
                api_key=settings.QDRANT_APIKEY
            )
            logger.info("Connected to Qdrant Vector DB.")
        except Exception as e:
            logger.error(f"Could not connect to Qdrant Vector DB: {e}")
            raise

        # Initialize Crawler (Optional)
        self.github_crawler = GithubCrawler(settings=settings)

    def process_document(self, db_name: str, collection_name: str, doc: Dict):
        """
        Process a single document: clean, chunk, generate Q&A, embed, and store.

        Args:
            db_name (str): Name of the MongoDB database.
            collection_name (str): Name of the collection within the database.
            doc (Dict): The document to process.
        """
        repo_id = str(doc.get("_id"))
        db = self.mongo_connector.get_database(db_name)
        collection = db[collection_name]

        try:
            # Step 1: Clean the data
            content = doc.get("content", {})
            cleaned_data = self.cleaning_handler.clean_content(content)
            if not cleaned_data:
                logger.warning(f"Document {repo_id} has no content after cleaning. Skipping.")
                return

            # Step 2: Chunk the content
            chunks = self.chunking_handler.chunk(cleaned_data)
            if not chunks:
                logger.warning(f"Document {repo_id} has no chunks after chunking. Skipping.")
                return

            # Step 3: Generate Q&A pairs if handler is available
            qas = []
            if self.qanda_handler:
                qas = self.qanda_handler.generate_qas(cleaned_data)
                if not qas:
                    logger.warning(f"Document {repo_id}: No Q&A pairs generated.")

            # Step 4: Embed the chunks and Q&A pairs
            embedded_chunks = self.embedding_handler.embed_chunks(chunks)
            embedded_qas = self.embedding_handler.embed_qas(qas) if qas else []

            # Step 5: Insert embedded data into Vector DB
            total_embeddings = embedded_chunks + embedded_qas
            if total_embeddings:
                # Determine the VectorBaseDocument subclass based on DataCategory
                data_category = doc.get("category", "general").lower()
                VectorBaseDocumentClass = self.get_vector_class(data_category)
                if VectorBaseDocumentClass is None:
                    logger.error(f"Unsupported data category: {data_category} for document {repo_id}. Skipping.")
                    return

                success = VectorBaseDocumentClass.bulk_insert(
                    client=self.vector_client,
                    collection_name=VectorBaseDocumentClass.Settings.name,
                    documents=total_embeddings
                )
                if success:
                    logger.info(f"Successfully inserted {len(total_embeddings)} embeddings into Vector DB for document {repo_id}.")
                else:
                    logger.error(f"Failed to insert some embeddings into Vector DB for document {repo_id}.")
            else:
                logger.warning(f"Document {repo_id} has no embeddings to insert into Vector DB.")

            # Step 6: Update MongoDB with processed data
            update_data = {
                "cleaned_content": cleaned_data,
                "embedded_chunks": convert_numpy_to_list(embedded_chunks),
            }
            if qas:
                update_data["embedded_qas"] = convert_numpy_to_list(embedded_qas)

            collection.update_one({"_id": doc["_id"]}, {"$set": update_data})
            logger.info(f"Document {repo_id} processed and updated successfully.")

        except Exception as e:
            logger.exception(f"An error occurred while processing document {repo_id} in {db_name}.{collection_name}: {e}")

    def get_vector_class(self, data_category: str) -> Optional[Type[VectorBaseDocument]]:
        """
        Determine the VectorBaseDocument subclass based on the data category.

        Args:
            data_category (str): The category of the data.

        Returns:
            Optional[Type[VectorBaseDocument]]: The corresponding VectorBaseDocument subclass or None if unsupported.
        """
        if data_category == DataCategory.ARTICLES:
            return EmbeddedArticleChunk
        elif data_category == DataCategory.POSTS:
            return EmbeddedPostChunk
        elif data_category == DataCategory.REPOSITORIES:
            return EmbeddedRepositoryChunk
        elif data_category == DataCategory.PROFILES:
            return EmbeddedProfileChunk
        elif data_category == DataCategory.QUERIES:
            return EmbeddedQuery
        else:
            return None

    def process_all_documents(self):
        """
        Process all documents across specified databases and collections.
        """
        for db_name in self.mongo_db_names:
            try:
                db = self.mongo_connector.get_database(db_name)
                collection_names = db.list_collection_names()
                logger.info(f"Processing database: {db_name} with collections: {collection_names}")

                for collection_name in collection_names:
                    if collection_name not in self.settings.COLLECTIONS:
                        logger.warning(f"Collection {collection_name} is not in the specified COLLECTIONS list. Skipping.")
                        continue

                    collection = db[collection_name]
                    total_docs = collection.count_documents({})
                    logger.info(f"Processing collection: {collection_name} with {total_docs} documents.")

                    cursor = collection.find()
                    for doc in cursor:
                        self.process_document(db_name, collection_name, doc)

            except errors.PyMongoError as e:
                logger.error(f"Failed to process database {db_name}: {e}")

    def run(self):
        """Run the feature pipeline."""
        logger.info("Starting the Feature Pipeline...")
        self.process_all_documents()
        logger.info("Feature Pipeline completed.")

# -------------------------
# Execution Entry Point
# -------------------------

if __name__ == "__main__":
    # Initialize settings
    settings = Settings.load_settings()

    # Optionally, set the OpenAI API key here if available
    # settings.OPENAI_API_KEY = "your_openai_api_key_here"

    # Optionally, set the Qdrant API key here if required
    # settings.QDRANT_APIKEY = "your_qdrant_api_key_here"

    # Initialize and run the feature pipeline
    try:
        pipeline = FeaturePipeline(settings=settings)
        pipeline.run()
    except Exception as e:
        logger.exception(f"Feature Pipeline failed to run: {e}")
