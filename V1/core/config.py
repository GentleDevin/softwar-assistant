"""
Improved configuration management.
"""
from dataclasses import dataclass, field
from typing import Optional
import os
import json
from enum import Enum
from pathlib import Path

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False


class Environment(Enum):
    """Running environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class LLMConfig:
    """LLM configuration."""
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen-plus"
    temperature: float = 0.5
    max_tokens: int = 2048
    timeout: int = 30


@dataclass
class Neo4jConfig:
    """Neo4j configuration."""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = ""
    database: str = "neo4j"
    max_connection_pool_size: int = 50
    connection_timeout: int = 30
    max_depth: int = 3


@dataclass
class RAGConfig:
    """RAG configuration."""
    chunk_size: int = 800
    chunk_overlap: int = 150
    similarity_threshold: float = 0.3
    top_k: int = 3
    embedding_model: str = "text-embedding-v4"
    embedding_dimension: int = 1024  # Fixed for text-embedding-v4
    cache_size: int = 1000
    cache_ttl: int = 3600


@dataclass
class EntityMatchingConfig:
    """Entity matching configuration."""
    similarity_threshold: float = 0.85
    use_cache: bool = True
    cache_path: str = "entity_embeddings.pkl"


@dataclass
class QASystemConfig:
    """Overall QA system configuration."""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    # Sub-configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    entity_matching: EntityMatchingConfig = field(default_factory=EntityMatchingConfig)
    
    # System configurations
    max_conversation_history: int = 10
    log_level: str = "INFO"
    log_file: str = "qa_system.log"
    enable_monitoring: bool = True
    enable_health_check: bool = True
    
    @classmethod
    def from_env_file(cls, env_file: str = ".env") -> 'QASystemConfig':
        """
        Load configuration from environment file.
        
        Args:
            env_file: Path to .env file
            
        Returns:
            QASystemConfig instance
        """
        if HAS_DOTENV:
            if Path(env_file).exists():
                load_dotenv(env_file, override=True)
            else:
                load_dotenv()  # Try system env
        
        # Read environment
        env_str = os.getenv('ENVIRONMENT', 'development').lower()
        environment = Environment(env_str)
        debug = os.getenv('DEBUG', 'false').lower() == 'true'
        
        # Build LLM config
        llm_config = LLMConfig(
            api_key=os.getenv('DASHSCOPE_API_KEY') or os.getenv('OPENAI_API_KEY') or '',
            base_url=os.getenv('OPENAI_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
            model=os.getenv('LLM_MODEL', 'qwen-plus'),
            temperature=float(os.getenv('LLM_TEMPERATURE', '0.5')),
            max_tokens=int(os.getenv('LLM_MAX_TOKENS', '2048')),
        )
        
        # Build Neo4j config
        neo4j_config = Neo4jConfig(
            uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            username=os.getenv('NEO4J_USERNAME', 'neo4j'),
            password=os.getenv('NEO4J_PASSWORD', ''),
            database=os.getenv('NEO4J_DATABASE', 'neo4j'),
            max_depth=int(os.getenv('NEO4J_MAX_DEPTH', '3')),
        )
        
        # Build RAG config
        rag_config = RAGConfig(
            chunk_size=int(os.getenv('RAG_CHUNK_SIZE', '800')),
            chunk_overlap=int(os.getenv('RAG_CHUNK_OVERLAP', '150')),
            similarity_threshold=float(os.getenv('RAG_SIMILARITY_THRESHOLD', '0.3')),
            top_k=int(os.getenv('RAG_TOP_K', '3')),
        )
        
        # Build entity matching config
        entity_matching_config = EntityMatchingConfig(
            similarity_threshold=float(os.getenv('ENTITY_MATCHING_THRESHOLD', '0.85')),
        )
        
        return cls(
            environment=environment,
            debug=debug,
            llm=llm_config,
            neo4j=neo4j_config,
            rag=rag_config,
            entity_matching=entity_matching_config,
            max_conversation_history=int(os.getenv('MAX_CONVERSATION_HISTORY', '10')),
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            log_file=os.getenv('LOG_FILE', 'qa_system.log'),
        )
    
    @classmethod
    def from_json_file(cls, json_file: str) -> 'QASystemConfig':
        """
        Load configuration from JSON file.
        
        Args:
            json_file: Path to JSON file
            
        Returns:
            QASystemConfig instance
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # Handle nested configs
        if 'llm' in config_dict:
            config_dict['llm'] = LLMConfig(**config_dict['llm'])
        if 'neo4j' in config_dict:
            config_dict['neo4j'] = Neo4jConfig(**config_dict['neo4j'])
        if 'rag' in config_dict:
            config_dict['rag'] = RAGConfig(**config_dict['rag'])
        if 'entity_matching' in config_dict:
            config_dict['entity_matching'] = EntityMatchingConfig(**config_dict['entity_matching'])
        
        if 'environment' in config_dict:
            config_dict['environment'] = Environment(config_dict['environment'])
        
        return cls(**config_dict)
    
    def to_dict(self, include_sensitive: bool = False) -> dict:
        """
        Convert to dictionary.
        
        Args:
            include_sensitive: Whether to include sensitive information
            
        Returns:
            Configuration as dictionary
        """
        config_dict = {
            'environment': self.environment.value,
            'debug': self.debug,
            'llm': {
                'model': self.llm.model,
                'temperature': self.llm.temperature,
                'base_url': self.llm.base_url,
            },
            'neo4j': {
                'uri': self.neo4j.uri,
                'database': self.neo4j.database,
            },
            'rag': {
                'chunk_size': self.rag.chunk_size,
                'top_k': self.rag.top_k,
                'similarity_threshold': self.rag.similarity_threshold,
            },
            'max_conversation_history': self.max_conversation_history,
            'log_level': self.log_level,
        }
        
        if include_sensitive:
            config_dict['llm']['api_key'] = self.llm.api_key
            config_dict['neo4j']['username'] = self.neo4j.username
            config_dict['neo4j']['password'] = self.neo4j.password
        
        return config_dict
    
    def validate(self) -> list:
        """
        Validate configuration.
        
        Returns:
            List of validation errors, empty list if valid
        """
        errors = []
        
        if not self.llm.api_key:
            errors.append("LLM API密钥未配置")
        
        if not self.neo4j.uri:
            errors.append("Neo4j URI未配置")
        
        if not 0.0 <= self.entity_matching.similarity_threshold <= 1.0:
            errors.append("实体匹配阈值必须在0-1之间")
        
        if not 0.0 <= self.rag.similarity_threshold <= 1.0:
            errors.append("RAG相似度阈值必须在0-1之间")
        
        if self.rag.chunk_size <= 0:
            errors.append("RAG块大小必须大于0")
        
        if self.rag.top_k <= 0:
            errors.append("RAG top_k必须大于0")
        
        if self.max_conversation_history <= 0:
            errors.append("对话历史长度必须大于0")
        
        return errors
    
    def save_json(self, json_file: str = "config.json", include_sensitive: bool = False) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            json_file: Output file path
            include_sensitive: Whether to include sensitive information
        """
        config_dict = self.to_dict(include_sensitive=include_sensitive)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
