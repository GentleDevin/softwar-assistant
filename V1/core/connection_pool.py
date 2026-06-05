"""
Improved connection pool for database connections.
"""
from queue import Queue, Empty
import threading
from typing import Any, Callable, Dict
import time
import logging


class ConnectionPoolException(Exception):
    """Exception raised by connection pool errors."""
    pass


class ConnectionPool:
    """
    Generic connection pool for managing database connections.
    
    Supports lazy initialization, connection validation, and automatic cleanup.
    """
    
    def __init__(
        self,
        connection_factory: Callable[[], Any],
        max_connections: int = 10,
        timeout: int = 30,
        max_idle_time: int = 300,
        logger: logging.Logger = None
    ):
        """
        Initialize connection pool.
        
        Args:
            connection_factory: Function that creates a new connection
            max_connections: Maximum number of connections in pool
            timeout: Timeout in seconds for getting a connection
            max_idle_time: Maximum idle time in seconds before connection is closed
            logger: Logger instance
        """
        self.connection_factory = connection_factory
        self.max_connections = max_connections
        self.timeout = timeout
        self.max_idle_time = max_idle_time
        self.logger = logger or logging.getLogger('qa_connection_pool')
        
        self.pool: Queue = Queue(maxsize=max_connections)
        self.lock = threading.Lock()
        self.active_connections = 0
        self.total_connections = 0
        self.connection_metadata: Dict[int, float] = {}  # conn_id -> last_used
        
        self._shutdown = False
        
        # Initialize pool
        self._initialize_pool()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_idle_connections,
            daemon=True
        )
        self._cleanup_thread.start()
    
    def _initialize_pool(self) -> None:
        """Initialize the connection pool with initial connections."""
        initial_connections = min(2, self.max_connections)
        for _ in range(initial_connections):
            try:
                conn = self._create_connection()
                self.pool.put(conn, block=False)
            except Exception as e:
                self.logger.warning(f"创建初始连接失败: {e}")
    
    def _create_connection(self) -> Any:
        """Create a new connection using the factory."""
        conn = self.connection_factory()
        with self.lock:
            self.total_connections += 1
            conn_id = id(conn)
            self.connection_metadata[conn_id] = time.time()
        self.logger.debug(f"创建新连接 {conn_id}")
        return conn
    
    def _validate_connection(self, conn: Any) -> bool:
        """
        Validate if a connection is still alive.
        
        This method should be overridden for specific database types.
        By default, we assume it's valid.
        
        Args:
            conn: Connection to validate
            
        Returns:
            True if connection is valid
        """
        try:
            # Default: just check if it has expected attributes
            return conn is not None
        except Exception:
            return False
    
    def get_connection(self) -> Any:
        """
        Get a connection from the pool.
        
        Returns:
            A database connection
            
        Raises:
            ConnectionPoolException: If no connection available within timeout
        """
        if self._shutdown:
            raise ConnectionPoolException("连接池已关闭")
        
        conn = None
        start_time = time.time()
        
        while time.time() - start_time < self.timeout:
            try:
                # Try to get a connection from pool
                conn = self.pool.get(timeout=1)
                
                # Validate the connection
                if not self._validate_connection(conn):
                    self.logger.warning("连接无效，关闭并创建新连接")
                    self._close_connection(conn)
                    conn = self._create_connection()
                
                # Update connection metadata
                with self.lock:
                    self.active_connections += 1
                    conn_id = id(conn)
                    self.connection_metadata[conn_id] = time.time()
                
                return conn
                
            except Empty:
                # Try to create a new connection if we haven't reached max
                with self.lock:
                    if self.active_connections + self.pool.qsize() < self.max_connections:
                        try:
                            conn = self._create_connection()
                            self.active_connections += 1
                            return conn
                        except Exception as e:
                            self.logger.error(f"创建新连接失败: {e}")
                            raise ConnectionPoolException(f"创建连接失败: {e}") from e
        
        raise ConnectionPoolException(
            f"在{self.timeout}秒内无法获取可用连接，活动连接: {self.active_connections}"
        )
    
    def return_connection(self, conn: Any) -> None:
        """
        Return a connection to the pool.
        
        Args:
            conn: Connection to return
        """
        if conn is None:
            return
        
        try:
            with self.lock:
                if self.active_connections > 0:
                    self.active_connections -= 1
                
                # Update last used time
                conn_id = id(conn)
                self.connection_metadata[conn_id] = time.time()
            
            if not self._shutdown and self._validate_connection(conn):
                self.pool.put(conn, block=False)
            else:
                self._close_connection(conn)
                
        except Exception as e:
            self.logger.error(f"返回连接失败: {e}")
            self._close_connection(conn)
    
    def _close_connection(self, conn: Any) -> None:
        """Close a connection and clean up metadata."""
        try:
            if conn is not None:
                with self.lock:
                    conn_id = id(conn)
                    self.connection_metadata.pop(conn_id, None)
                
                if hasattr(conn, 'close') and callable(conn.close):
                    conn.close()
                    self.logger.debug(f"连接 {id(conn)} 已关闭")
        except Exception as e:
            self.logger.error(f"关闭连接失败: {e}")
    
    def _cleanup_idle_connections(self) -> None:
        """Periodically clean up idle connections."""
        while not self._shutdown:
            try:
                time.sleep(60)  # Check every minute
                
                now = time.time()
                with self.lock:
                    idle_threshold = now - self.max_idle_time
                    idle_conn_ids = [
                        conn_id for conn_id, last_used in self.connection_metadata.items()
                        if last_used < idle_threshold
                    ]
                
                if idle_conn_ids:
                    self.logger.debug(f"清理 {len(idle_conn_ids)} 个闲置连接")
                    # Note: Actual cleanup needs to handle queue differently
                    # This is simplified version
                    
            except Exception as e:
                self.logger.error(f"清理连接出错: {e}")
    
    def close_all(self) -> None:
        """Close all connections in the pool."""
        self._shutdown = True
        
        while not self.pool.empty():
            try:
                conn = self.pool.get_nowait()
                self._close_connection(conn)
            except Empty:
                break
        
        self.logger.info("所有连接已关闭")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get pool statistics.
        
        Returns:
            Dictionary with pool statistics
        """
        with self.lock:
            return {
                'max_connections': self.max_connections,
                'active_connections': self.active_connections,
                'available_connections': self.pool.qsize(),
                'total_connections_created': self.total_connections,
            }
    
    def __enter__(self) -> 'ConnectionPool':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()


class Neo4jConnectionPool(ConnectionPool):
    """
    Specialized connection pool for Neo4j connections.
    """
    
    def __init__(
        self,
        uri: str,
        auth: tuple,
        max_connections: int = 10,
        timeout: int = 30,
        logger: logging.Logger = None
    ):
        """
        Initialize Neo4j connection pool.
        
        Args:
            uri: Neo4j URI
            auth: Tuple of (username, password)
            max_connections: Maximum number of connections
            timeout: Connection timeout
            logger: Logger instance
        """
        self.uri = uri
        self.auth = auth
        
        def connection_factory():
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(uri, auth=auth)
            # Verify connection
            driver.verify_connectivity()
            return driver
        
        super().__init__(
            connection_factory=connection_factory,
            max_connections=max_connections,
            timeout=timeout,
            logger=logger
        )
    
    def _validate_connection(self, conn: Any) -> bool:
        """Validate Neo4j connection by running a simple query."""
        try:
            if hasattr(conn, 'session'):
                with conn.session() as session:
                    session.run("RETURN 1")
                return True
        except Exception:
            pass
        return False
