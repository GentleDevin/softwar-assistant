# -*- coding: utf-8 -*-
"""
日志工具模块
提供统一的日志配置和管理，支持彩色控制台输出、按天和按大小轮转
"""

import logging
import os
import time
from typing import Optional
from datetime import datetime
import glob


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""
    
    # 颜色代码
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    
    # 前景色
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # 背景色
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"
    
    # 日志级别颜色映射
    LEVEL_COLORS = {
        logging.DEBUG: CYAN,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD + RED
    }
    
    def __init__(self, fmt: str = None, datefmt: str = None, use_color: bool = True):
        super().__init__(fmt, datefmt)
        self.use_color = use_color
    
    def format(self, record: logging.LogRecord) -> str:
        # 如果需要颜色，添加颜色代码
        if self.use_color:
            # 日志级别颜色
            level_color = self.LEVEL_COLORS.get(record.levelno, self.WHITE)
            
            # 文件名和方法名颜色
            file_method_color = self.CYAN
            
            # 构建带颜色的格式
            color_fmt = (
                f"%(asctime)s - "
                f"{level_color}%(levelname)s{self.RESET} - "
                f"%(threadName)s - "
                f"{file_method_color}%(filename)s{self.RESET} - "
                f"{file_method_color}%(funcName)s{self.RESET} - "
                f"%(message)s"
            )
        else:
            # 不带颜色的格式
            color_fmt = "%(asctime)s - %(levelname)s - %(threadName)s - %(filename)s - %(funcName)s - %(message)s"
        
        # 更新格式化器
        self._style._fmt = color_fmt
        
        return super().format(record)


class DailyTimedRotatingFileHandler(logging.FileHandler):
    """
    按天和大小轮转的日志处理器
    格式: app-YYYY-MM-DD.log, app-YYYY-MM-DD-01.log, app-YYYY-MM-DD-02.log
    """
    
    def __init__(self, base_filename: str, log_dir: str = "logs", 
                 max_bytes: int = 10 * 1024 * 1024, encoding: str = "utf-8"):
        """
        初始化日志处理器
        
        Args:
            base_filename: 基础文件名，如 "app"
            log_dir: 日志目录
            max_bytes: 单个日志文件最大大小
            encoding: 文件编码
        """
        self.base_filename = base_filename
        self.log_dir = log_dir
        self.max_bytes = max_bytes
        self.encoding = encoding
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.current_suffix = None
        self.current_file = self._get_initial_filename()
        
        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 调用父类初始化
        super().__init__(self.current_file, mode='a', encoding=encoding, delay=False)
    
    def _get_initial_filename(self) -> str:
        """获取初始日志文件名（不带序号的版本）"""
        # 检查不带序号的文件是否存在且没有超过大小
        base_file = os.path.join(self.log_dir, f"{self.base_filename}-{self.current_date}.log")
        
        if os.path.exists(base_file):
            # 检查文件大小
            if os.path.getsize(base_file) < self.max_bytes:
                return base_file
            else:
                # 文件已超过大小，需要找下一个序号
                self.current_suffix = self._get_next_suffix()
                return os.path.join(self.log_dir, f"{self.base_filename}-{self.current_date}-{self.current_suffix}.log")
        
        # 文件不存在，使用不带序号的版本
        return base_file
    
    def _get_next_suffix(self) -> str:
        """获取当前日期的下一个后缀"""
        # 查找当前日期的所有日志文件（包括不带序号的和带序号的）
        base_pattern = os.path.join(self.log_dir, f"{self.base_filename}-{self.current_date}.log")
        numbered_pattern = os.path.join(self.log_dir, f"{self.base_filename}-{self.current_date}-*.log")
        
        existing_files = []
        if os.path.exists(base_pattern):
            existing_files.append(base_pattern)
        existing_files.extend(glob.glob(numbered_pattern))
        
        if not existing_files:
            return "01"
        
        # 提取现有最大序号
        max_suffix = 0
        for file_path in existing_files:
            try:
                # 从文件名中提取序号
                base = os.path.basename(file_path)
                parts = base.split('-')
                if len(parts) == 3:  # app-YYYY-MM-DD.log
                    # 不带序号的版本，如果存在，说明需要从01开始
                    if max_suffix < 1:
                        max_suffix = 1
                elif len(parts) == 4:  # app-YYYY-MM-DD-01.log
                    suffix_part = parts[3].split('.')[0]  # 01.log
                    suffix_num = int(suffix_part)
                    if suffix_num > max_suffix:
                        max_suffix = suffix_num
            except (ValueError, IndexError):
                continue
        
        # 返回下一个序号，格式化为两位数字
        return f"{max_suffix + 1:02d}"
    
    def _get_current_filename(self) -> str:
        """获取当前日志文件名"""
        if self.current_suffix is None:
            # 没有序号，使用不带序号的格式
            return os.path.join(self.log_dir, f"{self.base_filename}-{self.current_date}.log")
        else:
            # 有序号
            return os.path.join(self.log_dir, f"{self.base_filename}-{self.current_date}-{self.current_suffix}.log")
    
    def _should_rotate(self) -> bool:
        """检查是否需要轮转"""
        # 检查日期是否变化
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self.current_date:
            return True
        
        # 检查文件大小
        if os.path.exists(self.current_file):
            file_size = os.path.getsize(self.current_file)
            if file_size >= self.max_bytes:
                return True
        
        return False
    
    def _do_rotate(self):
        """执行轮转"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        # 如果日期变化，重置序号
        if today != self.current_date:
            self.current_date = today
            self.current_suffix = None
        else:
            # 同一天，增加序号
            if self.current_suffix is None:
                # 第一个轮转，从01开始
                self.current_suffix = "01"
            else:
                # 已经有序号，增加
                self.current_suffix = self._get_next_suffix()
        
        # 更新当前文件路径
        self.current_file = self._get_current_filename()
        
        # 关闭旧文件，打开新文件
        self.close()
        self.baseFilename = self.current_file
        self.stream = self._open()
    
    def emit(self, record: logging.LogRecord):
        """记录日志，检查是否需要轮转"""
        if self._should_rotate():
            self._do_rotate()
        
        try:
            super().emit(record)
        except Exception:
            self.handleError(record)


def setup_logger(name: str, log_file: Optional[str] = None, log_level: str = "INFO", 
                 use_console_color: bool = True, log_dir: str = "logs") -> logging.Logger:
    """
    配置日志记录器
    
    Args:
        name: 日志记录器名称，通常使用 __name__
        log_file: 日志文件路径，为 None 时只输出到控制台（已废弃，保留兼容）
        log_level: 日志级别，可选值: DEBUG, INFO, WARNING, ERROR, CRITICAL
        use_console_color: 是否在控制台使用颜色输出
        log_dir: 日志目录
        
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    
    # 设置日志级别
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 控制台处理器（带颜色）
    console_formatter = ColoredFormatter(
        use_color=use_console_color,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（使用新的按天和大小轮转）
    try:
        # 不带颜色的文件格式化器
        file_formatter = ColoredFormatter(
            use_color=False,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # 创建按天和大小轮转的文件处理器
        file_handler = DailyTimedRotatingFileHandler(
            base_filename="app",
            log_dir=log_dir,
            max_bytes=10 * 1024 * 1024,  # 10MB
            encoding='utf-8'
        )
        
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Failed to create file logger: {e}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器，如果尚未配置则使用默认配置
    
    Args:
        name: 日志记录器名称
        
    Returns:
        日志记录器
    """
    logger = logging.getLogger(name)
    
    # 如果没有处理器，配置默认处理器
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


def setup_global_logging(log_file: Optional[str] = None, log_level: str = "INFO", 
                       use_console_color: bool = True, log_dir: str = "logs"):
    """
    配置全局日志
    
    Args:
        log_file: 日志文件路径（已废弃，保留兼容）
        log_level: 日志级别
        use_console_color: 是否在控制台使用颜色输出
        log_dir: 日志目录
    """
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # 清除现有处理器
    root_logger.handlers = []
    
    # 控制台处理器（带颜色）
    console_formatter = ColoredFormatter(
        use_color=use_console_color,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # 文件处理器（使用新的按天和大小轮转）
    try:
        # 不带颜色的文件格式化器
        file_formatter = ColoredFormatter(
            use_color=False,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # 创建按天和大小轮转的文件处理器
        file_handler = DailyTimedRotatingFileHandler(
            base_filename="app",
            log_dir=log_dir,
            max_bytes=10 * 1024 * 1024,  # 10MB
            encoding='utf-8'
        )
        
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Failed to create global file logger: {e}")
