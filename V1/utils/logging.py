# -*- coding: utf-8 -*-
"""
日志工具模块
提供统一的日志配置和管理，支持彩色控制台输出
"""

import logging
import os
from typing import Optional


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
        # 保存原始记录内容
        orig_message = record.getMessage()
        
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


def setup_logger(name: str, log_file: Optional[str] = None, log_level: str = "INFO", use_console_color: bool = True) -> logging.Logger:
    """
    配置日志记录器
    
    Args:
        name: 日志记录器名称，通常使用 __name__
        log_file: 日志文件路径，为 None 时只输出到控制台
        log_level: 日志级别，可选值: DEBUG, INFO, WARNING, ERROR, CRITICAL
        use_console_color: 是否在控制台使用颜色输出
        
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
    
    # 文件处理器（如果指定了文件路径，不带颜色）
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # 不带颜色的文件格式化器
        file_formatter = ColoredFormatter(
            use_color=False,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # 创建文件处理器，支持日志轮转
        try:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,               # 保留5个备份
                encoding='utf-8'
            )
        except ImportError:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
        
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
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


def setup_global_logging(log_file: Optional[str] = None, log_level: str = "INFO", use_console_color: bool = True):
    """
    配置全局日志
    
    Args:
        log_file: 日志文件路径
        log_level: 日志级别
        use_console_color: 是否在控制台使用颜色输出
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
    
    # 文件处理器
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # 不带颜色的文件格式化器
        file_formatter = ColoredFormatter(
            use_color=False,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        try:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding='utf-8'
            )
        except ImportError:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
        
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
