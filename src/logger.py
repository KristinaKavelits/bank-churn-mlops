import logging

class EmojiFormatter(logging.Formatter):
    """Custom logging formatter to add emojis based on the log level."""
    
    FORMATS = {
        logging.DEBUG: "⚪ %(asctime)s - %(name)s - DEBUG - %(message)s",
        logging.INFO: "✅ %(asctime)s - %(name)s - INFO - %(message)s",
        logging.WARNING: "⚠️ %(asctime)s - %(name)s - WARNING - %(message)s",
        logging.ERROR: "❌ %(asctime)s - %(name)s - ERROR - %(message)s",
        logging.CRITICAL: "🚨 %(asctime)s - %(name)s - CRITICAL - %(message)s"
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a logger with the EmojiFormatter.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevent adding multiple handlers if get_logger is called multiple times
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(EmojiFormatter())
        logger.addHandler(ch)
        
    # Prevent log messages from being propagated to the root logger
    logger.propagate = False
    
    return logger