import os
import sys
import inspect
from loguru import logger


def set_logger_style():
    logger.remove() # remove default console output 
    logger.add(
        sys.stdout,
        colorize=True,
        format=('<green>{time:YYYY-MM-DD :mm:ss}</green> '
                '<level>{level:<8}</level> '
                '<cyan>{extra[stack_info]:<40}</cyan> <level>{message}</level>')
    )


def log_rank_0(level):
    def log_func_of_level(*args):
        rank = int(os.environ.get("RANK", -1))
        if rank <= 0:
            caller = inspect.stack()[1]
            caller_frame = caller.frame
            function_name = caller_frame.f_code.co_name
            # module_name = caller_frame.f_globals["__name__"].split(".")[-1]
            file_name = os.path.basename(caller_frame.f_globals["__file__"])
            line = caller.lineno
            getattr(logger, level)(*args, stack_info=f"{file_name}:{function_name}:{line}")
    return log_func_of_level


def patch_logger():
    for level in ["error", "warning", "info", "debug"]:
        setattr(logger, level + "_rank_0", log_rank_0(level))

