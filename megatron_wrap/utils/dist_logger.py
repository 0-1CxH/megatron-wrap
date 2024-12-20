import os
import sys
import inspect
from loguru import logger


def set_logger_style():
    logger.remove() # remove default console output 
    logger.add(
        sys.stdout,
        colorize=True,
        level=os.environ.get("CONSOLE_LOG_LEVEL", "DEBUG"),
        format=('<green>{time:YYYY-MM-DD hh:mm:ss}</green> '
                '<level>{level:<8}</level>'
                '<blue>{extra[stack_info]:<40}</blue> <level>{message}</level>')
    )


def dist_log(level, rank0):
    def log_func_of_level(*args, **kwargs):
        flag = False
        rank = int(os.environ.get("RANK", -1))
        if rank0 is True:
            if rank <= 0:
                flag = True
        else:
            flag = True
        if flag is True:
            caller = inspect.stack()[1]
            caller_frame = caller.frame
            function_name = caller_frame.f_code.co_name
            # module_name = caller_frame.f_globals["__name__"].split(".")[-1]
            file_name = os.path.basename(caller_frame.f_globals["__file__"])
            line = caller.lineno
            message = " | ".join([str(_) for _ in args])
            if not rank0:
                message = f"(rank{rank}) " + message
            getattr(logger, level)(message, stack_info=f"{file_name}:{function_name}:{line}")
    return log_func_of_level


def patch_logger():
    levels = ["error", "warning", "info", "debug"]
    for level in levels:
        setattr(logger, level + "_rank_0", dist_log(level, rank0=True))
        setattr(logger, level + "_all_ranks", dist_log(level, rank0=False))
    logger.debug_rank_0(f"[PATCH] logger patched, use ({'|'.join(levels)})_(rank_0|all_ranks) instead of the original to avoid unexpected behavior")

