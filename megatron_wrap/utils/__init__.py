from .dist_logger import logger, set_logger_style, patch_logger
from .wandb_logger import WandbWrap
set_logger_style()
patch_logger()
