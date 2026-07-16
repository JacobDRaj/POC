import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bankruptcy_genbi.log", encoding="utf-8"),
    ],
    force=True,
)

logger = logging.getLogger("bankruptcy_genbi")