import logging
from parameter_selection import run_parameter_selection

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(filename)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


if __name__ == '__main__':
    run_parameter_selection()