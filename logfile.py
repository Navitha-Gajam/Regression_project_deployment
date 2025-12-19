import logging




def setup_logging(script_name):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    #logging.basicConfig(level=logging.INFO,)
    handler = logging.FileHandler(f'C:\\Users\\wishi\\Downloads\\ML\\logs\\{script_name}.log',mode='w')
    formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
