from utils.logger_initializer import init_custom_resolvers, init_logger

init_logger()
init_custom_resolvers()

__all__ = ["init_logger", "init_custom_resolvers"]