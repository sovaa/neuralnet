import sys
import os
import logging
import getopt

logger = logging.getLogger(__name__)

from .environment import create_env


def exit_usage():
    print("Usage: neuralnet <command>")
    print("General options:")
    print("-c <config>      : Specify another configuration file than default")
    print("-l <level>       : Specify logging level, one of ERROR, WARNING, INFO, DEBUG")
    # print("-D <key>=<value> : Perform a temporary override of a configuration value")
    sys.exit(1)


def entry():
    """
    Locate nn.yaml in the current working directory and bootstrap an interactive environment.
    """
    import yaml
    import json

    opts = {
        "config": ["nn.yaml", "nn.json"]
    }

    try:
        getopts, args = getopt.gnu_getopt(sys.argv[1:], "c:l:")
    except getopt.GetoptError as e:
        error("Option parsing failed: " + str(e))
        sys.exit(1)

    for (o, v) in getopts:
        if o == "-c":
            opts["config"] = [v]
        if o == "-l":
            opts["log_level"] = v
        #if o == "-D":
        #    if "=" not in v:
        #        opts[v] = True
        #    else:
        #        key, val = v.split("=", 1)
        #        opts[key] = ast.literal_eval(val)

    config_paths = opts["config"]
    config_path = None
    config_dict = None

    for conf in config_paths:
        path = os.path.join(os.getcwd(), conf)

        if not os.path.isfile(path):
            continue

        try:
            if conf.endswith(".yaml"):
                config_dict = yaml.load(open(path))
            elif conf.endswith(".json"):
                config_dict = json.load(open(path))
            else:
                error("Unsupported file extension: {0}".format(conf))
                sys.exit(1)
        except Exception as e:
            error("Failed to open configuration {0}: {1}".format(conf, str(e)))
            sys.exit(1)

        config_path = path
        break

    if not config_dict:
        error("No configuration found: {0}\n".format(", ".join(config_paths)))
        exit_usage()
        sys.exit(1)

    root = os.path.dirname(config_path)

    try:
        env = create_env(root, config_dict, opts);
    except RuntimeError as e:
        error(str(e))
        sys.exit(1)

    if "log_level" in opts:
        log_level = opts["log_level"]
    else:
        log_level = env.config.get("log_level", "INFO")

    if not hasattr(logging, log_level):
        error("No such log level: " + log_level)
        sys.exit(1)

    f = "%(asctime)s - %(name)-30s - %(levelname)-7s - %(message)s"
    logging.basicConfig(level=getattr(logging, log_level), format=f)

    if len(args) < 1:
        exit_usage()

    command = args[0]
    args = args[1:]

    try:
        command = env.get_command(command)
    except RuntimeError as e:
        error("Command error: " + str(e))
        exit_usage()

    try:
        args = command.validate(args)
    except RuntimeError as e:
        error("Invalid arguments: " + str(e))
        error("")
        error("Usage:", command.usage)
        error("Short:", command.short)
        sys.exit(1)

    status = 0

    try:
        if command.execute(*args):
            logger.info("Command Successful")
        else:
            logger.info("Command Failed")
            status = 1
    finally:
        env.shutdown()

    sys.exit(status)


def error(text, args=None):
    if args is None:
        print(text, file=sys.stderr)
    else:
        print(text, args, file=sys.stderr)