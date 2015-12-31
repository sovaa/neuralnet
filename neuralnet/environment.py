import os
import logging

from .commands import all_commands

logger = logging.getLogger(__name__)


class config_dict:
    def __init__(self, params={}, override={}, parent=None):
        self.params = params
        self.parent = parent
        self.override = override

    def subp(self, parent):
        p = dict(parent.params)
        p.update(self.params)
        p.update(self.override)
        return config_dict(p, self.override, parent)

    def sub(self, **params):
        p = dict(self.params)
        p.update(params)
        p.update(self.override)
        return config_dict(p, self.override, self)

    def get(self, key, default=None, params=None):
        def config_format(s, params):
            if isinstance(s, list):
                return [config_format(r, params) for r in s]

            if isinstance(s, dict):
                kw = dict()
                for k, v in s.items():
                    kw[k] = config_format(v, params)
                return kw

            try:
                import re

                keydb = set('{' + key + '}')

                while True:
                    sres = re.search("{.*?}", s)
                    if sres == None:
                        break

                    # avoid using the same reference twice
                    if sres.group() in keydb:
                        raise RuntimeError(
                            "found circular dependency in config value '{0}' using reference '{1}'".format(s,
                                                                                                           sres.group()))
                    keydb.add(sres.group())
                    s = s.format(**params)

                return s
            except KeyError as e:
                raise RuntimeError("missing configuration key: " + str(e))

        if params is None:
            params = self.params

        if key in self.params:
            return config_format(self.params.get(key), params)

        if self.parent:
            return self.parent.get(key, default, params)

        if default is not None:
            return config_format(default, params)

        raise KeyError(key)

    def __contains__(self, key):
        if key in self.params:
            return True
        if self.parent:
            return key in self.parent
        return False

    def __iter__(self):
        for k in sorted(self.params.keys()):
            yield k


class NNEnvironment(object):
    def __init__(self, root_path, config):
        """
        Initialize the environment with each specific amount of servers and modules.
        """
        self.root = root_path
        self.config = config
        self.commands = dict()

        for klass in all_commands:
            inst = klass()
            inst.setenv(self)
            self.commands[klass.command.lower()] = inst

    def setup(self):
        pass

    def list_commands(self):
        return self.commands

    def get_command(self, command_string):
        command = self.commands.get(command_string.lower(), None)
        if command is None:
            raise RuntimeError("no such command: " + command_string)
        return command

    def shutdown(self):
        pass


def validate_config(environ, config):
    pass


def create_components(env, environ, klass):
    comps = list()
    for k, v in environ[klass.__group__].items():
        sub = dict(name=k)
        sub.update(v)
        comps.append(klass(env.config.sub(**sub)))
    return comps


def create_env(root, environ, opts):
    config = dict(environ)

    if "config" in environ:
        config.update(environ["config"])

    config["root"] = root
    config["cwd"] = os.getcwd()

    config = config_dict(config, opts)

    try:
        validate_config(environ, config)
    except RuntimeError as e:
        raise RuntimeError("Invalid schema: " + str(e))

    env = NNEnvironment(root, config)

    try:
        env.setup()
    except RuntimeError as e:
        raise RuntimeError("Invalid environment: {0}".format(str(e)))

    return env
