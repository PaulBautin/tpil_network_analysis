
import subprocess
import os

def run(cmd, env=None, return_proc=False, quiet=False):
    merged_env = os.environ.copy()
    if env is not None:
        if not isinstance(env, dict):
            raise TypeError('Provided `env` must be a dictionary, not {}'
                            .format(type(env)))
        merged_env.update(env)

    opts = {}
    if quiet:
        opts = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    proc = subprocess.run(cmd, env=merged_env, shell=True, check=True,
                          universal_newlines=True, **opts)

    if return_proc:
        return proc