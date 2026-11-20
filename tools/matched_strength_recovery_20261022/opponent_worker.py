"""New startup source authentication around the unchanged qualified opponent."""
from tools.local_player_qualification_128.clock import serve
from tools.splendorust_120_qualification_20261018.worker import initialize as qualified_initialize, handle
from tools.matched_strength_recovery_20261022.worker import authenticate_runtime_sources


def initialize():
    authenticate_runtime_sources()
    return qualified_initialize()


if __name__ == '__main__':
    raise SystemExit(serve(initialize, handle))
