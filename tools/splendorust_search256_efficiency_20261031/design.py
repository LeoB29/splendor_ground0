"""Exactly two persistent256 trees; no game or replacement lifecycle."""
from . import common as c

def lifecycle_inventory():
    return {name:dict(route='research',graph_request_id='request-00000001',
        entrypoint_request_id='request-00000002',ordinary_first_request_id='request-00000003',
        ordinary_requests=96,startup_requests=2) for name in c.EFFICIENCY_OBSERVERS}
