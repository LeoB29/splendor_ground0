"""Qualified ownership checks with one explicit prospective service CPU."""
from tools.splendorust_teacher_variability_20261113.ownership import (
    inspect_service_tree,make_book,validate_snapshot,NATIVE_ROLES)
from . import common as c

def validate_book(book,expected_binaries,console_artifact,*,affinity):
    c.require(type(affinity) is list and len(affinity)==1 and type(affinity[0]) is int,
              'One frozen model CPU')
    c.require(type(book) is dict and set(book)=={'schema','service','native_pids','startup','end','affinity',
      'all_owned_identities','native_identities','expected_binaries','console_artifact'},'Complete ownership book')
    c.require(book['expected_binaries']==expected_binaries and book['console_artifact']==console_artifact and
              book['affinity']==affinity,'Prospective exact service affinity and executables')
    expected=make_book(book['service'],book['native_pids'],book['startup'],book['end'],
                       expected_binaries,console_artifact,affinity=affinity)
    c.require(book==expected,'Exact ownership and complete cleanup union')
    return dict(status='passed',native_roles=sorted(NATIVE_ROLES),native_processes=3,
                observed_owned_descendants=len(book['all_owned_identities']),all_console_identities_retained=True)
