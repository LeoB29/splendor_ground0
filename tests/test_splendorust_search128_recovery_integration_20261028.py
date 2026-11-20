"""Bounded saved-evidence integration, no model calls or game generation."""
import time
from tools.splendorust_search128_recovery_20261028 import common as c,runner


class Boundary:
    def __init__(self):
        self.deadline=time.monotonic()+90
    def guard(self):
        assert time.monotonic()<self.deadline


def test_exact_saved_bridge_context_and_full_old_game_replay():
    freeze=c.read(c.OLD_OUT/'freeze.json')
    freeze['recovery_manifest']=c.artifact(c.OUT/'recovery_manifest.json')
    boundary=Boundary()
    context=runner.recovery_context(freeze,boundary)
    assert len(context['old_completed'])==44 and len(context['prefix_rows'])==39
    assert context['bridge']['source_row']['clock']['accepted'] is False
    assert context['bridge']['bridge']['timing_evidence']['observed_complete_response_seconds']==1.0749384
    # Exercise real artifact binding, scientific row validation and independent
    # ledger replay of one already finished main game through the new dispatcher.
    spec=context['schedule'][6]
    entry,ending,rows=runner.game_manifest_entry(spec,context,freeze,boundary)
    assert spec['stage']=='main' and spec['game_index']==0
    assert ending['reason']=='natural' and ending['complete'] is True
    assert len(rows)==ending['plies'] and len(entry['events'])==len(rows)
    assert all(x['kind']=='ordinary' and x['source_freeze_sha256']==c.OLD_FREEZE_SHA for x in entry['events'])
    assert c.read(c.OLD_OUT/'completion.json')['status']=='incomplete'
