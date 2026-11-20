"""Frozen four-cell order and mandatory candidate-X checks / candidate-Y overlap."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED


def run_comparison(cells,play_cell,check_cell,stop_cell,quarantine,boundary=lambda:None):
    """Callbacks launch/drain owned workers; tests use the identical coordinator.

    play_cell(arm, cell) returns only after every actual player process has exited.
    check_cell includes selected fresh-service replays AND saved-evidence audit.
    stop_cell signals between decisions; the owner retains original hard clocks.
    No successful return is possible while an overlapped future is unfinished.
    """
    if set(cells)!= {'X','Y'}:
        raise ValueError('Only the two frozen matched cells')
    receipts=[]
    def checked(arm,key):
        boundary();result=check_cell(arm,cells[key])
        if result.get('status')!='passed':
            raise ValueError(f'{arm} {key} verification failed')
        receipts.append({'arm':arm,'cell':key,'check':result})
        return result
    def played(arm,key):
        boundary();result=play_cell(arm,cells[key])
        if result.get('status') not in ('completed','passed'):
            raise ValueError(f'{arm} {key} play failed; no retry')
        return result
    played('control','X');checked('control','X')
    played('candidate','X')
    with ThreadPoolExecutor(max_workers=2,thread_name_prefix='collection-overlap') as pool:
        # Submit Y before X checks so the intended overlap is possible. Evidence
        # timestamps must independently establish actual overlap at acceptance.
        play_y=pool.submit(played,'candidate','Y')
        check_x=pool.submit(checked,'candidate','X')
        pending={play_y,check_x}
        error=None
        while pending:
            completed,pending=wait(pending,timeout=.1,return_when=FIRST_COMPLETED)
            try:
                boundary()
                for item in completed:
                    item.result()
            except BaseException as caught:
                error=caught;stop_cell('candidate',cells['Y'])
                quarantine('candidate',cells['Y'],str(caught))
                # Active work is drained using its existing clocks, not retried.
                for item in pending:
                    try:item.result()
                    except BaseException:pass
                break
        if error is not None:
            raise error
    checked('candidate','Y')
    played('control','Y');checked('control','Y')
    return dict(status='passed',checks=receipts,retries=0,replacements=0,
        all_workers_drained=True,engineering_only=True)
