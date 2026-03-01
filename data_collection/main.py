
import os
import sys
import concurrent.futures
import subprocess as sp

def bgpdown(timestamps_t0,cmd):
    os.chdir('/home/data/bgp')
    file_path = f'/home/data/bgp/bgpcons2019/peer-pfx-origins.{timestamps_t0}.gz'
    ret = 0
    if not os.path.exists(file_path):
        os.system(cmd)
    return ret

def makelist():
    from datetime import datetime, timedelta
    import pytz

    tz = pytz.timezone('UTC')

    # Dates
    start_date = tz.localize(datetime(2019, 9, 15))
    end_date = tz.localize(datetime(2020, 12, 31))
    ddelta = timedelta(days=1) # for the collection of a single daily snapshots
    #ddelta = timedelta(minutes=5) # for the collection of 5-mins snapshots
    now = start_date

    cmds = []
    while True:
        if now > end_date: break
        timestamps_t0 = int(now.timestamp())
        bsrt = f'bsrt -d broker -w {timestamps_t0},{timestamps_t0+150} -i 300 -p routeviews -O ./bgpcons_logs/%X.%Y%m%d.%H%M'
        cmd = f'bgpview-consumer -b ascii -i "{bsrt}" -c "peer-pfx-origins -o ./bgpcons2019 -c"'
        cmds.append([timestamps_t0,cmd])
        now += ddelta
    return cmds

if __name__ == '__main__':
    cmds = makelist()
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(bgpdown,timestamps_t0,cmd): (timestamps_t0,cmd) for timestamps_t0,cmd in cmds}

        for future in concurrent.futures.as_completed(futures):
            cmd = futures[future]
            try:
                returncode = future.result()
                print(f'Command {cmd} finished with returncode {returncode}')
            except Exception as e:
                print(f'Command {cmd} raised an exception: {e}')
