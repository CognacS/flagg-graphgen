import subprocess


from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import time

def run_cmd(cmd, output, timeout=None):
    """
    Run command (list form) with subprocess.run, return a dict with results.
    cmd: list, e.g. ['python', '-c', 'import time; time.sleep(1)']
    timeout: seconds or None
    """
    start = time.time()
    try:
        proc = subprocess.run(cmd, text=True, check=False, timeout=timeout, stdout=open(output, 'wb'))
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "duration": time.time() - start,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "cmd": cmd,
            "returncode": None,
            "stdout": e.stdout or "",
            "stderr": str(e),
            "duration": time.time() - start,
            "timed_out": True,
        }

def run_parallel(commands, outputs, max_workers=4, timeout=None):
    """
    commands: list of list (each inner is argv list)
    outputs: list of output file paths corresponding to each command
    max_workers: maximum concurrent subprocesses
    timeout: per-process timeout in seconds (or None)
    Returns: list of result dicts in order of completion.
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(run_cmd, cmd, output, timeout): cmd for cmd, output in zip(commands, outputs)}
        for fut in as_completed(futures):
            res = fut.result()
            # simple progress print
            print(f"Finished: {res['cmd']} -> rc={res['returncode']} (took {res['duration']:.2f}s)")
            results.append(res)
    return results


def main():
    print('Running set of commands...')
    common_commands = [
        "python",
        "main.py",
        "+preset/ifh/qm9=1-4",
        "seed=0",
        "load_ckp=2",
        "enable_log=false",
        "compile=false",
        "mode=eval",
    ]
    
    jumps = [1, 2, 5, 10, 20, 25, 50, 100, 250, 500]
    
    options = [f"+options=[remap_cls,djump_{j}]" for j in jumps]
    outputs = [f"outputs/djump_1_4_2/djump_{j}.txt" for j in jumps]
    
    # for option, output in zip(options, outputs):
    #     subprocess.Popen(
    #         common_commands + [option, "|", "tee", output]
    #     )
    
    cmds = [common_commands + [option] for option in options]
    
    all_results = run_parallel(cmds, outputs, max_workers=4)
    print("All commands completed.")
    
if __name__ == "__main__":
    main()