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
    
    command = [
        "python",
        "main.py"
    ]
    
    common_commands = [
        "enable_log=false",
        "compile=false",
        "mode=eval",
        "+options=node_empirical"
    ]
    
    seeds = [0,1,2]
    blocks = ['1-3', '1-2-8']
    
    seeds = [[f'load_ckp={seed}', f"seed={seed}"] for seed in seeds]
    model = [f"+preset/ifh/enz={block}" for block in blocks]
    outputs = [f"outputs/enz/enz_{seed}_{block}.txt" for seed in seeds for block in blocks]
    
    # for option, output in zip(options, outputs):
    #     subprocess.Popen(
    #         common_commands + [option, "|", "tee", output]
    #     )
    
    cmds = [command + [mdl] + common_commands + seed for seed in seeds for mdl in model]
    
    all_results = run_parallel(cmds, outputs, max_workers=1)
    print("All commands completed.")
    
if __name__ == "__main__":
    main()