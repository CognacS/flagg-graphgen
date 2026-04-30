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
        "seed=0",
        "load_ckp=0",
        "enable_log=false",
        "compile=false",
        "mode=eval",
    ]
    
    molecule_option = ['task.test.molecular.how_many_to_generate=10', '+options=remap_cls']
    generic_option = ['task.test.graph.how_many_to_generate=10']
    num_graphs_comm = {
        'qm9': molecule_option,
        'zinc': molecule_option,
        'comm': generic_option,
        'ego-small': generic_option,
        'enz': generic_option,
        'ego': generic_option
    }
    
    ifh_ds_models = {
        'qm9': ['1-2', '1-4'],
        'zinc' : ['1-3', '1-4-8'],
        'comm': ['1-2', '1-2-8'],
        'ego-small': ['1-2', '1-2-8'],
        'ego': ['1-3', '1-4-16']
    }
    mifh_ds_models = {
        'comm': ['deg-dist'],
        'ego-small': ['deg-dist'],
        'enz': ['deg-dist'],
        'ego': ['deg-dist']
    }
    
    # ifh_ds_models = {
    #     'qm9': ['1-2', '1-4'],
    #     'zinc' : ['1-3', '1-4-8'],
    # }
    # mifh_ds_models = {
    #     'comm': ['deg-dist'],
    #     'ego-small': ['deg-dist'],
    #     'ego': ['deg-dist']
    # }
    
    commands = [
        command + 
        [f"+preset/ifh/{ds}={model}"] + 
        common_commands +
        num_graphs_comm[ds] for ds, models in ifh_ds_models.items() for model in models
    ]
    commands += [
        command + 
        [f"+preset/mifh/{ds}={model}"] + 
        common_commands +
        num_graphs_comm[ds] for ds, models in mifh_ds_models.items() for model in models
    ]

    outputs = [f"outputs/all_gen/out_{ds}_{model}.txt" for ds, models in ifh_ds_models.items() for model in models]
    outputs += [f"outputs/all_gen/out_{ds}_{model}.txt" for ds, models in mifh_ds_models.items() for model in models]
    
    #print(commands)
    
    all_results = run_parallel(commands, outputs, max_workers=6)
    print("All commands completed.")
    
if __name__ == "__main__":
    main()