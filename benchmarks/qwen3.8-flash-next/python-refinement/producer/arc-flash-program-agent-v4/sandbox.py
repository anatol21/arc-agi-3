"""Fresh bubblewrap namespace per grid. No fallback to host execution."""
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path


class SandboxUnavailable(RuntimeError):pass


def execute_case(code,input_grid,timeout=12):
    executable=shutil.which('bwrap')
    if not executable:raise SandboxUnavailable('bwrap is required; refusing host execution')
    if len(code.encode())>65536:return {'executable':False,'output':None,'error':'Program exceeds 64 KiB','seconds':0}
    with tempfile.TemporaryDirectory(prefix='arc-program-') as temp:
        folder=Path(temp)
        (folder/'program.py').write_text(code)
        (folder/'input.json').write_text(json.dumps(input_grid))
        command=[executable,'--unshare-all','--die-with-parent','--new-session','--cap-drop','ALL',
                 '--ro-bind','/usr','/usr']
        for directory in ('/bin','/lib','/lib64'):
            if Path(directory).exists():command+=['--ro-bind',directory,directory]
        command+=['--proc','/proc','--remount-ro','/proc','--dev','/dev','--remount-ro','/dev','--dir','/tmp',
                  '--ro-bind',str(folder/'program.py'),'/program.py',
                  '--ro-bind',str(folder/'input.json'),'/input.json',
                  '--ro-bind',str(Path(__file__).with_name('worker.py').resolve()),'/worker.py',
                  '--remount-ro','/','--chdir','/tmp','/usr/bin/python3','-I','-S','/worker.py']
        started=time.monotonic()
        with (folder/'stdout').open('wb') as out,(folder/'stderr').open('wb') as err:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,
                                     env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'})
            try:process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait()
                return {'executable':False,'output':None,'error':'Execution wall-time limit','seconds':time.monotonic()-started}
        elapsed=time.monotonic()-started
        stderr=(folder/'stderr').read_bytes()[:2000].decode(errors='replace')
        if 'bwrap:' in stderr:
            raise SandboxUnavailable('Sandbox setup failed: '+stderr)
        if (folder/'stdout').stat().st_size>2*1024**2:
            return {'executable':False,'output':None,'error':'Execution output limit','seconds':elapsed}
        try:
            response=json.loads((folder/'stdout').read_text())
            if not isinstance(response,dict) or type(response.get('executable')) is not bool or 'output' not in response:
                raise ValueError('Wrong worker schema')
        except (ValueError,UnicodeError,RecursionError):
            return {'executable':False,'output':None,'error':'Worker failed or exceeded resources; no valid result',
                    'exit_code':process.returncode,'stderr':stderr,'seconds':elapsed}
        if process.returncode!=0:
            response.update(executable=False,output=None,error='Worker terminated',exit_code=process.returncode)
        response['seconds']=elapsed
        return response


def execute(code,inputs):
    # One failure cannot prevent execution and feedback for other demonstrations.
    return {'cases':[execute_case(code,grid) for grid in inputs]}


# Bubblewrap 0.4.1 sets PWD after --chdir (upstream bubblewrap.c:2635).
# Check its exact sandbox value; never accept arbitrary inherited variables.
ENVIRONMENT_PROBE = """def solve(grid):
    import os
    allowed={'PATH','LANG','LC_CTYPE','PWD'}
    unexpected=sorted(set(os.environ)-allowed)
    if unexpected:
        raise RuntimeError('Unexpected environment variable names: '+','.join(unexpected))
    if os.environ.get('PWD')!='/tmp' or os.getcwd()!='/tmp':
        raise RuntimeError('Sandbox working directory must be /tmp')
    if os.environ.get('PATH')!='/usr/bin:/bin' or os.environ.get('LANG')!='C.UTF-8':
        raise RuntimeError('Sandbox PATH/LANG differs from the explicit child environment')
    return grid
"""


def preflight():
    identity='def solve(grid):\n    print("diagnostic output")\n    return grid\n'
    case=execute_case(identity,[[1,0],[0,1]])
    if not case['executable'] or case['output']!=[[1,0],[0,1]] or 'diagnostic output' not in case.get('stdout',''):
        raise SandboxUnavailable('Sandbox identity/stdout check failed: '+json.dumps(case))
    case=execute_case(ENVIRONMENT_PROBE,[[1]])
    if not case['executable'] or case['output']!=[[1]]:
        raise SandboxUnavailable('Environment isolation probe failed: '+json.dumps(case))
    probe='''def solve(grid):
    import os, socket, errno
    assert not os.path.exists('/pfs') and not os.path.exists('/home')
    assert not any(os.path.exists('/dev/'+name) for name in ('nvidia0','nvidiactl','dri','kfd'))
    for path in ('/tmp/probe-write','/dev/probe-write'):
        try:
            with open(path,'w') as stream:stream.write('unexpected')
        except OSError as error:
            assert error.errno==errno.EROFS
        else:raise RuntimeError('Unexpected writable filesystem: '+path)
    s=socket.socket();s.settimeout(0.2)
    try:s.connect(('1.1.1.1',443))
    except OSError:return grid
    raise RuntimeError('Unexpected external network access')
'''
    case=execute_case(probe,[[1]])
    if not case['executable'] or case['output']!=[[1]]:
        raise SandboxUnavailable('Workspace/network isolation probe failed: '+json.dumps(case))
    return {'identity_execution':True,'stdout_capture':True,'workspace_hidden':True,'network_probe_blocked':True,
            'environment_allowlist':True,'readonly_filesystem_probe':True,'gpu_devices_hidden':True,
            'writable_scratch_bytes':0,'worker_python':'/usr/bin/python3 -I -S','case_wall_seconds':12,
            'case_cpu_seconds':5,'case_memory_MiB':768}


if __name__=='__main__':print('SANDBOX_PREFLIGHT '+json.dumps(preflight()),flush=True)
