"""Trusted single-case worker; receives only code and one input, no expected output."""
import contextlib
import json
import resource
import sys
import traceback


def limits():
    resource.setrlimit(resource.RLIMIT_CPU,(5,5))
    resource.setrlimit(resource.RLIMIT_AS,(768*1024**2,768*1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE,(2*1024**2,2*1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_NPROC,(32,32))


class BoundedCapture:
    def __init__(self):self.text=''
    def write(self,text):
        self.text+=str(text)[:max(0,8192-len(self.text))]
        return len(text)
    def flush(self):pass


def run(code_path,input_path):
    limits()
    stdout,stderr=BoundedCapture(),BoundedCapture()
    try:
        input_grid=json.load(open(input_path))
        code=open(code_path).read()
        namespace={'__name__':'arc_candidate'}
        with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
            exec(compile(code,'/program.py','exec'),namespace)
            result=namespace['solve'](input_grid)
            # Serialization is part of executable success, separate from valid-grid success.
            json.dumps(result,allow_nan=False)
        return {'executable':True,'output':result,'stdout':stdout.text,'stderr':stderr.text}
    except BaseException as error:
        return {'executable':False,'output':None,'error':type(error).__name__+': '+str(error)[:1000],
                'traceback':traceback.format_exc()[-2000:],'stdout':stdout.text,'stderr':stderr.text}


if __name__=='__main__':
    code_path=sys.argv[1] if len(sys.argv)>1 else '/program.py'
    input_path=sys.argv[2] if len(sys.argv)>2 else '/input.json'
    print(json.dumps(run(code_path,input_path),allow_nan=False))
