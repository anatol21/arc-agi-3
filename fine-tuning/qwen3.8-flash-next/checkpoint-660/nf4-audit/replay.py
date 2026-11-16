"""Inspect live packed base weights after loading the pinned training configuration. No training."""
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


def write(path, data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n')
    tmp.replace(path)


def canonical(name):
    return name.removeprefix('base_model.model.').replace('.base_layer.', '.')


def inventory(model, bnb):
    from bitsandbytes.nn.parametrize import Bnb4bitParametrization
    quantized={}
    def record(name, param, state, route):
        if state is None: raise ValueError('Missing quant_state for '+name)
        name=canonical(name)
        if name in quantized: raise ValueError('Duplicate canonical name: '+name)
        quantized[name]=dict(route=route, quant_type=state.quant_type, nested=state.state2 is not None,
            shape=list(state.shape), storage_shape=list(param.shape), storage_dtype=str(param.dtype),
            storage_bytes=param.numel()*param.element_size(), device=str(param.device),
            blocksize=state.blocksize, absmax_elements=state.absmax.numel(), absmax_device=str(state.absmax.device))
    for name,module in model.named_modules():
        if isinstance(module,bnb.nn.Linear4bit):
            p=module.weight
            if not isinstance(p,bnb.nn.Params4bit):raise ValueError('Unexpected packed parameter at '+name)
            record(name+'.weight',p,p.quant_state,'bitsandbytes.Linear4bit')
        parameterizations=getattr(module,'parametrizations',None)
        if parameterizations is not None:
            for pname,chain in parameterizations.items():
                for op in chain:
                    if isinstance(op,Bnb4bitParametrization):
                        record(name+'.'+pname,chain.original,op.quant_state,'bitsandbytes.Bnb4bitParametrization')
    # The physical inventory includes all other parameters, adapters, and bases
    # underneath PEFT. Avoid accessing virtual full-size dequantized experts.
    physical=[dict(name=n,shape=list(p.shape),dtype=str(p.dtype),device=str(p.device),
                   cls=type(p).__name__,trainable=p.requires_grad)
              for n,p in model.named_parameters(remove_duplicate=False)]
    return quantized,physical


def main():
    folder=Path(__file__).resolve().parent
    result=folder/os.environ['SLURM_JOB_ID']
    result.mkdir()
    started=time.time()
    report=dict(job=os.environ['SLURM_JOB_ID'],historical_job='7853809',phase='loading',
                optimizer_steps=0,scope='Fresh training-configuration reload; no historical packed payload retained')
    write(result/'report.json',report)
    try:
        import torch
        from axolotl.cli.config import load_cfg
        from axolotl.cli.utils import load_model_and_tokenizer
        import bitsandbytes as bnb
        cfg_path=Path.cwd()/'arc-flash-run-v6/arc-flash.yml'
        source=json.loads(cfg_path.read_text())
        cfg_sha=hashlib.sha256(cfg_path.read_bytes()).hexdigest()
        if cfg_sha!=json.loads((cfg_path.parent/'preparation.json').read_text())['config_sha256']:
            raise ValueError('Training config fingerprint mismatch')
        if torch.cuda.device_count()!=3 or int(os.getenv('WORLD_SIZE','1'))!=1:
            raise RuntimeError('Require three GPUs in one process')
        cfg=load_cfg(str(cfg_path))
        # Isolate loader output from the original training/export directory.
        cfg['output_dir']=str(result/'loader')
        report.update(training_config_sha256=cfg_sha,revision=source['revision_of_model'],gpus=[torch.cuda.get_device_name(i) for i in range(3)],
                      versions=dict(torch=torch.__version__,bitsandbytes=bnb.__version__))
        model,_,_=load_model_and_tokenizer(cfg=cfg,inference=False)
        sys.path.insert(0,str(cfg_path.parent))
        from ple_placement import place_small_ple_buffers
        from guarded_train import check_layout
        place_small_ple_buffers(model)
        report['layout']=check_layout(model)
        for i in range(3):torch.cuda.synchronize(i)
        quant,physical=inventory(model,bnb)
        write(result/'inventory.json',dict(quantized=quant,physical_parameters=physical))
        expected=json.loads((folder/'expected.json').read_text())
        report.update(missing=sorted(set(expected)-set(quant)),unexpected=sorted(set(quant)-set(expected)),
            invalid_state=[k for k,v in quant.items() if v['quant_type']!='nf4' or v['storage_bytes']<=0
                           or v['absmax_elements']<=0 or not v['nested'] or not v['device'].startswith('cuda')],
            shape_mismatches=[k for k in quant.keys()&expected.keys() if quant[k]['shape']!=expected[k]],
            linear=sum(v['route']=='bitsandbytes.Linear4bit' for v in quant.values()),
            experts=sum(v['route']=='bitsandbytes.Bnb4bitParametrization' for v in quant.values()))
        report['phase']='verified' if not any(report[k] for k in ('missing','unexpected','invalid_state','shape_mismatches')) else 'differences_observed'
    except BaseException:
        report.update(phase='failed',error=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds']=time.time()-started
        write(result/'report.json',report)
        print('NF4_REPLAY',json.dumps(report),flush=True)


if __name__=='__main__':main()
