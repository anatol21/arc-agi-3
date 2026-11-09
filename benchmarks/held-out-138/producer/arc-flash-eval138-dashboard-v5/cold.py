"""Opt-in, process-scoped cold-load comparisons for the pinned native loader."""
import ast
import contextlib
from concurrent.futures import ThreadPoolExecutor
import inspect
import threading
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import time
from unittest.mock import patch

from sdk import file_hash, model_cache_path, prepare_cache


@contextlib.contextmanager
def packed_experts(cache, report):
    import torch
    import bitsandbytes
    import bitsandbytes.nn.parametrize as ops
    from safetensors.torch import load_file, save_file

    original = ops.replace_parameter_4bit
    def replace(module, param_name, compress_statistics=False, quant_type='nf4', blocksize=None):
        weight = getattr(module,param_name)
        device = weight.device
        # Include the actual BF16 bytes presented to the native quantizer, including fusion order.
        raw = weight.detach().contiguous().view(torch.uint8).cpu()
        source = hashlib.sha256(memoryview(raw.numpy())).hexdigest()
        del raw
        path, identity = model_cache_path(cache, weights=source,
            loader=['bitsandbytes',bitsandbytes.__version__,torch.__version__,torch.version.cuda,file_hash(ops.__file__)],
            precision=[str(weight.dtype),quant_type,compress_statistics,blocksize],
            layout=[list(weight.shape),list(torch.cuda.get_device_capability(device))])
        name = 'expert.safetensors'
        def build(stage):
            original(module,param_name,compress_statistics=compress_statistics,quant_type=quant_type,blocksize=blocksize)
            param = getattr(module.parametrizations,param_name)
            values = param[0].quant_state.as_dict(packed=True)
            tensors = {k:v.detach().cpu().contiguous() for k,v in values.items()}
            tensors['packed'] = param.original.detach().cpu().contiguous()
            save_file(tensors,str(stage/name))
            return [name]
        before = time.monotonic()
        hit = prepare_cache(path,identity,build)
        if hit:
            tensors = load_file(str(path/name))
            packed = tensors.pop('packed').to(device)
            setattr(module,param_name,torch.nn.Parameter(packed,requires_grad=False))
            ops.replace_parameter_4bit_prequantized(module,param_name,tensors,device)
        else:
            report['preparation_seconds'] += time.monotonic()-before
        report['expert_hits' if hit else 'expert_builds'] += 1

    report.update(expert_hits=0,expert_builds=0,preparation_seconds=0)
    with patch.object(ops,'replace_parameter_4bit',replace):
        yield
    if report['expert_hits']+report['expert_builds'] != 96:
        raise RuntimeError('Native expert dispatch count differs from verified loader')


def table_parts(source):
    mapping=json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
    prefix='model.language_model.layers.1.ple.ple_embedding.ngram_embedding'
    count=json.loads((source/'config.json').read_text())['text_config']['split_ngram_parts']
    headers={}
    result=[]
    for i in range(count):
        key=f'{prefix}.shard_{i}.weight'
        path=source/mapping[key]
        if path not in headers:
            with path.open('rb') as f:
                size=struct.unpack('<Q',f.read(8))[0]
                headers[path]=(size+8,json.loads(f.read(size)))
        offset,header=headers[path]
        t=header[key]
        begin,end=t['data_offsets']
        rows,width=t['shape']
        if t['dtype']!='BF16' or end-begin!=2*rows*width or (result and width!=result[0][4]):
            raise ValueError('Incompatible tensor shape or encoding')
        result.append((path,offset+begin,end-begin,rows,width))
    return result


def transfer(parts, output=None):
    digest=hashlib.sha256()
    for path,offset,size,_,_ in parts:
        with path.open('rb') as f:
            f.seek(offset)
            while size:
                block=f.read(min(8<<20,size))
                if not block: raise EOFError('Truncated checkpoint tensor')
                digest.update(block)
                if output is not None: output.write(block)
                size-=len(block)
    return digest.hexdigest()


@contextlib.contextmanager
def mapped_table(source, cache, report):
    import torch
    from transformers import core_model_loading as loading
    parts=table_parts(source)
    shape=[sum(p[3] for p in parts),parts[0][4]]
    # An identity receipt avoids rescanning the host table on every fresh process.
    signatures=[[str(p),p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns,off,n] for p,off,n,_,_ in parts]
    from sdk import atomic_json, read_json
    sig=hashlib.sha256(json.dumps(signatures).encode()).hexdigest()
    receipt=cache/'table-sources'/(sig+'.json')
    before=time.monotonic()
    source_digest=read_json(receipt)
    if source_digest is None:
        source_digest=transfer(parts)
        atomic_json(receipt,source_digest)
    path,identity=model_cache_path(cache,weights=source_digest,loader='raw-bf16-cat-dim0/v1',
                                   precision='BF16',layout=shape)
    def build(stage):
        with (stage/'table.bf16').open('xb') as f:
            if transfer(parts,f)!=source_digest: raise ValueError('Source changed during concat')
        return ['table.bf16']
    hit=prepare_cache(path,identity,build)
    report.update(table_cache_hit=hit,preparation_seconds=0 if hit else time.monotonic()-before)
    original=loading.WeightConverter.convert
    matches=[]
    def convert(converter,layer_name,*args,**kwargs):
        if not layer_name.endswith('.ple.ple_embedding.ngram_embedding.weight'):
            return original(converter,layer_name,*args,**kwargs)
        ops=converter.operations
        if not (converter.force_cpu and not converter.quantization_operation and len(ops)==1 and
                isinstance(ops[0],loading.Concatenate) and ops[0].dim==0 and
                ops[0].num_shards_attribute=='split_ngram_parts'):
            raise RuntimeError('Installed table conversion changed')
        model=kwargs['model']
        expected=model.get_parameter(layer_name)
        if list(expected.shape)!=shape or expected.dtype!=torch.bfloat16:
            raise ValueError('Model table does not match the prepared artifact')
        matches.append(layer_name)
        mapped=torch.from_file(str(path/'table.bf16'),shared=False,size=shape[0]*shape[1],dtype=torch.bfloat16)
        return {layer_name:mapped.view(*shape)}
    with patch.object(loading.WeightConverter,'convert',convert):
        yield
    if len(matches)!=1: raise RuntimeError('Expected exactly one prepared host table')


def copy_verified(source_file, dest):
    target = dest / source_file.name
    shutil.copyfile(source_file, target)
    if file_hash(source_file) != file_hash(target):
        raise ValueError(f'Local snapshot differs: {source_file.name}')


@contextlib.contextmanager
def local_snapshot(source,modules,report,*,workers=1):
    tmp=os.environ.get('TMPDIR')
    if not tmp: raise RuntimeError('Slurm did not provide node-local TMPDIR')
    files = [path for path in source.iterdir() if path.is_file()]
    required = sum(path.stat().st_size for path in files)
    if shutil.disk_usage(tmp).free<required*1.1: raise RuntimeError('Insufficient node-local disk for exact snapshot')
    before=time.monotonic()
    cpu_before=time.process_time()
    with tempfile.TemporaryDirectory(prefix='restart-base-',dir=tmp) as d:
        dest=Path(d)
        if workers == 1:
            for path in files:
                copy_verified(path, dest)
        else:
            available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
            if available is not None and workers > available:
                raise RuntimeError('Copy workers exceed the CPUs available to this process')
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(copy_verified, path, dest) for path in files]
                for future in futures:
                    future.result()
        report.update(preparation_seconds=time.monotonic()-before,staged_bytes=required,
                      copy_workers=workers,verified_files=len(files),staging_cpu_seconds=time.process_time()-cpu_before)
        original=modules['configuration'].load_inference_config
        def config(path):
            cfg=original(path)
            if Path(cfg['base_model']).resolve()!=source: raise ValueError('Base path changed')
            cfg['base_model']=str(dest)
            return cfg
        with patch.object(modules['configuration'],'load_inference_config',config): yield


def loader_context(kind,source,cache,modules,report):
    report['artifact']=kind
    if kind=='packed_experts': return packed_experts(cache,report)
    if kind=='mapped_table': return mapped_table(source,cache,report)
    if kind=='local_snapshot': return local_snapshot(source,modules,report)
    if kind in ('local_snapshot_8','local_snapshot_16','local_snapshot_32'):
        workers = int(kind.rsplit('_', 1)[1])
        return local_snapshot(source,modules,report,workers=workers)
    raise ValueError('Unknown loader trial')


@contextlib.contextmanager
def materialization_times(report):
    import transformers.core_model_loading as loading
    expected=ast.parse("""def _materialize_copy(tensor: torch.Tensor, device=None, dtype=None) -> torch.Tensor:
        tensor = tensor[...]
        if dtype is not None or device is not None:
            tensor = tensor.to(device=device, dtype=dtype)
        return tensor""").body[0]
    actual=ast.parse(inspect.getsource(loading._materialize_copy)).body[0]
    if ast.dump(ast.Module(body=expected.body,type_ignores=[]))!=ast.dump(ast.Module(body=actual.body,type_ignores=[])):
        raise RuntimeError('Installed materializer changed; review before instrumenting')
    lock=threading.Lock()
    times=defaultdict(float)
    def measured(tensor,device=None,dtype=None):
        start=time.monotonic()
        tensor=tensor[...]
        mid=time.monotonic()
        dest='cuda' if str(device).startswith('cuda') or isinstance(device,int) else 'cpu'
        if dtype is not None or device is not None:
            tensor=tensor.to(device=device,dtype=dtype)
        end=time.monotonic()
        with lock:
            times['slice_seconds']+=mid-start
            times[dest+'_to_seconds']+=end-mid
            times['calls']+=1
        return tensor
    try:
        with patch.object(loading,'_materialize_copy',measured):
            yield
    finally:
        report['worker_thread_materialization']=dict(times)
        report['materialization_scope']='Sum of worker host time. Concurrent calls overlap. File page faults can occur during the subsequent copy.'
