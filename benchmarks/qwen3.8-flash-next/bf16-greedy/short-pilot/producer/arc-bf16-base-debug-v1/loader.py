import csv,json,re,time,sys
from pathlib import Path
from runpy import run_path


def inputs(native_dir, inventory_path):
    cfg=json.loads((native_dir/'inference-config.json').read_text())
    layout={}
    for key,gpu in cfg["device_map"].items():
        m=re.match(r"model\.language_model\.layers\.(\d+)(\.|$)", key)
        layout[key]=int(m[1])//12 if m else (3 if gpu==2 else gpu)
    if set(layout.values())!={0,1,2,3} or "" in layout:
        raise RuntimeError("Expected four-stage explicit placement")
    cfg.update(device_map=layout, load_in_4bit=False, load_in_8bit=False, quantize_moe_experts=False,
               adapter=None,lora_model_dir=None,ple_cpu_offload=False,gradient_checkpointing=False,
               experimental_skip_move_to_device=True,attn_needs_dtype_cast=True)
    for k in list(cfg):
        if k.startswith("lora_") or k.startswith("peft_"): cfg.pop(k)
    expected={}
    with inventory_path.open() as f:
        for row in csv.DictReader(f):
            expected[row["logical_tensor"]]=tuple(map(int,row["shape"].split("x")))
    if len(expected)!=1497: raise ValueError("Incorrect logical base inventory")
    return cfg,layout,expected


def preflight(model_path,native_dir,inventory_path):
    import torch
    from transformers import AutoConfig,AutoModelForImageTextToText
    cfg,layout,expected=inputs(native_dir,inventory_path)
    config=AutoConfig.from_pretrained(model_path,local_files_only=True)
    if config.model_type!='qwen4_exp' or getattr(config,'quantization_config',None):
        raise ValueError('Wrong source model')
    with torch.device("meta"):
        model=AutoModelForImageTextToText.from_config(config,torch_dtype=torch.bfloat16)
    names={n:tuple(p.shape) for n,p in model.named_parameters()}
    if expected!=names:
        raise ValueError("Unexpected meta inventory: "+str(sorted(set(names)^set(expected))[:12]))
    ple=run_path(str(native_dir/'ple_placement.py'))
    table=next(k for k in names if k.endswith(ple['PLE_SUFFIX']))
    return dict(placement=layout, **ple['probe_loading_and_hooks'](model,layout,dict(cpu_ple_parameter=table)),
                meta_tensors=len(names),gpu_devices=sorted(set(layout.values())))


def load_base(model_path,tokenizer_path,native_dir,inventory_path,output_dir):
    import torch
    from axolotl.cli.config import load_cfg
    from axolotl.cli.utils import load_model_and_tokenizer
    cfg,layout,expected=inputs(native_dir,inventory_path)
    cfg.update(base_model=str(model_path),tokenizer_config=str(tokenizer_path),output_dir=str(output_dir/'native-loader'))
    path=output_dir/'bf16-config.json';path.write_text(json.dumps(cfg))
    live=load_cfg(str(path))
    if live.adapter or live.load_in_4bit or live.load_in_8bit or live.quantize_moe_experts or live.lora_model_dir:
        raise ValueError("Unexpected adapter/quant settings")
    model,tokenizer,_=load_model_and_tokenizer(cfg=live,inference=False)
    model.eval();model.requires_grad_(False)
    ple=run_path(str(native_dir/'ple_placement.py'))
    buffers=ple['place_small_ple_buffers'](model)
    if any(bool(getattr(model,k,False)) for k in ('peft_config','is_loaded_in_4bit','is_loaded_in_8bit','hf_quantizer','_moe_experts_quantized')):
        raise ValueError('Unexpected quantizer or PEFT state on model')
    for n,m in model.named_modules():
        if ('lora' in n.lower() or type(m).__module__.startswith(('peft.','bitsandbytes.'))
                or getattr(m,'parametrizations',None)):
            raise ValueError(f'Unexpected quantized/adapter module {n}')
    params=dict(model.named_parameters())
    if set(params)!=set(expected):
        raise RuntimeError(f'Live base inventory differs: {sorted(set(params)^set(expected))[:12]}')
    dtypes={};amount={}
    for name,p in params.items():
        expected_dev=ple['mapped_device'](layout,name)
        target='cpu' if expected_dev=='cpu' else f'cuda:{expected_dev}'
        if (tuple(p.shape)!=expected[name] or str(p.device)!=target or p.dtype not in (torch.bfloat16,torch.float32)
                or getattr(p,'quant_state',None) is not None or p.requires_grad):
            raise ValueError(f'Unexpected live weight {name} dtype={p.dtype} device={p.device}')
        key=str(p.dtype);dtypes[key]=dtypes.get(key,0)+1
        k=str(p.device);amount[k]=amount.get(k,0)+p.numel()*p.element_size()
    report=dict(phase='verified',quantized_parameters=0,adapter_tensors=0,base_parameters=len(params),
                dtypes=dtypes,weight_bytes_by_device=amount,ple_buffers=buffers,device_map=layout)
    print('BF16_LOADER_VERIFIED',json.dumps({k:v for k,v in report.items() if k!='device_map'}),flush=True)
    return model,tokenizer,report
