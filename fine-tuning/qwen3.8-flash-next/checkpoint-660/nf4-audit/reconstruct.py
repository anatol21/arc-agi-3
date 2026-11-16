import json, os, sys
from pathlib import Path
from collections import Counter
import torch
import bitsandbytes as bnb
from transformers import AutoConfig, AutoModelForImageTextToText, BitsAndBytesConfig
from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer

work=Path.cwd()
cfg=json.loads((work/'arc-flash-run-v6/arc-flash.yml').read_text())
assert cfg['adapter']=='qlora' and cfg['load_in_4bit'] and cfg['bf16'] and cfg['quantize_moe_experts'] and not cfg.get('bnb_config_kwargs')
source=Path(cfg['base_model'])
config=AutoConfig.from_pretrained(source,local_files_only=True)
assert not getattr(config,'quantization_config',None)
index=json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
quant=BitsAndBytesConfig(load_in_4bit=True,llm_int8_threshold=6.0,llm_int8_has_fp16_weight=False,bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True,bnb_4bit_quant_type='nf4',bnb_4bit_quant_storage=torch.bfloat16)
with torch.device('meta'):
 model=AutoModelForImageTextToText.from_config(config,torch_dtype=torch.bfloat16)
 before={n:(list(p.shape),str(p.dtype)) for n,p in model.named_parameters(remove_duplicate=False)}
 quantizer=Bnb4BitHfQuantizer(quant,pre_quantized=False)
 quantizer._process_model_before_weight_loading(model,device_map=cfg['device_map'])
 places=cfg['device_map']
 rows=[]
 for module_name,module in model.named_modules():
  for name,param in module.named_parameters(recurse=False):
   full=f'{module_name}.{name}' if module_name else name
   assert param.device.type=='meta' and full in before
   prefixes=[prefix for prefix in places if not prefix or full==prefix or full.startswith(prefix+'.')]
   mapped=places[max(prefixes,key=len)] if prefixes else None
   kind='not_selected'
   if isinstance(module,bnb.nn.Linear4bit) and name=='weight':
    assert isinstance(param,bnb.nn.Params4bit) and param.quant_type=='nf4'
    kind='bnb_linear_weight'
   elif len(before[full][0])>=3 and 'expert' in full.lower() and mapped not in ('cpu','disk',None) and not isinstance(module,(bnb.nn.Linear4bit,bnb.nn.Linear8bitLt)):
    kind='axolotl_expert_weight'
   rows.append(dict(name=full,shape=before[full][0],category=kind,module=type(module).__name__,assigned_device=mapped,shard=index.get(full)))
print(json.dumps(dict(checkpoint_revision=cfg['revision_of_model'],skip_modules=quantizer.modules_to_not_convert,indexed=len(index),counts=Counter(r['category'] for r in rows),parameters=rows,source_names_without_direct_runtime_match=sorted(set(index)-{r['name'] for r in rows}))))
