"""Debug completion requires observable evidence, not a correct ARC answer."""
import json
from pathlib import Path

if __package__:
    from .setup_runtime import NAME,ready,atomic_json,digest,utcnow
    from .score import collect
    from .protocol import rows,grid
else:
    from setup_runtime import NAME,ready,atomic_json,digest,utcnow
    from score import collect
    from protocol import rows,grid


def check_debug(folder):
    _,recipe=ready(folder);output=folder.parent/'runs'/NAME
    states,frontier=collect(folder,recipe)
    task=rows(folder/'prompts.jsonl')[0]['task_id']
    receipt={'phase':'incomplete','task_id':task,'benchmark_id':recipe['benchmark_id'],
             'checked_at':utcnow(),'full_run_submitted':False,'arms':{}}
    required=[states.get((task,arm)) for arm in ('finetuned','base')]
    if not all(s and s['complete'] for s in required):
        atomic_json(output/'debug-status.json',receipt)
        print('PROGRAM_DEBUG_INCOMPLETE: saved work can resume; this is not a pass',flush=True)
        return receipt
    try:
        sandbox=json.loads((output/'sandbox-preflight.json').read_text())
        if not all(sandbox.get(k) is True for k in ('identity_execution','stdout_capture','workspace_hidden','network_probe_blocked','environment_allowlist','readonly_filesystem_probe','gpu_devices_hidden')):
            raise ValueError('Sandbox evidence incomplete')
        reload=json.loads((output/'model-reload.json').read_text())
        if reload['optimizer_updates']!=0 or reload['requires_grad_parameters']!=0:raise ValueError('Model was not frozen')
        if not reload['adapter']['roundtrip_equal'] or reload['adapter']['parameters']!=33478656:raise ValueError('Adapter reload evidence incomplete')
        if sorted(reload['layout']['decoder_layers_per_gpu'].values())!=[16,16,16]:raise ValueError('GPU placement evidence incomplete')
        template=json.loads((output/'thinking-template-validation.json').read_text())
        if not template['thinking_template_differs'] or not template['prefilled_open_think']:raise ValueError('Thinking template evidence incomplete')
        report=json.loads((output/'report.json').read_text())
        if report['benchmark_id']!=recipe['benchmark_id'] or report['state_frontier']!=frontier:raise ValueError('Scorer frontier differs')
        for arm,state in zip(('finetuned','base'),required):
            index=state['selected_candidate']
            if index is None:raise ValueError(arm+': no executable candidate')
            selected=state['attempts'][index];response=selected['response']
            if (response['finish_reason']!='eos' or response['separation_error'] is not None
                or not response['reasoning'] or response['reasoning_tokens']<1
                or not response['accounting_validated'] or response['adapter_layers_checked']<1):
                raise ValueError(arm+': missing reasoning, clean stopping, token or adapter evidence')
            for case in state['selected_query_execution']['cases']:
                if not case['executable']:raise ValueError(arm+': selected query program did not execute')
                grid(case['output'])
            trace=output/task/arm
            for attempt in state['attempts']:
                if 'code' in attempt and (trace/('candidate-'+str(attempt['candidate'])+'.py')).read_text()!=attempt['code']:
                    raise ValueError('Inspection code differs from execution evidence')
            for label in ('first','selected'):
                if json.loads((trace/(label+'-output.json')).read_text())!=state[label+'_output_json']:
                    raise ValueError('Inspection output differs from execution evidence')
            receipt['arms'][arm]={'candidates':len(state['attempts']),'selected_candidate':index,
                'demonstrations_passed':selected['feedback']['passed'],'demonstrations_total':selected['feedback']['total'],
                'new_generated_tokens':state['total_generated_tokens'],'state_sha256':digest(trace/'state.json')}
        receipt.update(phase='passed',report_id=report['report_id'],evidence_sha256={name:digest(output/name) for name in ('sandbox-preflight.json','model-reload.json','thinking-template-validation.json')})
    except (ValueError,KeyError,OSError) as failure:
        receipt.update(phase='failed',error=str(failure))
        atomic_json(output/'debug-status.json',receipt)
        raise RuntimeError('Program debug gate failed: '+str(failure)) from failure
    atomic_json(output/'debug-status.json',receipt)
    print('PROGRAM_DEBUG_PASSED '+json.dumps(receipt),flush=True)
    print('Review programs, reasoning, grids and runtime before authorizing a full run.',flush=True)
    return receipt


if __name__=='__main__':check_debug(Path(__file__).resolve().parent)
