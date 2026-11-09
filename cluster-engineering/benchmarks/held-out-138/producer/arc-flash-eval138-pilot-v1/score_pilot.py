"""Separate diagnostic-cohort reports using the unchanged strict scorer."""
import json
from pathlib import Path
from pilot_common import NAME, dependencies


def report(answers,predictions,recipe,receipt,protocol,reload_checked):
    summary,details = protocol.score(answers,predictions)
    cohorts = {}
    for name,ids in receipt['cohorts'].items():
        subset = [r for r in answers if r['task_id'] in set(ids)]
        if len(subset) != len(ids):raise RuntimeError('Missing scorer cohort IDs')
        cohorts[name],_ = protocol.score(subset,predictions)
        cohorts[name]['task_ids'] = ids
        cohorts[name]['selection'] = ('First ten tasks in the original validation order'
            if name=='ordered_prefix' else 'Selected for prior OpenRouter success; not representative')
    ids=receipt['target_task_ids']
    paired=[task for task in ids if all((task,arm) in predictions for arm in protocol.ARMS)]
    status=dict(phase='complete' if len(predictions)==2*len(ids) else 'partial',
                target_tasks=len(ids),required_predictions=2*len(ids),completed_predictions=len(predictions),
                paired_tasks=len(paired),model_reload_checked=reload_checked,
                scope='selected pilot; no combined pilot accuracy estimate',full_run_submitted=False)
    summary.update(benchmark_id=recipe['benchmark_id'],global_step=660,
                   adapter_sha256=recipe['adapter_sha256'],generation=recipe['generation'],
                   split='evaluation',protected_evaluation_used=True,ttt=False,
                   pilot=status,cohorts=cohorts)
    return summary,details


def score_run(folder):
    native,recipe,m,receipt=dependencies(folder)
    output=folder.parent/'runs'/NAME
    with m['common'].output_lock(output):
        answers=m['protocol'].rows(native/'answers.scorer-only.jsonl')
        predictions=m['protocol'].load_predictions(output/'predictions.jsonl',set(receipt['target_task_ids']))
        if any(r.get('benchmark_id')!=recipe['benchmark_id'] for r in predictions.values()):
            raise RuntimeError('Pilot prediction provenance differs from the parent recipe')
        reload_path=output/'model-reload-validation.json'
        check=json.loads(reload_path.read_text()) if reload_path.exists() else {}
        checked=(check.get('optimizer_updates')==0 and check.get('requires_grad_parameters')==0
                 and check.get('layout',{}).get('trainable_parameters')==33478656)
        summary,details=report(answers,predictions,recipe,receipt,m['protocol'],checked)
        m['common'].atomic_json(output/'summary.json',summary)
        m['common'].atomic_json(output/'pilot-status.json',summary['pilot'])
        temp=output/'per-task.jsonl.tmp';m['protocol'].write_rows(temp,details);temp.replace(output/'per-task.jsonl')
    print('ARC_EVAL_PILOT '+json.dumps(summary),flush=True)
    return summary


if __name__=='__main__':score_run(Path(__file__).resolve().parent)
