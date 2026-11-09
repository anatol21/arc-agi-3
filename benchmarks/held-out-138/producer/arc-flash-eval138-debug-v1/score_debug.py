"""Score a fixed validation prefix; never claim full-set accuracy."""
import json
from pathlib import Path
from debug_common import NAME, dependencies


def report(answers, predictions, recipe, receipt, protocol, reload_checked):
    summary, details = protocol.score(answers, predictions)
    ids = receipt['target_task_ids']
    prefix_answers = [answer for answer in answers if answer['task_id'] in set(ids)]
    if len(prefix_answers) != len(ids):
        raise RuntimeError('Scorer prefix does not match staged task IDs')
    pilot, _ = protocol.score(prefix_answers, predictions)
    paired = [task for task in ids if all((task, arm) in predictions for arm in protocol.ARMS)]
    paired_scores = {arm: sum(row['task_correct'] for row in details if row['arm'] == arm and row['task_id'] in paired)
                     for arm in protocol.ARMS}
    all_completed = len(paired) == len(ids)
    responses_healthy = all(predictions[task, arm]['finish_reason'] == 'eos'
        for task in ids for arm in protocol.ARMS) if all_completed else False
    valid_formats = all(row['format_error'] is None for row in details) and len(details) == 2*len(ids)
    phase = ('incomplete' if not all_completed else 'reload_unverified' if not reload_checked
             else 'passed' if responses_healthy and valid_formats else 'response_failure')
    debug = dict(phase=phase, target_task_ids=ids, paired_tasks=len(paired),
                 paired_solved_tasks=paired_scores, model_reload_checked=reload_checked,
                 both_arm_eos=responses_healthy, all_formats_valid=valid_formats,
                 scope='three-task fixed-prefix pilot; not full validation-set accuracy',
                 full_run_submitted=False)
    summary.update(benchmark_id=recipe['benchmark_id'], split='evaluation',
        global_step=660, adapter_sha256=recipe['adapter_sha256'], generation=recipe['generation'],
        protected_evaluation_used=True, ttt=False, debug=debug, prefix_summary=pilot)
    return summary, details


def score_run(folder):
    native, recipe, m, receipt = dependencies(folder)
    output = folder.parent/'runs'/NAME
    with m['common'].output_lock(output):
        answers = m['protocol'].rows(native/'answers.scorer-only.jsonl')
        predictions = m['protocol'].load_predictions(output/'predictions.jsonl', set(receipt['target_task_ids']))
        if any(r.get('benchmark_id') != recipe['benchmark_id'] for r in predictions.values()):
            raise RuntimeError('Debug prediction provenance differs from parent evaluation')
        reload_path = output/'model-reload-validation.json'
        check = json.loads(reload_path.read_text()) if reload_path.exists() else {}
        reload_checked = (check.get('requires_grad_parameters') == 0 and check.get('optimizer_updates') == 0
                          and check.get('layout', {}).get('trainable_parameters') == 33478656)
        summary, details = report(answers, predictions, recipe, receipt, m['protocol'], reload_checked)
        m['common'].atomic_json(output/'summary.json', summary)
        m['common'].atomic_json(output/'debug-status.json', summary['debug'])
        temporary = output/'per-task.jsonl.tmp'
        m['protocol'].write_rows(temporary, details)
        temporary.replace(output/'per-task.jsonl')
    print('ARC_EVAL_DEBUG ' + json.dumps(summary), flush=True)
    return summary


if __name__ == '__main__':
    score_run(Path(__file__).resolve().parent)
