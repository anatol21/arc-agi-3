import ast
import hashlib
import json
import re
import textwrap
from pathlib import Path

BLOCKS = ("AX", "BX", "BY", "AY")
RETRY_MINUTES = {"BX": 26, "BY": 22, "AY": 22}
PANEL = {"X": "29c11459 1b60fb0c 342dd610 6455b5f5".split(),
         "Y": "e633a9e5 5792cb4d 6ca952ad d93c6891".split()}
PROBE = """Before your first solver you may instead submit one fenced ```probe block containing Python statements. Variables train and test contain the visible demonstration pairs and query inputs. Use print to inspect them. Each call starts fresh. You have at most two calls, 512 output tokens per call and 256 tokens of feedback. You may submit the solver immediately."""
SOLVER = "Return the complete solver in one fenced python block."


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, separators=(',', ':')))
    tmp.replace(path)


def new_state(task, condition, identity):
    return dict(task=task, condition=condition, identity=identity, attempts=[], probes=[],
                prefix=[], rng=None, used=0, reasoning=0, total=0, done=False)


def messages(controller, visible, state):
    history = controller.messages_for(visible, state['attempts'])
    if state['condition'] == 'B' and not state['attempts']:
        history[0]['content'] += '\n' + PROBE
        for call in state['probes']:
            history.extend([dict(role='assistant', content=call['text']),
                            dict(role='user', content='Python output:\n' + call['observation'] +
                                 ('\n' + SOLVER if len(state['probes']) == 2 else '\nContinue.'))])
    return history


def probe_body(text):
    if text.count('```') != 2:
        return None
    match = re.fullmatch(r'\s*```(probe|python)[ \t]*\n(.*?)\n```\s*', text, re.S)
    if not match:
        return None
    body = match[2]
    if match[1] == 'python':
        try:
            nodes = ast.parse(body).body
        except SyntaxError:
            return None
        if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == 'solve' for n in nodes):
            return None
        if not any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                   and isinstance(n.value.func, ast.Name) and n.value.func.id == 'print' for n in nodes):
            return None
    return body


def run_probe(text, visible, execute, tokenizer):
    body = probe_body(text)
    if body is None:
        raw = 'Error: malformed probe'
    else:
        source = "def solve(data):\n    train, test = data['train'], data['test']\n"
        source += textwrap.indent(body, '    ') + '\n    return []'
        case = execute(source, visible)
        raw = case.get('stdout', '') + ('\n' + case.get('error', 'execution failed') if not case['executable'] else '')
    ids = tokenizer.encode(raw, add_special_tokens=False)
    return tokenizer.decode(ids[:256], skip_special_tokens=False)


def finish_turn(state, text, end, controller, visible, execute, tokenizer):
    before_first = state['condition'] == 'B' and not state['attempts'] and len(state['probes']) < 2
    if before_first and (text.lstrip().startswith('```probe') or probe_body(text) is not None):
        if len(tokenizer.encode(text, add_special_tokens=False)) > 512:
            end = 'probe_limit'
        observation = (run_probe(text, visible, execute, tokenizer) if end == 'eos'
                       else 'Error: incomplete or over-budget probe')
        state['probes'].append(dict(text=text, observation=observation, tokens=list(state['prefix'])))
        state['prefix'] = []
        return
    demos = visible['train']
    attempt = dict(candidate=len(state['attempts']), response=dict(final_answer=text, finish_reason=end),
                   tokens=list(state['prefix']), charged_tokens=state['used'], reasoning_tokens=state['reasoning'])
    try:
        if end != 'eos':
            raise ValueError(end)
        code = controller.extract_code(text)
        controller.validate_code(code)
    except (ValueError, TypeError, SyntaxError) as exc:
        feedback = dict(passed=0, total=len(demos), executed_cases=0, error=str(exc)[:2000])
    else:
        attempt['code'] = code
        feedback = controller.feedback({'cases':[execute(code, d['input']) for d in demos]}, demos)
    attempt['feedback'] = feedback
    state['attempts'].append(attempt)
    state.update(prefix=[], used=0, reasoning=0, rng=None)
    if feedback['passed'] == len(demos) or len(state['attempts']) == 4 or end == 'context_limit':
        select_program(state,visible,execute)


def select_program(state,visible,execute):
    eligible = [a for a in state['attempts'] if a['feedback']['executed_cases']]
    best = max(eligible, key=lambda a:a['feedback']['passed'], default=None)
    state['selected'] = best['candidate'] if best else None
    for label, a in [('first',state['attempts'][0]), ('selected', best)]:
        if label == 'selected' and a is state['attempts'][0]:
            state[label] = state['first']
        else:
            state[label] = [execute(a['code'], t['input']).get('output') if a and 'code' in a else None
                            for t in visible['test']]
    state['selected_candidate'] = best['candidate'] if best else None
    state['done'] = True


def allowance(state, prompt):
    return min(8192-state['used'], 32768-state['total'], 18432-len(prompt)-len(state['prefix']))
