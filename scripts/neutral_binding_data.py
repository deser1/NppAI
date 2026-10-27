"""Reviewed neutral-name views derived only from the prior authored TRAIN bank.

This module never imports exposed development, diagnostic or released task rows.
Only authored controls are renamed; generated code is never transformed.
"""
import ast
import copy
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def view(base, name, arguments, leading, identity, cell, primary=False):
    row = copy.deepcopy(base)
    mapping = dict(zip(base['arguments'], arguments))
    tree = ast.parse(base['answer'])
    tree.body[0].name = name
    for node in tree.body[0].args.args:
        node.arg = mapping[node.arg]
    for node in ast.walk(tree.body[0]):
        if isinstance(node, ast.Name):
            node.id = mapping.get(node.id, node.id)
    description = base['request']['prompt'].split('. ', 1)[1]
    row.update(id=identity, function=name, arguments=arguments, answer=ast.unparse(tree)+'\n',
               cell=cell, primary=primary, leading=leading, split='train' if cell=='training' else 'development')
    row['request'] = dict(prompt=' '*leading + ('Implement ' if row['language']=='en' else 'Napisz funkcję ')
                          + name+'('+', '.join(arguments)+'). '+description, max_new_tokens=256)
    for case in row['cases']:
        case['function'] = name
    return row


def banks():
    source = json.loads((ROOT/'evaluation/function-body-v1/train.json').read_bytes())
    training, development = [], []
    for family in sorted({r['family'] for r in source}):
        bases = [next(r for r in source if r['family']==family and r['id'].endswith(f'-0-0-{w}')) for w in range(4)]
        arity = len(bases[0]['arguments'])
        for n, a, w in itertools.product(range(3), range(2), range(4)):
            training.append(view(bases[w], ('task_aa','task_bb','task_cc')[n],
                (['aa','bb'] if a==0 else ['cc','dd'])[:arity], 0,
                f'train-{family}-{n}-{a}-{w}', 'training'))
        for w in (0, 2):
            for n, a, shift in itertools.product((0,1), repeat=3):
                cell = f'name{n}-args{a}-shift{shift}'
                development.append(view(bases[w], 'task_dd' if n else 'task_aa',
                    (['ee','ff'] if a else ['aa','bb'])[:arity], 8*shift,
                    f'dev-{family}-{w}-{cell}', cell, primary=bool(n and a)))
            for field in ('name', 'args'):
                cell = 'long-'+field
                development.append(view(bases[w], 'task_extended' if field=='name' else 'task_aa',
                    ['input_left','input_right'][:arity] if field=='args' else ['aa','bb'][:arity], 0,
                    f'dev-{family}-{w}-{cell}', cell))
    return training, development
