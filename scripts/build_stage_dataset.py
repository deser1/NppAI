"""Author training-only stage targets and freeze a separate synthetic pilot suite."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Project-authored annotations: no model-generated rationale is accepted as truth.
# References are consulted only for train signatures, never held-out supervision.
ANNOTATIONS = {
 'add': ('sum of two numbers', 'Zero and negative numbers allowed.', 'Return a + b.'),
 'subtract': ('first number minus second', 'Zero and negative numbers allowed.', 'Return a - b.'),
 'multiply': ('product of two numbers', 'Zero and negative numbers allowed.', 'Return a * b.'),
 'square': ('square of a number', 'Zero and negative numbers allowed.', 'Return x * x.'),
 'negate': ('negation of a number', 'Zero and negative numbers allowed.', 'Return -x.'),
 'absolute': ('absolute value', 'Zero and negative numbers allowed.', 'Return abs(x).'),
 'length': ('number of list items', 'Empty list returns zero.', 'Return len(values).'),
 'total': ('sum of list items', 'Empty list returns zero.', 'Return sum(values).'),
 'largest': ('maximum list item', 'List must be nonempty.', 'Return max(values).'),
 'smallest': ('minimum list item', 'List must be nonempty.', 'Return min(values).'),
 'reverse': ('reversed text', 'Empty text returns empty text.', 'Return text[::-1].'),
 'lower': ('lowercase text', 'Empty text returns empty text.', 'Return text.lower().'),
 'upper': ('uppercase text', 'Empty text returns empty text.', 'Return text.upper().'),
 'strip': ('text without outer whitespace', 'All whitespace returns empty text.', 'Return text.strip().'),
 'even': ('whether integer is even', 'Zero and negative integers allowed.', 'Return x % 2 == 0.'),
 'positive': ('whether number is positive', 'Zero is not positive.', 'Return x > 0.'),
}


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def derive_training(records):
    selected = {}
    result = []
    for r in records:
        family = r['family']
        if not r['id'].startswith('train-') or family not in ANNOTATIONS:
            raise ValueError('Only recognized training-split records may supply supervision')
        names = selected.setdefault(family, [])
        if r['function'] not in names:
            if len(names) == 2:
                continue
            names.append(r['function'])
        tree = ast.parse(r['answer'])
        f = tree.body[0]
        if not isinstance(f, ast.FunctionDef) or f.name != r['function']:
            raise ValueError('Training reference signature mismatch')
        args = ', '.join(a.arg for a in f.args.args)
        outcome, edge, plan = ANNOTATIONS[family]
        spec = f'{f.name}({args}): {outcome}. {edge}'
        result.append({**r, 'specification': spec, 'plan': plan})
    if set(selected) != set(ANNOTATIONS):
        raise ValueError('Missing training family')
    return result


def new_cases():
    tasks = [
      ('sum', 'suma', 'a, b', 'the sum of a and b', 'sumę a i b', 'return a + b',
       [([0,0],0), ([-3,5],2), ([8,9],17)]),
      ('absolute', 'modul', 'x', 'the absolute value of x', 'wartość bezwzględną x', 'return abs(x)',
       [([0],0), ([-7],7), ([4],4)]),
      ('total', 'razem', 'values', 'the sum of values; empty list returns zero',
       'sumę values; pusta lista daje zero', 'return sum(values)',
       [([[]],0), ([[1,-2,4]],3), ([[8]],8)]),
      ('even', 'parzysta', 'x', 'whether integer x is even, including zero and negative integers',
       'czy liczba całkowita x jest parzysta, także dla zera i liczb ujemnych', 'return x % 2 == 0',
       [([0],True), ([-3],False), ([-4],True)]),
      ('factorial', 'silnia', 'n', 'factorial of nonnegative integer n; zero gives one',
       'silnię nieujemnej liczby całkowitej n; dla zera zwróć jeden',
       'result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result',
       [([0],1), ([1],1), ([5],120), ([10],3628800)]),
      ('positive_sum', 'dodatnie', 'values', 'the sum of positive values; ignore zero and negatives; empty list gives zero',
       'sumę dodatnich elementów values; pomiń zero i ujemne; pusta lista daje zero',
       'return sum(x for x in values if x > 0)',
       [([[]],0), ([[-2,0,3,4]],7), ([[-8]],0)]),
    ]
    result = []
    for family, name, args, en, pl, body, tests in tasks:
        for language in ('en', 'pl'):
            task = (f'Implement {name}({args}) in Python returning {en}.' if language == 'en'
                    else f'Napisz w Pythonie funkcję {name}({args}) zwracającą {pl}.')
            result.append({'id':f'new-{family}-{language}', 'family':family, 'function':name,
                'prompt':task, 'answer':f'def {name}({args}):\n    {body}\n',
                'cases':[{'function':name, 'args':a, 'kwargs':{}, 'expected':v} for a,v in tests]})
    return result


def build(destination):
    source = ROOT/'evaluation/instruction-v2/train.json'
    expected = json.loads((source.parent/'manifest.json').read_bytes())['splits']['train']['sha256']
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError('Frozen source train bytes changed')
    destination.mkdir(parents=True, exist_ok=True)
    train = derive_training(json.loads(source.read_bytes()))
    (destination/'train.json').write_bytes(encoded(train))
    (destination/'new-evaluation.json').write_bytes(encoded(new_cases()))
    manifest = {'version':1, 'source_train_sha256':expected,
        'limits':'Authored correlated synthetic templates; new names/wording, six tasks repeated in English/Polish. Families have prior project exposure. Not an independent general coding benchmark.',
        'protocol':{'seed':42,'steps_per_arm':1024,'batch_size':4,'context_bytes':1024,
            'stage_token_budget':128,'learning_rate':0.001,'gradient_clip':1.0,
            'arms':['direct','staged_mix'], 'staged_mix':'one direct, specification, plan, code example per update; rotate assignment across identical sampled base tasks',
            'selection':'first two distinct names per train family; retain original wording variants',
            'initial_model_sha256':'6a49081e1d94e85d31efe6bdf2a4e3d9e4d7b06bb381c492de3a98d1300ed9ac',
            'context_expansion':'preserve original 256 positions; initialize remaining positions once from seed42; identical expanded state for both arms',
            'promotion':False},
        'files':{name:{'sha256':hashlib.sha256((destination/name).read_bytes()).hexdigest(),
                       'count':len(json.loads((destination/name).read_bytes()))}
                 for name in ('train.json','new-evaluation.json')}}
    (destination/'manifest.json').write_bytes(encoded(manifest))
    return manifest


if __name__ == '__main__':
    print(json.dumps(build(ROOT/'evaluation/staged-protocol-v1'), indent=2))
