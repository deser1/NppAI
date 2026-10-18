"""Frozen oracle interventions: assisted diagnostics, never autonomous gains."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import protected_sequential as pilot
from scripts.alias_copy_failure_diagnostic import source_starts
from scripts.audit_protected_sequential import retained

MODES = ('autonomous', 'oracle_start', 'oracle_exit', 'oracle_both')


def oracle_span(prompt, name):
    starts = source_starts(prompt, name)
    if len(starts) != 1:
        raise ValueError('oracle requires exactly one requested source span')
    return starts[0], len(name.encode())


class ReadoutProxy:
    """Use the unchanged decoder; intervene only on its start/exit readouts."""
    def __init__(self, model, prompt, name, mode):
        if mode not in MODES:
            raise ValueError('unknown intervention')
        self.model, self.base, self.mode = model, model.base, mode
        self.start, self.length = oracle_span(prompt, name)
        self.frame_length = len(('[USER]: '+prompt+'\n[AI]:\n').encode())
        self.query = -1
        self.readouts = []

    def components(self, tokens, mask):
        hidden, vocab, attention = self.model.components(tokens, mask)
        self.query += 1
        offset = max(0, self.frame_length+self.query-self.base.max_seq_len)
        chosen = int(attention[0, -1].argmax())
        record = {'query': self.query, 'vocabulary_byte': int(vocab[0, -1].argmax()),
                  'learned_start': offset+chosen,
                  'learned_start_probability': float(attention[0, -1, chosen]),
                  'oracle_start_readout': False}
        if self.query == 4 and self.mode in ('oracle_start', 'oracle_both'):
            position = self.start-offset
            if not 0 <= position < tokens.shape[1] or not bool(mask[0, position]):
                raise ValueError('oracle source is outside the causal prompt mask')
            attention = attention.clone()
            attention[0, -1] = 0
            attention[0, -1, position] = 1
            record['oracle_start_readout'] = True
        self.readouts.append(record)
        return hidden, vocab, attention

    def exit_probability(self, query_hidden, source_hidden):
        probability = self.model.exit_probability(query_hidden, source_hidden)
        self.readouts[-1]['learned_exit_probability'] = float(probability)
        if self.mode in ('oracle_exit', 'oracle_both'):
            # Response-length oracle only: does not fix the selected start.
            probability = probability.new_full(probability.shape, float(self.query >= 4+self.length))
        self.readouts[-1]['effective_exit_probability'] = float(probability)
        return probability


def generate(model, prompt, name, mode):
    from scripts.sequential_copy_model import generate as decode
    proxy = ReadoutProxy(model, prompt, name, mode)
    output, events, raw = decode(proxy, prompt, with_events=True)
    triggered = bool(events)
    return {'output': output, 'output_bytes': raw, 'events': events,
            'readouts': proxy.readouts, 'triggered': triggered,
            'assisted': mode != 'autonomous', 'mode': mode}


def load_checkpoint(archive, output, manifest, report):
    from scripts.sequential_copy_model import load_bundle
    raw = archive.read_bytes()
    if pilot.quality.digest(raw) != manifest['archive_sha256']:
        raise ValueError('source archive hash differs')
    with zipfile.ZipFile(archive) as bundle:
        weight = bundle.read('frozen_sequential/0/experimental.nppaicopy')
        if weight != bundle.read('frozen_sequential/1/experimental.nppaicopy'):
            raise ValueError('source repeated weights differ')
    expected = report['arms']['frozen_sequential']['identity']['checkpoint_sha256']
    if pilot.quality.digest(weight) != expected:
        raise ValueError('pinned checkpoint differs')
    path = output/'frozen.nppaicopy'
    path.write_bytes(weight)
    return load_bundle(path)[0]


def run(archive, output, image):
    import torch
    import sandbox_validate
    from scripts.training_repro import configure, provenance, state_digest
    configure(torch, pilot.quality.PROFILE)
    output.mkdir(parents=True, exist_ok=True)
    directory = pilot.ROOT/'docs/evidence/protected-sequential'
    manifest = json.loads((directory/'manifest.json').read_bytes())
    old = retained(directory)
    dataset = pilot.ROOT/'evaluation/instruction-v3'
    task_path = pilot.ROOT/'evaluation/gated-copy-probes-v1/tasks.json'
    pilot.audit(old, dataset, task_path)
    groups, inputs = pilot.prior.inputs(dataset, task_path)
    model = load_checkpoint(archive, output, manifest, old)
    initial_state = state_digest(model)
    report = {'modes': {}, 'functional': {}, 'promotion': False,
              'checkpoint_sha256': old['arms']['frozen_sequential']['identity']['checkpoint_sha256'],
              'source_archive_sha256': manifest['archive_sha256'], 'inputs': inputs,
              'runtime': provenance(torch, pilot.quality.PROFILE),
              'source_sha256': {p: pilot.quality.digest((pilot.ROOT/p).read_bytes()) for p in
                  ('scripts/alias_oracle_diagnostic.py', 'scripts/sequential_copy_model.py')},
              'limits': 'Exposed regression inputs only. Oracle modes are assisted interventions, not autonomous success or mathematical upper bounds. No training/promotion. Exit-only forces the requested byte length even after a wrong start; neither oracle supplies the def-space trigger.'}
    for mode in MODES:
        result = {}
        for group, rows in groups.items():
            cases = []
            for index, row in enumerate(rows):
                case = generate(model, row['prompt'], row['function'], mode)
                repeated = generate(model, row['prompt'], row['function'], mode)
                if case != repeated:
                    raise ValueError('same-host repeated generation differs')
                if mode == 'autonomous':
                    historical = old['arms']['frozen_sequential']['after'][group][index]
                    if any(case[k] != historical[k] for k in ('output', 'output_bytes', 'events')):
                        raise ValueError('autonomous historical replay differs: '+group+'/'+row['id'])
                cases.append({'id': row['id'], 'index': index, **case,
                              'metrics': pilot.base.score(row, case['output'])})
            result[group] = {'total': len(cases), 'metrics': dict(Counter(
                metric for c in cases for metric, passed in c['metrics'].items() if passed))}
            path = output/(mode+'-'+group.replace('/', '-')+'.json')
            path.write_bytes(pilot.quality.encoded(cases))
        report['modes'][mode] = result
        print(json.dumps({'mode': mode, 'summary': result}), flush=True)
        # Always write a partial record, but mark completion only after all gates.
        (output/'report.json').write_bytes(pilot.quality.encoded(report))
    for mode in MODES:
        functional = []
        for index, task in enumerate(json.loads(task_path.read_bytes())):
            tests, suite = sandbox_validate.load_cases(Path(task['cases_file']))
            names = {c['function'] for c in tests}
            if len(names) != 1:
                raise ValueError('oracle functional suite must request one callable')
            name = names.pop()
            case = generate(model, task['prompt'], name, mode)
            if case != generate(model, task['prompt'], name, mode):
                raise ValueError('functional generation repeat differs')
            source = case['output']
            if mode == 'autonomous' and source != old['arms']['frozen_sequential']['functional']['models']['copy']['cases'][index]['source']:
                raise ValueError('functional historical source differs')
            category, detail = 'empty', 'Empty/whitespace output'
            if source.strip():
                status, detail = sandbox_validate.validate(source, 'syntax', tests, image)
                category = 'syntax_failure'
                if status == 0:
                    status, detail = sandbox_validate.validate(source, 'tests', tests, image)
                    category = 'passed' if status == 0 else 'functional_failure'
                if status not in (0, 1):
                    raise RuntimeError('Docker infrastructure failure: '+detail)
            functional.append({**task, **case, 'suite_sha256': suite,
                               'category': category, 'diagnostics': detail})
        report['functional'][mode] = {'total': len(functional),
            'passed': sum(c['category'] == 'passed' for c in functional),
            'image': image, 'cases': functional}
        print(json.dumps({'functional_mode': mode, 'passed': report['functional'][mode]['passed'],
                          'total': len(functional)}), flush=True)
    if state_digest(model) != initial_state:
        raise ValueError('diagnostic changed model weights')
    report['complete'] = True
    report['state_unchanged'] = True
    report['baseline_replay_equal'] = True
    (output/'report.json').write_bytes(pilot.quality.encoded(report))
    hashes = {p.name: pilot.quality.digest(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()}
    (output/'manifest.json').write_bytes(pilot.quality.encoded(hashes))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image', required=True)
    args = parser.parse_args()
    run(args.archive, args.output, args.image)
