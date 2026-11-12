"""Optional research tools with verified models and source-preserving extraction."""
from pathlib import Path
import hashlib
import json
import re
import time

import torch

from .planning_v4 import PlanningEnvironment, PlanningPolicy, rollout
from .structured_extraction import chunks, prompt, output_schema, validate, normalize_items


def annotate_quotes(items, source):
    """Expose ambiguous quote locations and polarity disagreements for human review."""
    polarity = {}
    for item in items:
        polarity.setdefault(item['text'], set()).add(item['present'])
    output = []
    for item in items:
        start, end = item['segment_start'], item['segment_end']
        matches = [{'start': start+m.start(), 'end': start+m.end(),
                    'context': source[max(0, start+m.start()-12):min(len(source), start+m.end()+12)]}
                   for m in re.finditer(re.escape(item['text']), source[start:end])]
        notes = []
        if len(matches) > 1:
            notes.append('本段中有多处相同短语，不能确定模型指向哪一处')
        if len(polarity[item['text']]) > 1:
            notes.append('同一短语有不同存在性标记，请结合时间和上下文核对')
        output.append({**item, 'occurrences': matches, 'review_notes': notes})
    return output


class GroundedExtractor:
    def __init__(self, model_path):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path = Path(model_path)
        manifest = json.loads((path/'verified_model.json').read_text(encoding='utf8'))
        if manifest['revision'] != '989aa7980e4cf806f80c7fef2b1adb7bc71aa306':
            raise ValueError('模型版本与已验证的抽取配置不一致')
        if manifest['weights_sha256'] != 'dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee':
            raise ValueError('模型权重指纹与固定版本不一致')
        with (path/'model.safetensors').open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != manifest['weights_sha256']:
                raise ValueError('模型文件校验失败')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True,
            dtype=torch.float16 if self.device == 'cuda' else torch.float32, attn_implementation='sdpa').to(self.device).eval()
        self.manifest = manifest

    def extract(self, source, parser, progress=None):
        from lmformatenforcer import JsonSchemaParser, CharacterLevelParserConfig
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        if not source.strip() or len(source) > 3000:
            raise ValueError('请输入1至3000字的原文')
        segments = chunks(source); accepted = []; quality = []; started = time.perf_counter()
        for index, (offset, part) in enumerate(segments):
            alphabet = ''.join(sorted(set(CharacterLevelParserConfig().alphabet + part)))
            grammar = JsonSchemaParser(output_schema(part), config=CharacterLevelParserConfig(
                alphabet=alphabet, max_consecutive_whitespaces=1, force_json_field_order=True))
            inputs = self.tokenizer.apply_chat_template(prompt(part), add_generation_prompt=True, return_tensors='pt', return_dict=True)
            if inputs['input_ids'].shape[1] >= 4096:
                raise ValueError('文本分段超过模型上下文限制')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=768, do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
                    prefix_allowed_tokens_fn=build_transformers_prefix_allowed_tokens_fn(self.tokenizer, grammar))
            new_tokens = output[0, inputs['input_ids'].shape[1]:]
            checked = validate(self.tokenizer.decode(new_tokens, skip_special_tokens=True), part)
            accepted.extend({**r, 'start': r['start']+offset, 'end': r['end']+offset, 'segment': index+1,
                             'segment_start': offset, 'segment_end': offset+len(part)} for r in checked['items'])
            quality.append({'segment': index+1, 'invalid_json': checked['invalid_json'], 'invalid_schema': checked['invalid_schema'],
                            'output_limit': len(new_tokens) >= 768, 'rejected': checked['rejected']})
            if progress:
                progress((index+1)/len(segments))
        accepted = annotate_quotes(accepted, source)
        return {'items': accepted, 'normalized': normalize_items(accepted, parser), 'quality': quality,
                'seconds': time.perf_counter()-started, 'model_revision': self.manifest['revision'],
                'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                'scope': '待人工核对的研究抽取；存在性标记由模型判断，原文引用校验不等于医学语义正确'}


def planning_comparison(graph, task, root, variant='residual_kan', seed=42):
    root = Path(root)
    meta = json.loads((root/f'experiments/v4/planning_training/{variant}_{seed}.json').read_text(encoding='utf8'))
    if meta['provenance']['evidence'] != graph.fingerprint:
        raise ValueError('当前证据图与模型训练版本不一致')
    for name in ('src/dmease/planning_v4.py', 'src/dmease/evidence_policy.py', 'src/dmease/research_policy.py'):
        if hashlib.sha256((root/name).read_text(encoding='utf8').encode()).hexdigest() != meta['provenance']['source_lf_sha256'][name]:
            raise ValueError('当前规划实现与模型训练版本不一致')
    path = root/f'models/planning_v4/{variant}_{seed}.pt'
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta['checkpoint_sha256']:
        raise ValueError('规划模型校验失败')
    env = PlanningEnvironment(graph, task)
    policy = PlanningPolicy(env.observation_dim, env.feature_dim, variant)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['provenance'] != meta['provenance']:
        raise ValueError('规划模型来源记录不一致')
    policy.load_state_dict(payload['state_dict']); policy.eval()
    rows = []
    for method in ('greedy', 'lookahead2', 'network', 'exact'):
        env.score.cache_clear(); started = time.perf_counter()
        if method == 'exact':
            masks, scores = env.exact_table(); selected = int(masks[scores.argmax()])
        else:
            selected, _ = rollout(env, policy if method == 'network' else None, method)
        value = env.score(selected)
        rows.append({'method': method, 'objective': value[0], 'symptom_coverage': value[1], 'target_coverage': value[2],
                     'ppi_coverage': value[3], 'seconds': time.perf_counter()-started,
                     'herbs': [graph.herbs[i]['name'] for i in range(env.h) if selected & (1 << i)]})
    greedy, network = rows[0], rows[2]
    rows.append({**(network if network['objective'] > greedy['objective']+1e-7 else greedy), 'method': 'network_with_greedy_guard',
                 'seconds': network['seconds']+greedy['seconds'], 'used': 'network' if network['objective'] > greedy['objective']+1e-7 else 'greedy'})
    gate_file = root/'experiments/v4/gating_selection.json'
    if gate_file.exists():
        from .conservative_planner import gated_rollout
        gate = json.loads(gate_file.read_text(encoding='utf8'))['selections'][f'{variant}_{seed}']
        configuration = gate['candidates'][gate['selected_index']]
        if configuration['mode'] == 'warm':
            warm_path = root/f'models/planning_v4/{variant}_{seed}_warm.pt'
            if hashlib.sha256(warm_path.read_bytes()).hexdigest() != meta['warm_checkpoint_sha256']:
                raise ValueError('预训练模型校验失败')
            policy.load_state_dict(torch.load(warm_path, weights_only=True, map_location='cpu')['state_dict'])
        start = time.perf_counter()
        selected, _, changes = gated_rollout(env, policy if configuration['mode'] != 'greedy' else None, configuration['margin'])
        value = env.score(selected)
        gated = {'method': 'gated_network', 'objective': value[0], 'symptom_coverage': value[1], 'target_coverage': value[2],
                 'ppi_coverage': value[3], 'seconds': time.perf_counter()-start, 'configuration': configuration, **changes,
                 'herbs': [graph.herbs[i]['name'] for i in range(env.h) if selected & (1 << i)]}
        rows.append(gated)
        rows.append({**(gated if gated['objective'] > greedy['objective']+1e-7 else greedy), 'method': 'gated_with_greedy_guard',
                     'seconds': gated['seconds']+greedy['seconds'], 'used': 'gated_network' if gated['objective'] > greedy['objective']+1e-7 else 'greedy'})
    return {'rows': rows, 'variant': variant, 'seed': seed, 'selected_iteration': meta['selected_iteration'],
            'evidence_fingerprint': graph.fingerprint, 'task': task}
