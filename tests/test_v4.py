import itertools
import json

import numpy as np
import pytest
import torch

from dmease.evidence_policy import EvidenceGraph
from dmease.planning_v4 import PlanningEnvironment, PlanningPolicy, PlannerPPO, rollout, training_states
from dmease.structured_extraction import chunks, output_schema, validate, character_positions


def test_chunks_preserve_original_and_offsets():
    source = '口干，患者否认心悸。' * 61
    pieces = chunks(source, 37)
    assert ''.join(part for _, part in pieces) == source
    assert all(source[offset:offset+len(part)] == part and len(part) <= 37 for offset, part in pieces)


def test_strict_grounding_and_schema():
    raw = json.dumps({'items': [{'text': '口干', 'present': True}, {'text': '心悸', 'present': 'false'},
                               {'text': '发热', 'present': True}, {'text': '口干', 'present': True}]})
    result = validate(raw, '口干，无心悸')
    assert len(result['items']) == 1
    assert result['rejected'] == {'item_schema': 1, 'ungrounded': 1, 'duplicate': 1}
    assert validate('{"items": [', '口干')['invalid_json']
    assert validate('{"items": [], "extra": 1}', '口干')['invalid_schema']
    assert character_positions('口干、口干', ['口干']) == {0, 1, 3, 4}


def test_chinese_grammar_accepts_only_source_terms():
    lm = pytest.importorskip('lmformatenforcer')
    source = '口干，无心悸'
    def accepts(raw):
        parser = lm.JsonSchemaParser(output_schema(source), config=lm.CharacterLevelParserConfig(
            alphabet=lm.CharacterLevelParserConfig().alphabet + source, force_json_field_order=True))
        for char in raw:
            if char not in parser.get_allowed_characters():
                return False
            parser = parser.add_character(char)
        return parser.can_end()
    assert accepts('{"items":[{"text":"口干","present":true}]}')
    assert not accepts('{"items":[{"text":"发热","present":true}]}')
    assert not accepts('{"items":[{"text":"口干"}]}')


def task(**kw):
    return {'symptoms': {'口渴': 1., '乏力': 1.}, 'budget': 3, **kw}


def test_exact_table_matches_independent_feasible_enumeration(evidence_path):
    graph = EvidenceGraph(evidence_path)
    env = PlanningEnvironment(graph, task(budget=2, forbidden_pairs=[['黄芪', '山药']]))
    masks, scores = env.exact_table()
    expected = {}
    for bits in itertools.product((0, 1), repeat=graph.action_dim):
        if sum(bits) > 2 or (bits[graph.hnames['黄芪']] and bits[graph.hnames['山药']]):
            continue
        mask = sum(bit << i for i, bit in enumerate(bits))
        expected[mask] = env.score(mask)[0]
    assert set(masks) == set(expected)
    assert np.allclose(scores, [expected[int(m)] for m in masks], atol=1e-7)
    rows, _ = training_states(env)
    assert abs(rows[0][4] - max(expected.values())) < 1e-6


def test_full_constraint_state_stop_and_telescoping(evidence_path):
    graph = EvidenceGraph(evidence_path)
    env = PlanningEnvironment(graph, task(forbidden_pairs=[['黄芪', '山药']]))
    other = PlanningEnvironment(graph, task())
    assert torch.equal(env.state()[2], other.state()[2])
    assert not torch.equal(env.state()[0], other.state()[0])
    assert not torch.equal(other.state()[0], PlanningEnvironment(graph, task(budget=2)).state()[0])
    reward = env.step(graph.hnames['黄芪'])
    assert not env.mask()[graph.hnames['山药']]
    with pytest.raises(ValueError):
        env.step(graph.hnames['山药'])
    reward += env.step(env.stop)
    assert env.done and abs(reward - env.score(env.selected)[0]) < 1e-7


@pytest.mark.parametrize('variant', ['residual_kan', 'residual_mlp'])
def test_initial_residual_prior_and_finite_ppo(evidence_path, variant):
    graph = EvidenceGraph(evidence_path)
    env = PlanningEnvironment(graph, task(forbidden=['麦冬']))
    policy = PlanningPolicy(env.observation_dim, env.feature_dim, variant)
    assert rollout(env, policy) == rollout(env)
    trainer = PlannerPPO(policy)
    batch = trainer.collect([env], np.random.default_rng(42), episodes=3)
    assert all(int(a) != graph.hnames['麦冬'] for a in batch[3])
    assert all(np.isfinite(value) for value in trainer.update(batch).values())


def test_document_components_keep_transitive_links(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from affinity_v4 import grouping
    rows = [dict(smiles='A', scaffold='S1', document_ids=['D1']),
            dict(smiles='B', scaffold='S2', document_ids=['D1', 'D2']),
            dict(smiles='C', scaffold='S3', document_ids=['D2']),
            dict(smiles='D', scaffold='S3', document_ids=['D3'])]
    docs = grouping(rows, False)
    assert docs[0] == docs[1] == docs[2] != docs[3]
    assert len(set(grouping(rows, True))) == 1


def test_expert_syndrome_choices_preserve_multi_label_gold(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from diagnosis_v4 import choice_labels
    assert choice_labels('A:气虚证; B:阴虚证;C:血瘀证', 'A;C') == {'气虚', '血瘀'}
    with pytest.raises(ValueError):
        choice_labels('A:气虚证', 'B')


def test_conservative_override_can_retain_greedy(evidence_path):
    from dmease.conservative_planner import gated_rollout
    graph = EvidenceGraph(evidence_path)
    env = PlanningEnvironment(graph, task(forbidden=['麦冬']))
    class StopScorer:
        def actor(self, obs, features):
            return features[:, :, 8] * .01
    greedy_mask, greedy_value = rollout(env)
    chosen, value, _ = gated_rollout(env, StopScorer(), .02)
    assert chosen == greedy_mask and value == greedy_value
    chosen, value, stats = gated_rollout(env, StopScorer(), 0.)
    assert chosen == 0 and value == 0 and stats['overrides'] == 1


def test_repeated_quote_and_polarity_are_explicitly_ambiguous():
    from dmease.v4_runtime import annotate_quotes
    source = '昨日口渴，今日无口渴。'
    items = [{'text': '口渴', 'present': value, 'segment_start': 0, 'segment_end': len(source)} for value in (True, False)]
    annotated = annotate_quotes(items, source)
    assert all(len(r['occurrences']) == 2 and len(r['review_notes']) == 2 for r in annotated)
    assert all(source[m['start']:m['end']] == '口渴' for r in annotated for m in r['occurrences'])


def test_paired_f1_identical_predictions_have_zero_difference(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from extraction_statistics_v4 import compare
    rows = [{'row_hash': 'one', 'surface': {'tp': 3, 'fp': 1, 'fn': 2}},
            {'row_hash': 'two', 'surface': {'tp': 0, 'fp': 0, 'fn': 0}}]
    result = compare(rows, rows, repeats=100)
    assert result['micro_f1_difference'] == 0 and result['case_bootstrap_95ci'] == [0., 0.]
    assert result['paired_randomization_p'] == 1
