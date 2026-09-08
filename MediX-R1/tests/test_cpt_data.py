import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.cpt_data import pack_excerpts, sample_spans


def excerpt(ids, document_id='doc', start=0):
    return dict(input_ids=ids + [999], document_id=document_id, source='test', bucket='test',
                source_token_start=start, source_token_end=start + len(ids), is_document_end=True)


class CptDataTests(unittest.TestCase):
    def test_short_document_is_retained_whole(self):
        self.assertEqual(sample_spans(list(range(4095)), 'doc', 4096, 2048, 42), [(0, 4095)])

    def test_long_document_sampling_is_repeatable_disjoint_and_within_budget(self):
        ids = list(range(20000))
        spans = sample_spans(ids, 'doc', 4096, 2048, 42)
        self.assertEqual(spans, sample_spans(ids, 'doc', 4096, 2048, 42))
        self.assertEqual(sum(end - start + 1 for start, end in spans), 4096)
        self.assertLessEqual(spans[0][1], 10000)
        self.assertGreaterEqual(spans[1][0], 10000)
        self.assertLessEqual(spans[-1][1], len(ids))
        self.assertNotEqual(spans, sample_spans(ids, 'other-doc', 4096, 2048, 42))

    def test_pack_preserves_every_token_and_eos_across_documents(self):
        rows = [excerpt([1, 2], 'a'), excerpt([3, 4, 5, 6], 'b', 50)]
        packed = pack_excerpts(rows, 4)
        self.assertEqual([b['input_ids'] for b in packed], [[1, 2, 999, 3], [4, 5, 6, 999]])
        self.assertEqual(packed[0]['spans'][1]['source_token_start'], 50)
        self.assertEqual(packed[1]['spans'][0]['source_token_start'], 51)
        self.assertEqual(packed[1]['spans'][0]['source_token_end'], 54)
        self.assertTrue(packed[1]['spans'][0]['ends_with_eos'])

    def test_short_final_tail_is_kept_even_when_it_is_only_eos(self):
        for length in range(1, 40):
            rows = [excerpt(list(range(length)))]
            packed = pack_excerpts(rows, 8)
            self.assertEqual([x for b in packed for x in b['input_ids']], rows[0]['input_ids'])
            self.assertTrue(all(2 <= len(b['input_ids']) <= 8 for b in packed))
            for block in packed:
                self.assertEqual(sum(s['block_end'] - s['block_start'] for s in block['spans']), len(block['input_ids']))

    def test_provenance_reconstructs_full_packed_stream(self):
        documents = {'a': list(range(30)), 'b': list(range(100, 120))}
        rows = [excerpt(documents['a'][3:19], 'a', 3), excerpt(documents['b'][4:18], 'b', 4)]
        for block in pack_excerpts(rows, 8):
            recovered = []
            for span in block['spans']:
                recovered.extend(documents[span['document_id']][span['source_token_start']:span['source_token_end']])
                if span['ends_with_eos']:
                    recovered.append(999)
            self.assertEqual(recovered, block['input_ids'])

    def test_real_qwen_eos_is_supervised_while_equal_id_padding_is_masked(self):
        if not (Path(__file__).resolve().parents[1]/'models/Qwen3.5-2B/tokenizer.json').exists():
            self.skipTest('Requires the locally downloaded Qwen tokenizer.')
        from transformers import AutoTokenizer
        from training.cpt import ProseCollator
        tokenizer = AutoTokenizer.from_pretrained(Path(__file__).resolve().parents[1]/'models/Qwen3.5-2B', local_files_only=True)
        eos = tokenizer.convert_tokens_to_ids('<|endoftext|>')
        self.assertEqual(tokenizer.pad_token_id, eos)
        batch = ProseCollator(tokenizer)([dict(input_ids=[42, eos, 43, eos]), dict(input_ids=[44, eos])])
        self.assertEqual(batch['labels'][0, 1].item(), eos)
        self.assertEqual(batch['labels'][1, 1].item(), eos)
        self.assertTrue((batch['labels'][batch['attention_mask'] == 0] == -100).all())

    def test_packed_2048_tokens_support_qwen_hybrid_forward_and_backward_on_cpu(self):
        import torch
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        torch.set_num_threads(2)
        config = Qwen3_5Config(
            text_config=dict(vocab_size=1024, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                             linear_key_head_dim=16, linear_value_head_dim=16,
                             linear_num_key_heads=2, linear_num_value_heads=2,
                             layer_types=['linear_attention', 'full_attention'], max_position_embeddings=2048,
                             rope_parameters=dict(rope_type='default', rope_theta=10000.,
                                                  partial_rotary_factor=0.5, mrope_section=[1, 1, 2])),
            vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                               out_hidden_size=32, num_position_embeddings=16), tie_word_embeddings=True)
        model = Qwen3_5ForConditionalGeneration(config)
        model.config.use_cache = False
        packed = pack_excerpts([excerpt([42] * 999, 'a'), excerpt([43] * 1047, 'b')], 2048)
        ids = torch.tensor([packed[0]['input_ids']])
        result = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone())
        self.assertTrue(torch.isfinite(result.loss))
        result.loss.backward()
        layers = model.model.language_model.layers
        self.assertIsNotNone(layers[0].linear_attn.in_proj_qkv.weight.grad)
        self.assertIsNotNone(layers[1].self_attn.q_proj.weight.grad)


if __name__ == '__main__':
    unittest.main()
