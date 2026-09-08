import json
from unittest.mock import MagicMock, patch

import pytest

from examples.inference_chain import load_chain, verify_package


def test_dependent_adapters_merge_before_rl_is_attached(tmp_path):
    manifest = {'base_model': 'base', 'base_revision': 'frozen', 'stages': {'rl': {'merge_before': ['cpt','sft']}}}
    base, cpt_merged, sft_merged, final = [MagicMock() for _ in range(4)]
    cpt, sft = MagicMock(), MagicMock()
    cpt.merge_and_unload.return_value = cpt_merged
    sft.merge_and_unload.return_value = sft_merged
    final.to.return_value.eval.return_value = final
    with patch('examples.inference_chain.verify_package', return_value=manifest), \
         patch('transformers.Qwen3_5ForConditionalGeneration.from_pretrained', return_value=base), \
         patch('transformers.AutoProcessor.from_pretrained'), \
         patch('peft.PeftModel.from_pretrained', side_effect=[cpt,sft,final]) as attach:
        model, _ = load_chain(tmp_path, 'rl')
    assert model is final
    assert [(c.args[0], c.args[1]) for c in attach.call_args_list] == [
        (base,tmp_path/'cpt'), (cpt_merged,tmp_path/'sft'), (sft_merged,tmp_path/'rl')]
    cpt.merge_and_unload.assert_called_once_with(safe_merge=True)
    sft.merge_and_unload.assert_called_once_with(safe_merge=True)
    final.merge_and_unload.assert_not_called()


def test_corrupt_adapter_is_rejected_before_loading(tmp_path):
    (tmp_path/'rl').mkdir()
    (tmp_path/'rl/adapter_model.safetensors').write_bytes(b'corrupt')
    manifest={'stages':{'rl':{'files':{'adapter_model.safetensors':'0'*64}}}}
    (tmp_path/'release_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Checksum mismatch'):
        verify_package(tmp_path)
