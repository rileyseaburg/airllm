"""
AirLLM support for GLM-4 and GLM-4 MoE models (including GLM-4.7)

GLM-4.7 is a 358B parameter MoE model from Z.ai with:
- 73.8% on SWE-bench Verified
- Interleaved Thinking capability
- MIT License
"""

from .airllm_base import AirLLMBaseModel


class AirLLMGLM4(AirLLMBaseModel):
    """
    AirLLM implementation for GLM-4 models.
    
    Supports:
    - GLM-4 base models
    - GLM-4 MoE models (like GLM-4.7)
    """
    
    def set_layer_names_dict(self):
        """Set layer names for GLM-4 architecture."""
        self.layer_names_dict = {
            'embed': 'transformer.embedding.word_embeddings',
            'layer_prefix': 'transformer.encoder.layers',
            'norm': 'transformer.encoder.final_layernorm',
            'lm_head': 'transformer.output_layer',
            'rotary_pos_emb': 'transformer.rotary_pos_emb',
        }
    
    def get_use_better_transformer(self):
        """GLM-4 uses its own attention implementation."""
        return False
    
    def get_past_key_values_cache_seq_len(self, past_key_values):
        """Get sequence length from KV cache."""
        return past_key_values[0][0].shape[0]
    
    def get_sequence_len(self, seq):
        """Get sequence length."""
        return seq.shape[0]
    
    def get_pos_emb_args(self, len_p, len_s):
        """Get position embedding arguments for GLM-4."""
        return {'rotary_pos_emb': self.model.transformer.rotary_pos_emb(self.max_seq_len)[len_p:len_p + len_s]}
    
    def get_past_key_value_args(self, k_cache, v_cache):
        """Get past key value arguments."""
        return {'kv_cache': (k_cache, v_cache)}
    
    def get_attention_mask_args(self, full_attention_mask, len_p, len_s):
        """Get attention mask arguments."""
        return {'attention_mask': full_attention_mask[-len_s:, -len_p - len_s:][None, None, ...]}
    
    def get_position_ids_args(self, full_position_ids, len_p, len_s):
        """Get position IDs arguments."""
        return {}
    
    def run_norm(self, layer, seq):
        """Run final layer norm."""
        return layer(seq)
    
    def run_lm_head(self, layer, seq):
        """Run output layer."""
        return layer(seq).float()


class AirLLMGLM4MoE(AirLLMGLM4):
    """
    AirLLM implementation for GLM-4 MoE models like GLM-4.7.
    
    GLM-4.7 features:
    - 358B total parameters
    - MoE architecture with expert routing
    - Interleaved Thinking
    - 256K context support
    """
    
    def set_layer_names_dict(self):
        """Set layer names for GLM-4 MoE architecture."""
        # GLM-4 MoE may have slightly different naming
        self.layer_names_dict = {
            'embed': 'transformer.embedding.word_embeddings',
            'layer_prefix': 'transformer.encoder.layers',
            'norm': 'transformer.encoder.final_layernorm', 
            'lm_head': 'transformer.output_layer',
            'rotary_pos_emb': 'transformer.rotary_pos_emb',
        }
