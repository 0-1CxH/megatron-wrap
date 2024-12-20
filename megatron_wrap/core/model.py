from megatron_wrap.utils import logger

class MegatronModelProviderEntry:

    @classmethod
    def get_provider(cls, model_provider_args, megatron_lm_args):
        model_type_to_provider_map = {
            "GPT": cls.gpt_model_provider,
        }
        return model_type_to_provider_map.get(model_provider_args.model_type)(model_provider_args, megatron_lm_args)

    
    @classmethod
    def gpt_model_provider(cls, model_provider_args, megatron_lm_args):
        import megatron
        from megatron.training.arguments import core_transformer_config_from_args
        from megatron.core.transformer.spec_utils import import_module
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_local_spec,
            get_gpt_layer_with_transformer_engine_spec,
        )
        from megatron.core.models.gpt import GPTModel

        args = megatron_lm_args

        def _model_(pre_process, post_process):
            use_te = args.transformer_impl == "transformer_engine"
            config = core_transformer_config_from_args(args)
            logger.info_rank_0(f'[STATUS] building model')
            logger.debug_rank_0(f"use transformer engine: {use_te}, model provider args: {model_provider_args}")

            if args.use_legacy_models:
                logger.warning_rank_0(f"legacy model is deprecated")
                model = megatron.legacy.model.GPTModel(
                    config,
                    num_tokentypes=0,
                    parallel_output=model_provider_args.parallel_output,
                    pre_process=pre_process,
                    post_process=post_process,
                )
            else: # using mcore models
                if args.spec is not None:
                    transformer_layer_spec = import_module(args.spec)
                else:
                    if use_te:
                        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
                    else:
                        transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)

                model = GPTModel(
                    config=config,
                    transformer_layer_spec=transformer_layer_spec,
                    vocab_size=args.padded_vocab_size,
                    max_sequence_length=args.max_position_embeddings,
                    pre_process=pre_process,
                    post_process=post_process,
                    fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                    parallel_output=model_provider_args.parallel_output,
                    share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                    position_embedding_type=args.position_embedding_type,
                    rotary_percent=args.rotary_percent,
                    rotary_base=args.rotary_base
                )
            return model

        return _model_