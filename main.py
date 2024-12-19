from megatron_wrap.config import MegatronWrapConfig

c = MegatronWrapConfig("/gpfs/public/infra/qhu/projects/megatron-wrap/configs/debug.yaml")
print(c.format_megatron_lm_args())
print(c.format_megatron_wrap_args())
print(c.format_all_args())