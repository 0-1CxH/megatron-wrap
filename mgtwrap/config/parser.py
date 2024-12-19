from confignest import ConfigNest


c = ConfigNest("/gpfs/public/infra/qhu/projects/megatron-wrap/configs/nest", "/gpfs/public/infra/qhu/projects/megatron-wrap/configs/view/debug.yaml")


for key, value in sorted(vars(c.export_flatten_namespace()).items()):
    print(f"{key}.....{value}")
