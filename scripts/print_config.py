import yaml, pprint
cfg = yaml.safe_load(open("configs/model.yaml"))
print("=== configs/model.yaml ===")
pprint.pprint(cfg)
