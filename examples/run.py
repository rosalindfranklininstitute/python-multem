import multem

input_multislice = multem.Input()
config = multem.SystemConfiguration()

# print(multem.is_gpu_available())
# if not multem.is_gpu_available():
config.device = "host"
result = multem.simulate(config, input_multislice)

# print(result)
