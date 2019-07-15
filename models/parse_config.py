def parse_model_cfg(path):
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [line for line in lines if line and not line.startswith('#')]
    model_defs = []
    
    for line in lines:
        if line.startswith('['):
            model_defs.append({})
            model_defs[-1]['type'] = line[1:-1].strip()
        else:
            key, value = line.split('=')
            model_defs[-1][key.strip()] = value.strip()
    
    return model_defs