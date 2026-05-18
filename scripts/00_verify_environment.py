import sys, importlib
for m in ['numpy','pandas','matplotlib']:
    importlib.import_module(m)
print('environment ok:', sys.version.split()[0])
