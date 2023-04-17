import os

dir_path = 'submission_results'

output_file = 'track1.txt'
output_path = os.path.join(dir_path, output_file)


result_paths = {
    'S001': './MOTFormat/multicam/preds/aicity/data/S001.txt',
    'S003': './MOTFormat/multicam/preds/aicity/data/S003.txt',
    'S009': './MOTFormat/multicam/preds/aicity/data/S009.txt',
    'S014': './MOTFormat/multicam/preds/aicity/data/S014.txt',
    'S018': './MOTFormat/multicam/preds/aicity/data/S018.txt',
    'S021': './MOTFormat/multicam/preds/aicity/data/S021.txt',
    'S022': './MOTFormat/multicam/preds/aicity/data/S022.txt',
    }


txt_files = [
    result_paths['S001'],
    result_paths['S003'],
    result_paths['S009'],
    result_paths['S014'],
    result_paths['S018'],
    result_paths['S021'],
    result_paths['S022']
    ]

with open(output_path, "w") as outfile:
    for txt_file in txt_files:
        with open(txt_file, "r") as infile:
            for line in infile:
                outfile.write(line)