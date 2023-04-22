import os


dir_path = 'results'
output_file = 'track1_submission.txt'
output_path = os.path.join(dir_path, output_file)

result_paths = [
    './results/S001.txt',
    './results/S003.txt',
    './results/S009.txt',
    './results/S014.txt',
    './results/S018.txt',
    './results/S021.txt',
    './results/S022.txt',
    ]

global_id_dict = {}
max_global_id = 1

with open(output_path, "w") as outfile:
    for result_path in result_paths:
        with open(result_path, "r") as f:
            lines = f.readlines()
            lines = [line.replace('\n','').split(' ') for line in lines]
        for line in lines:
            global_id = line[1]
            if global_id not in global_id_dict:
                global_id_dict[global_id] = max_global_id
                line[1] = max_global_id
                max_global_id += 1
            else:
                line[1] = global_id_dict[global_id]

            out_line = " ".join([str(l) for l in line]) + '\n'
            outfile.write(out_line)
        
        global_id_dict = {}