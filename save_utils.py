import os
from pathlib import Path
import pickle

def pickle_load(file_name):

    with open(file_name, 'rb') as file:
        l = pickle.load(file)
        file.close()
    return l

def save_progress(folder, episode, trial_index, dict_):
    #print(folder, episode, trial_index)
    results_folder_index = os.path.join(folder, trial_index)
    #rint(results_folder_index)

    Path(results_folder_index).mkdir(parents=True, exist_ok=True)
    #print(dict_)
    for value_name in dict_.keys():
        #print(value_name)

        vals = dict_[value_name] 
        # if value_name == 'pis':
        #     vals = vals[-1]
        file_name = f'{episode}_{value_name}.pkl'
        new_name = os.path.join(results_folder_index, file_name)



        if len(os.listdir(results_folder_index)) > 0:
            #print('hello')
            for f in os.listdir(results_folder_index):
                #print('f')
                if value_name in f:
                    #print('value name in f')
                    old_name = os.path.join(results_folder_index, f)
                    
                    os.rename(old_name, new_name)
                    break
        #print('about to write')
        with open(new_name, 'wb') as fname:
            #print('set fname', new_name)
            pickle.dump(vals, fname) 
            #print('dumped')
            fname.close()  
            #print('fname closed')
    # names = list(dict_.keys())
    # names.sort()

    #print('titles', titles)
    # for vals, title in zip(values, titles):

    #     for f in os.listdir(results_folder_index):
    #         if title in f


    #     #print(title)
    #     with open(os.path.join(results_folder_index, f'{title}.pkl'), 'wb') as f:
    #         #print('saving', title)
    #         pickle.dump(vals, f) 
    #         #print('dumped', title)
    #         f.close()  
    #         #print('closed file')   

def find_latest_episode(folder):

    episode_list = os.listdir(folder)

    episode_ints = [int(e.split('_')[0]) for e in episode_list]

    return max(episode_ints)