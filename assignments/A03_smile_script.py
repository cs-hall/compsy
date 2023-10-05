# Load in the most common SMILE states
import A02_ListGen_script as LG  # List Gen
import numpy as np
from smile.common import *
from smile.startup import InputSubject
import random
import csv
from copy import deepcopy
import pickle


# enter configuration variables here (including the listgen variables)
## List gen
# # My list gen params
# lg_block = 1
# lg_subs = 1
# lg_block_params = {
#     "pools": ["indoor", "outdoor"],
#     "condition_types": ["1p", "massed-rep", "spaced-rep"],
#     "rep_types": [2],
#     "distance_types": [np.arange(3, 7)],
#     "ntrials": 6,
#     "test_length": 12,
#     "old_prop": 0.5,
#     "lure_types": ["lure"],
#     "time_blocks": 1,  # not used :(
# }
# lg_filename_dict = {"indoor": "indoor.csv", "outdoor": "outdoor.csv"}

# Per's list gen params
lg_pool_files = {'indoor': 'indoor.csv',
              'outdoor': 'outdoor.csv'}
lg_rep_conds = ['once', 'massed', 'spaced']
lg_loc_conds = ['indoor', 'outdoor']
lg_spaced_range = (4, 9)
lg_num_reps = 1
lg_num_blocks = 2
num_tries = 1000

## Experiment
INST_TEXT = """[u][size=40]SPACED REP INSTRUCTIONS[/size][/u]

In this task, you will see pictures one at at time on the screen. 
Please try to remember these pictures because there will be test after.
    
Press ENTER key to continue."""
INST_FONT_SIZE = 45
INST_STUDY = """[u][size=40]STUDY PHASE[/size][/u]

This is the study phase of the task. The images will advance automatically. 

Please focus on each image and try to commit it to memory
    
Press ENTER key to continue."""
INST_TEST = """[u][size=40]TEST PHASE[/size][/u]

This is the test phase of the task. You will see images and be asked if they are new or old. 
The old images are ones that you have seen on the last study session. 
The new images are ones that you have never seen before. 

Press "F" if the image is OLD
Press "J" if the image is NEW
    
Press ENTER key to continue."""
END_TEXT = """[u][size=40]THANK YOU[/size][/u]

Thanks for participating! 
    
Press ENTER key to close."""
RESP_KEYS = ["F", "J"]
RESP_MAP = {"target": "F", "lure": "J"}
STIM_DUR = 1
STIM_JITTER = 0
STUDY_ISI = 1
STUDY_JITTER = .5
TEST_ISI = 0.5
TEST_JITTER = 0.5
STUDY_TEST_WAIT = 5




# call the listgen code to create your blocks
# (you can copy it in here from the solution notebook)
##### My list gen ######
# final_dict = LG.create_experiment(
#     lg_block_params, nBlocks=lg_block, nSubjects=lg_subs, filename_dict=lg_filename_dict
# )
# blocks = final_dict["subj_0"]

# pers code
# read all the pools into a dictionary
# Code to read in the pools
def read_and_shuffle(pool_file):
    """Read in and shuffle a pool."""
    # create a dictionary reader
    dr = csv.DictReader(open(pool_file, 'r'))

    # read in all the lines into a list of dicts
    pool = [l for l in dr]

    # shuffle it so that the we get new items each time
    random.shuffle(pool)
    
    # report out some pool info
    print(pool_file, len(pool))

    # return the shuffled pool
    return pool
pools = {loc: read_and_shuffle(lg_pool_files[loc])
         for loc in lg_loc_conds}
# create the conds
# fully crossed with all combos of val and rep
conds = []
for loc in lg_loc_conds:
    for rep in lg_rep_conds:
        # I decided to call the repetition condition cond
        conds.append({'loc': loc, 'cond': rep})

# make a function for generating a block
# with a study and test list
def make_block():
    """Generate a block, uses global variables"""
    # loop and create the repeated conditions
    block_conds = []
    for i in range(lg_num_reps):
        # extend the trials with copies of the conditions
        block_conds.extend(deepcopy(conds))

    # try a number of times to satisfy the listgen
    # store temp items so that we can put them
    # back on the pools on failure
    temp_items = {k:[] for k in pools.keys()}
    for i in range(num_tries):
        print(i, end=': ')
        
        # put any temp items back into the pools
        for k in pools.keys():
            if len(temp_items[k]) > 0:
                pools[k].extend(temp_items[k])
                
        # shuffle the conds for that block
        random.shuffle(block_conds)

        # ensure there are enough non-spaced items at the end
        # loop backwards
        num_items = 0
        worked = False
        for c in block_conds[::-1]:
            num_items += 1
            if c['cond'] == 'spaced':
                # make sure we have enough items
                if num_items >= lg_spaced_range[0]:
                    # it worked
                    worked = True

                # break and try again if needed
                break
        if not worked:
            print('x')
            continue

        # we've shuffled our conds, so fill them in with items
        # create the blank study list
        study_list = []
        for cond in block_conds:
            # add a place to fill
            study_list.append(None)
            if cond['cond'] in ['massed', 'spaced']:
                # append another
                study_list.append(None)

        test_list = []
        
        # loop over block conds and 
        # add items to study/test lists
        worked = True   # let's be optimistic this time
        for cond in block_conds:
            # use the valence to grab study and test items
            study_item = pools[cond['loc']].pop()
            test_item = pools[cond['loc']].pop()
            
            # add those items to the temp_items
            temp_items[cond['loc']].extend([study_item, 
                                            test_item])
            
            # update with the cond info
            study_item.update(cond)
            test_item.update(cond)
            
            # add in relevant info for study and test
            study_item['pres_num'] = 1
            study_item['type'] = 'target'
            test_item['type'] = 'lure'
            test_item['pres_num'] = 1   # just so the keys match
            
            # insert the item into the study list
            if cond['cond'] == 'once':
                # just insert in the first open spot
                try:
                    ind = study_list.index(None)
                except ValueError:
                    # no index found, so try again
                    worked = False
                    break
                    
                # use the index to set the item
                study_item['lag'] = 0
                test_item['lag'] = 0
                study_list[ind] = study_item
                print('O', end='')
                
            elif cond['cond'] == 'massed':
                # find the first index with two open spots
                success = False
                for ind in range(len(study_list)-1):
                    if study_list[ind] is None and \
                       study_list[ind+1] is None:
                        # add in the item
                        study_item['lag'] = 1
                        test_item['lag'] = 1
                        study_list[ind] = study_item
                        rep_item = deepcopy(study_item)
                        rep_item['pres_num'] = 2
                        study_list[ind+1] = rep_item
                        success = True
                        print('M', end='')
                        break
                
                # test for failure
                if not success:
                    worked = False
                    break
            else:
                # cond is spaced
                # find the first index with open slots
                # for the second item
                success = False
                for ind in range(len(study_list)-lg_spaced_range[0]):
                    if study_list[ind] is None:
                        # see if we have an open space
                        pos_ind = []
                        for ind2 in range(ind+lg_spaced_range[0], ind+lg_spaced_range[1]):
                            if ind2 < len(study_list) and study_list[ind2] is None:
                                pos_ind.append(ind2)
                        if len(pos_ind) > 0:
                            # pick from the options at random
                            ind2 = random.choice(pos_ind)
                            lag = ind2 - ind
                            
                            # add in the item
                            study_item['lag'] = lag
                            test_item['lag'] = lag
                            study_list[ind] = study_item
                            rep_item = deepcopy(study_item)
                            rep_item['pres_num'] = 2
                            study_list[ind2] = rep_item
                            success = True
                            print('S', end='')
                            break

                # test for failure
                if not success:
                    worked = False
                    break

            # append them to the respective lists
            # study item is added to both study and test
            test_list.append(study_item)
            test_list.append(test_item)
            
        # if it worked, break
        if worked:
            print(' Success!')
            break
        else:
            print('X')
    
    if not worked:
        raise RuntimeError("Unable to generate list.")
        
    # must shuffle the test list
    random.shuffle(test_list)
    
    # make a dictionary to return
    block = {'study': study_list, 'test': test_list}
    
    return block

# generate the proper number of blocks
blocks = []
for b in range(lg_num_blocks):
    blocks.append(make_block())            


# create an experiment instance
exp = Experiment(show_splash=False, resolution=(1024, 768))


# # YOUR CODE HERE TO BUILD THE STATE MACHINE
# show the stimulus (will default to center of the screen)
@Subroutine
def studyTrial(self, block_num, trial_num,trial):
    
    # present stimulus
    stim = Label(text=trial["filename"], font_size=50)
    # wait 
    with UntilDone():
        Wait(STIM_DUR, STIM_JITTER)
    # trial ISI
    Wait(STUDY_ISI, STUDY_JITTER)

    Log(
        log_dict=trial,
        name="old-new-study",
        location=trial['loc'],
        condition=trial['cond'],
        block_num=block_num,
        trial_num=trial_num,
        stim_on=stim.appear_time,
        # TODO: stim offset time?
    )


@Subroutine
def testTrial(self, block_num, trial_num, trial):
    # present the stimulus
    stim = Label(text=trial["filename"], font_size=50)

    with UntilDone():
        # make sure the stimulus has appeared on the screen
        Wait(until=stim.appear_time)

        # collect a response (with no timeout)
        kp = KeyPress(
            keys=RESP_KEYS,
            base_time=stim.appear_time["time"],
            correct_resp=Ref.object(RESP_MAP)[trial["type"]],
        )

    # wait the ISI with jitter
    Wait(Ref.object(TEST_ISI), Ref.object(TEST_JITTER))

    # # TODO: provide feedback to participant?
    # num_correct = Ref.object(num_correct) + kp.correct

    # log the result of the trial
    Log(
        name="old-new-test",
        log_dict=trial,
        block_num=block_num,
        trial_num=trial_num,
        stim_on=stim.appear_time,
        resp=kp.pressed,
        resp_time=kp.press_time,
        rt=kp.rt,
        correct=kp.correct,
    )
    
    

@Subroutine
def studyTestBlock(self, block_num, block_dict):
    # study block
    Label(text=INST_STUDY, font_size=INST_FONT_SIZE,
        text_size=(exp.screen.width*0.75, None),
        markup=True)
    with UntilDone():
        Wait(3)
        KeyPress(keys=['ENTER'])
    with Loop(block_dict['study']) as trial:
        studyTrial(block_num, trial.i, trial.current)
    
    # Interval block
    Wait(STUDY_TEST_WAIT)
    
    # test block
    Label(text=INST_TEST, font_size=INST_FONT_SIZE,
        text_size=(exp.screen.width*0.75, None),
        markup=True)
    with UntilDone():
        Wait(3)
        KeyPress(keys=['ENTER'])
    with Loop(block_dict['test']) as trial:
        testTrial(block_num, trial.i, trial.current)

    # TODO: provide feedback on performance?



# InputSubject("old-new")

Label(text=INST_TEXT, font_size=INST_FONT_SIZE,
        text_size=(exp.screen.width*0.75, None),
        markup=True)
with UntilDone():
    Wait(3)
    KeyPress(keys=['ENTER'])

with Loop(blocks) as block_dict:
    studyTestBlock(block_dict.i, block_dict.current)

Label(text=END_TEXT, font_size=INST_FONT_SIZE,
        text_size=(exp.screen.width*0.75, None),
        markup=True)
with UntilDone():
    Wait(3)
    KeyPress(keys=['ENTER'])


# run the experiment
exp.run()
