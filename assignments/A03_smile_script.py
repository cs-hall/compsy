# Load in the most common SMILE states
import A02_ListGen_script as LG  # List Gen
import numpy as np
#from smile.common import *


# enter configuration variables here (including the listgen variables)
## List gen
lg_block = 1
lg_subs = 1
lg_block_params = {
    "pools": ["indoor", "outdoor"],
    "condition_types": ["1p", "massed-rep", "spaced-rep"],
    "rep_types": [2],
    "distance_types": [np.arange(3, 7)],
    "ntrials": 12,
    "test_length": 24,
    "old_prop": 0.5,
    "lure_types": ["lure"],
    "time_blocks": 1,  # not used :(
}
lg_filename_dict = {"indoor": "indoor.csv", "outdoor": "outdoor.csv"}

## Experiment
font_size = 75
resp_keys = ["F", "J"]
ISI_dur = 0.5
ISI_jitter = 0.5


# call the listgen code to create your blocks
# (you can copy it in here from the solution notebook)
final_dict = LG.create_experiment(
    lg_block_params, nBlocks=lg_block, nSubjects=lg_subs, filename_dict=lg_filename_dict
)
subj_0 = final_dict["subj_0"]
s0_block_0 = subj_0["block_0"]
s0b0_study = s0_block_0["study"]
s0b0_test = s0_block_0["test"]


# # create an experiment instance
# exp = Experiment(show_splash=False, resolution=(1024, 768))


# # YOUR CODE HERE TO BUILD THE STATE MACHINE
# # show the stimulus (will default to center of the screen)
# with Loop(s0b0_study) as trial:
#     stim = Label(text=trial.current["image_filename"], font_size=font_size)
#     with UntilDone():
#         kp = KeyPress(keys=resp_keys)

#     Wait(ISI_dur, jitter=ISI_jitter)

#     Log(
#         trial.current,
#         name="flanker",
#         stim_on=stim.appear_time,
#         resp=kp.pressed,
#         resp_time=kp.press_time,
#     )

# # run the experiment
# exp.run()
