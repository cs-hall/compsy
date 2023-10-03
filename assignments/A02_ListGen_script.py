import csv  # for reading in files
import logging  # practice doing good stuff
import random  # for shuffling lists
from copy import deepcopy  # for fixing terrible bugs

import numpy as np  # for classic array stuff
import pandas as pd  # for tables

logging_level = logging.WARNING
logging.basicConfig(
    format="%(levelname)s (%(asctime)s): %(message)s (Line: %(lineno)d [%(filename)s])",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging_level,
)

"""
Example call
block_params = {
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

filename_dict = {"indoor": "indoor.csv", "outdoor": "outdoor.csv"}

final_dict = create_experiment(
    block_params, nBlocks=2, nSubjects=3, filename_dict=filename_dict
)
"""


def test_study_list(study_list: list[dict], reps: int) -> set:
    """
    Test the study list for counterbalancing

    Parameters
    ----------
    study_list: A list of dictionaries for the study session
    reps: The number of repetitions for both massed and spaced items

    Returns
    -------
    unique_images: a set of images from the study
    """
    study_indoor_count = 0
    study_outdoor_count = 0
    once_p_count = 0
    massed_count = 0
    spaced_count = 0
    unique_images = set()
    for trial in study_list:
        unique_images.add(trial["image_filename"])
        # Count the types
        if trial["type"] == "1p":
            once_p_count += 1
        elif trial["type"] == "massed-rep":
            massed_count += 1
        else:
            spaced_count += 1
        # count the image pools
        if trial["pool"] == "indoor":
            study_indoor_count += 1
        elif trial["pool"] == "outdoor":
            study_outdoor_count += 1

    assert (
        study_indoor_count == study_outdoor_count
    ), f"FAILED: Study indoor outdoor not balanced; in={study_indoor_count}, out={study_outdoor_count}"
    print(
        f"PASSED: Study indoor outdoor balanced; in={study_indoor_count}, out={study_outdoor_count}"
    )

    assert (
        once_p_count == massed_count / reps == spaced_count / reps
    ), f"FAILED: Study conditions not balanced; 1p={once_p_count}, massed={massed_count}, spaced={spaced_count}"
    print(
        f"PASSED: Study conditions balanced reps={reps}; 1p={once_p_count}, massed={massed_count}, spaced={spaced_count}"
    )

    assert len(unique_images) == once_p_count + (massed_count / reps) + (
        spaced_count / reps
    ), f"FAILED: images not equal to trials"
    print(f"PASSED: images equal to trials")

    return unique_images


def test_test_list(test_list: list[dict], old_prop: float) -> set:
    """
    Test the test list for counterbalancing

    Parameters
    ----------
    test_list: A list of dictionaries for the test session
    old_prop: The proportion of old items in the test list

    Returns
    -------
    unique_images: a set of images from the test session
    """

    test_indoor_count = 0
    test_outdoor_count = 0
    lure_count = 0
    nonlure_count = 0
    lure_in = 0
    lure_out = 0
    once_p_count = 0
    massed_count = 0
    spaced_count = 0
    targ_in = 0
    targ_out = 0
    unique_images = set()
    for trial in test_list:
        unique_images.add(trial["image_filename"])
        # Count the types
        if trial["type"] == "lure":
            lure_count += 1
            # count the image pool
            if trial["pool"] == "indoor":
                lure_in += 1
            else:
                lure_out += 1
        else:
            nonlure_count += 1
            # count the type
            if trial["type"] == "1p":
                once_p_count += 1
            elif trial["type"] == "massed-rep":
                massed_count += 1
            else:
                spaced_count += 1
            # count the image pool
            if trial["pool"] == "indoor":
                targ_in += 1
            else:
                targ_out += 1
        # count the image pool
        if trial["pool"] == "indoor":
            test_indoor_count += 1
        elif trial["pool"] == "outdoor":
            test_outdoor_count += 1

    assert (
        test_indoor_count == test_outdoor_count
    ), f"FAILED: Test indoor outdoor not balanced ({test_indoor_count}=={test_outdoor_count})"
    print(
        f"PASSED: indoor test==outdoor test ({test_indoor_count}=={test_outdoor_count})"
    )

    assert lure_in == lure_out, "FAILED: Test lure indoor outdoor not balanced"
    print(f"PASSED: lure pool balancing({lure_in}=={lure_out})")

    assert targ_in == targ_out, "FAILED: Test target indoor outdoor not balanced"
    print(f"PASSED: target pool balancing({targ_in}=={targ_out})")

    assert (
        nonlure_count / (nonlure_count + lure_count) == old_prop
    ), f"FAILED: Proportion of target and lures wrong; Actual proportion: {nonlure_count / (nonlure_count+lure_count)} target proportion: {old_prop}"
    print(
        f"PASSED: Proportion of target and lures ({lure_count*(1-old_prop)}={nonlure_count*old_prop})"
    )

    assert (
        once_p_count == massed_count == spaced_count
    ), f"FAILED: Test target conditions not balanced; 1p={once_p_count}, massed={massed_count}, spaced={spaced_count}"
    print(
        f"PASSED: Test target conditions balanced; 1p={once_p_count}, massed={massed_count}, spaced={spaced_count}"
    )

    assert len(unique_images) == len(test_list), f"FAILED: images not equal to trials"
    print(f"PASSED: images equal to trials")

    return unique_images


def test_subj(subj_blocks: dict[str, dict], reps: int, old_prop: float) -> bool:
    """
    Tests the subject for correct counterbalancing and no duplicate images

    Parameters
    ----------
    subj_blocks: a dictionary of all the blocks for this subject
    reps: an int describing the number of repetitions of both massed and spaced items
    old_prop: a float describing the ratio of old items in the test session

    Returns
    -------
    True: only returned if passed all tests
    """
    all_items = set()
    for block_name, block_dict in subj_blocks.items():
        # test this block
        study_items = test_study_list(block_dict["study"], reps)
        test_items = test_test_list(block_dict["test"], old_prop)
        # get all image names from this block
        this_block = study_items.union(test_items)
        # find any repeats from previous blocks
        all_blocks_intersection = all_items.intersection(this_block)

        assert (
            len(all_blocks_intersection) == 0
        ), f"FAILED: {block_name}: Some items repeated, number repeated = {len(all_blocks_intersection)}"
        print(f"PASSED: No items repeated through {block_name}")

        # add this blocks images to all images
        all_items = all_items.union(this_block)

    return True


def create_conditions(
    pools: list,
    condition_types: list,
    rep_types: list,
    distance_types: list,
    verbose=False,
) -> list[dict]:
    """
    Make the set of possible conditions
    Output should be list of dictionaries like the examples below
    {"pool":"indoor", "type":"1p", "reps":0, "distances":[None], "placement":[0]}
    {"pool":"indoor", "type":"massed-rep", "reps":3, "distances":[1,1], "placement":[0,1,2]}
    {"pool":"indoor", "type":"spaced-rep", "reps":3, "distances":[3,3], "placement":[0,3,6]}
    ------
    INPUTS
        pools = ["indoor", "outdoor"] # for the stimulus types
        condition_types = ["1p", "massed-rep", "spaced-rep"] # for the type of presentation
        rep_types = [2] # for the number of repetitions
        distance_types = [np.arange(3,7),]  # for the lists to randomly pull the distance from
    OUTPUT
        conditions = a list of dictionaries
    """

    conditions = []
    for pool in pools:
        for condition_type in condition_types:
            if condition_type == "1p":
                conditions.append(
                    {
                        "pool": pool,
                        "type": condition_type,
                        "reps": 1,
                        "distances": None,
                        "placement": np.array([0]),
                    }
                )
            else:
                for reps in rep_types:
                    placements = np.arange(reps)
                    if condition_type == "massed-rep":
                        distances = np.diff(placements)
                        conditions.append(
                            {
                                "pool": pool,
                                "type": condition_type,
                                "reps": reps,
                                "distances": distances,
                                "placement": placements,
                            }
                        )

                    elif condition_type == "spaced-rep":
                        # If a list then it picks a random one later on.
                        for dist in distance_types:
                            conditions.append(
                                {
                                    "pool": pool,
                                    "type": condition_type,
                                    "reps": reps,
                                    "distances": dist.copy(),
                                    "placement": placements,
                                }
                            )

    if verbose:
        print(f"There are {len(conditions)} conditions")
        if verbose == "VERY":
            for c in conditions:
                print(c)
    return conditions


def create_trial_set(ntrials: int, conditions: list[dict], verbose=False) -> list[dict]:
    """
    Create the list of all the unique trials for this study session.
    Note: this is note shuffled
    ------
    INPUTS
        ntrials: number of trials for the study session
        conditions: list; list of dictionaries
    OUTPUT
        trial_set: list of dictionaries with all the metadata
    """
    # Chek the inputs
    assert (
        type(ntrials) == int
    ), f"ntrials needs to be an interger; current type:{type(ntrials)}"
    assert (
        len(conditions) >= 1
    ), f"conditions needs to be at least one long; current length:{len(conditions)}"
    for i, cond in enumerate(conditions):
        assert (
            type(cond) == dict
        ), f"conditions must contain only dictionaries; item {i} of conditions is type:{type(cond)}"

    # Create set of trials
    if ntrials % len(conditions) != 0:
        logging.warning("Number of trials will result in imperfect condition balancing")
    condition_reps = int(np.ceil(ntrials / len(conditions)))

    # initialize the trial set
    trial_set = []
    logging.debug(f"Conditions are repeated {condition_reps} times")
    for i in range(condition_reps):
        for condition in conditions:
            trial_set.append(condition.copy())
    trial_set = trial_set[:ntrials]  # this trims off the extra if needed

    # add id and spacing information
    for i, trial in enumerate(trial_set):
        # add id to trial
        trial["id"] = i

        # add spacing information
        if trial["type"] == "spaced-rep":
            dist_choice = random.choice(trial["distances"])
            logging.debug(dist_choice)

            # using copy to prevent edits later on
            trial["placement"] = dist_choice * trial["placement"].copy()
            trial["distances"] = np.diff(trial["placement"])

    if verbose:
        print(
            f"There are {len(trial_set)} study trials (not counting massed/spaced reps)"
        )
        if verbose == "VERY":
            for trial in trial_set:
                print(trial)

    return trial_set


def check_placement(working_list: list, trial: dict, placement: int) -> bool:
    """
    Check if a specific trial can be placed at a given location in a list
    ------
    INPUTS
        working_list: a list with None in all empty/available slots
        trial: a dictionary with a relative array of repetitions
        placement: an index for where
    OUTPUTS
        a boolean True or False
    """
    # The only reason this is a separate function is so that I can return out of for loops
    locations = trial["placement"] + placement

    for location in locations:
        try:
            if not working_list[location] == None:
                # if there is no empty spot at this location, then this trial doesn't work here
                return False
        except IndexError:
            # if this location doesn't exist then this trial doesn't work
            logging.debug("IndexError prevented in checking placement")
            return False
    # only if all locations fit is it true
    return True


def find_placement(working_list: list, trial: dict) -> list[int]:
    """
    Finds all the indices of 'working_list' where 'trial' could be placed
    ------
    INPUTS
        working_list: a list with None in all empty/available slots
        trial: a dictionary with a relative array of repetitions
    OUTPUTS
        good_inds: a list of indices where this trial can be placed in working list
    """
    # gets all the good indices
    proposal_indices = [
        index
        for index in range(len(working_list))
        if check_placement(working_list, trial, index)
    ]

    logging.debug(f"proposed_indices: {proposal_indices}")
    return proposal_indices


def place_trial_in_list(working_list: list, trial: dict, proposal: int) -> list:
    """
    Place a trial into the working list at the given location
    ------
    INPUTS
        working_list: a list with None in all empty/available slots, and trials in the other
        trial: a dictionary with a relative array of repetitions
        proposal: an index of working list to place the first repetition of the trial
    OUTPUTS
        working_list: same as input but now with the trial and its possible repetitions added
    """
    placed_trial = trial.copy()
    placed_trial["placement"] = placed_trial["placement"] + proposal
    logging.debug(f"trial placements: {placed_trial['placement']}")
    for i, placement in enumerate(placed_trial["placement"]):
        working_list[placement] = placed_trial
        working_list[placement]["repetition"] = i
        working_list[placement]["location"] = placement

    return working_list


def fit_trials_in_list(working_list: list, trial_set: list, level: int = 0) -> list:
    """
    Takes a working_list and fit all the trials in trial_set into it.
    This works by calling itself after each proposal until all the trials have been used up.
    ------
    INPUTS
        working_list: a list with None in all empty/available slots, and trials in the other
        trial_set: a list of trials (dictionaries) that need to be placed in the working list
        level: an int describing the level/depth of recursions (used for debugging)
    OUTPUTs
        temp_list: a solution to fitting the trial_set in the working_list or empty list
    """

    logging.debug(f"starting loops with {len(trial_set)} trials")
    if len(trial_set) < 1:
        return working_list
    logging.debug(f"Trying this trial: {trial_set[0]}")
    # gone through in order of trial set so we can optimize which trials are hardest to fit first
    proposal_inds = find_placement(working_list, trial_set[0])
    logging.debug(f"starting loops with {len(proposal_inds)} proposals")

    if len(proposal_inds) > 1:
        random.shuffle(proposal_inds)  # don't want to try the proposals in order

        temp_list = deepcopy(working_list)
        for proposal in proposal_inds:
            # attempt a fit
            temp_list = place_trial_in_list(temp_list, trial_set[0], proposal)
            # Check if that fit works for the rest of the trials
            temp_list = fit_trials_in_list(temp_list, trial_set[1:], level=level + 1)
            if len(temp_list) > 0:
                return temp_list  # it works so let's use it
            else:
                temp_list = deepcopy(working_list)  # it doesn't work so keep going

        # this should only happen if all proposals don't work for future trials
        return []

    elif len(proposal_inds) == 1:
        temp_list = deepcopy(working_list)
        # attempt a fit
        place_trial_in_list(temp_list, trial_set[0], proposal_inds[0])

        # Check if that fit works for the rest of the trials
        # But only if there are future trials
        if len(trial_set) > 1:
            temp_list = fit_trials_in_list(temp_list, trial_set[1:], level=level + 1)
            if len(temp_list) > 0:
                # it works so let's use it
                return temp_list
            else:
                # no good future fits
                return []
        # Only one trial left == nearly done!
        elif len(trial_set) == 1:
            logging.debug("Last trial placed!")
            return temp_list

    elif len(proposal_inds) < 1:
        # only happens if there are trials left (otherwise it would have returned in the ==1 condition)
        logging.debug(f"No good proposals; recursion level {level}")
        return []

    logging.critical("Don't know what happened here")
    return []


def complete_list_gen(trial_set: list, conditions: list, verbose=False) -> list[dict]:
    """
    Takes a trial set and turns it into a list of stimuli(+metadata) to present
    ------
    INPUTS
        trial_set: a list of dictionaries of trials that should be experienced
        conditions: a list of condition types(str) in order of most constrained to least constrained
    OUTPUT
        final_list: a list of trials in order to be presented
    """
    # first create null list
    trial_df = pd.DataFrame(trial_set)
    null_list = [None] * trial_df["reps"].sum()
    logging.debug(f"Null list is {len(null_list)} long")

    trial_df = trial_df.sample(frac=1)  # randomize the trials

    # sort list from most constrained to least constrained
    sorted_trials = []
    for condition in conditions:
        df = trial_df[trial_df["type"] == condition]
        sorted_trials += df.to_dict("records")

    logging.debug(f"sorted trials is {len(sorted_trials)} long")

    # run the fitting process
    final_list = fit_trials_in_list(null_list, sorted_trials)

    assert len(final_list) > 0, "Final list len=0; error in fit_trials_in_list"

    for trial in final_list:
        assert not trial == None, "At least one trial missing from final_list"

    if verbose:
        print(f"There are {len(final_list)} trials in the final study list")
        if verbose == "VERY":
            for trial in final_list:
                print(trial)

    return final_list


def target_lure_order(test_length: int, old_prop: float) -> tuple[list[bool], int, int]:
    """
    Create the order of old/new trials for the test
    ------
    INPUTS
        test_length: int; number of trials for the test
    OUTPUTs
        old_new_order: list of booleans; if true then item should be old, if false item should be new
        nOld: number of old items
        nNew: number of new items
    """
    # Make sure length and prop make sense

    if not float(test_length * old_prop).is_integer():
        logging.warning(f"Exact prop ({old_prop}) is not possible")
        old_items = np.round(test_length * old_prop, 0)
        old_prop = old_items / test_length
        logging.warning(f"Prop is corrected to:{old_prop}")

    n_old_items = int(test_length * old_prop)
    n_new_items = int(test_length - n_old_items)

    if not n_old_items / (n_old_items + n_new_items) == old_prop:
        logging.critical(
            f"Prop is not as expected; expected={old_prop}, actual={n_old_items/(n_old_items+n_new_items)}"
        )

    # generate list
    olds = [True] * n_old_items
    news = [False] * n_new_items
    old_new_order = olds + news
    # make it so that not all old items come first
    random.shuffle(old_new_order)

    return old_new_order, n_old_items, n_new_items


def get_old_trials(
    study_list: list[dict],
    time_blocks: int,
    nOld: int,
    pools: list,
    condition_types: list,
) -> list[dict]:
    """
    Get the old items for the test list

    Parameters
    ----------
    study_list: list of the items seen before
    time_blocks: number of blocks to counterbalance list (not used)
    nOld: number of old items to pick
    pools: image pools to counterbalance
    condition_types: conditions to counterbalance

    Returns
    -------
    old_trials: list of trial dictionaries for the test section
    """
    # get the unique trials in order
    unique_trials = []
    unique_ids = []
    for trial in study_list:
        if not trial["id"] in unique_ids:
            trial_copy = trial.copy()  # watch out for those references

            # This information is not applicable to test list and could confuse me later
            del trial_copy["repetition"]
            del trial_copy["location"]  # location is still stored in placement

            unique_trials.append(trial_copy)
            unique_ids.append(trial_copy["id"])

    time_block_len = int(len(unique_trials) / time_blocks)
    time_blocked_study_list = []
    for i in range(time_blocks):
        time_section_ind = i * time_block_len
        # get block
        try:
            # for all but the last index this should work
            loop_time_block = unique_trials[
                time_section_ind : time_section_ind + time_block_len
            ].copy()
        except IndexError:
            # the last index may need more flexibility
            loop_time_block = unique_trials[time_section_ind:].copy()
        # randomize within the block
        random.shuffle(loop_time_block)
        time_blocked_study_list.append(loop_time_block)

    # report the lengths of each grouping
    if not all(
        len(i) == len(time_blocked_study_list[0]) for i in time_blocked_study_list
    ):
        logging.warning(f"Not all blocks of old trials are equal")
        for i, group in enumerate(time_blocked_study_list):
            logging.warning(f"Study trials in the {i}th grouping: {len(group)}")
    else:
        for i, group in enumerate(time_blocked_study_list):
            logging.debug(f"Study trials in the {i}th grouping: {len(group)}")

    assert nOld <= len(
        unique_trials
    ), f"Not enough study trials: required={nOld} availible={len(unique_trials)}"

    old_trials = []
    # the following are used to sample evenly from each counterbalance condition
    time_section = 0
    pool = 0
    study_type = 0
    for i in range(nOld):
        old_pool = pools[pool]
        old_type = condition_types[study_type]
        found = False
        for ind, old_trial in enumerate(time_blocked_study_list[time_section]):
            if old_trial["pool"] == old_pool:
                if old_trial["type"] == old_type:
                    old_trials.append(old_trial)
                    time_blocked_study_list[time_section].pop(ind)
                    found = True
                    break
        assert found, "Could not counterbalance old trials"

        # iterate to the next block
        time_section += 1
        if time_section > len(time_blocked_study_list) - 1:
            time_section = 0
        # iterate to the next pool type
        pool += 1
        if pool > len(pools) - 1:
            pool = 0
        # iterate to the next study type
        study_type += 1
        if study_type > len(condition_types) - 1:
            study_type = 0

    if not study_type == 0:
        logging.warning("old items do not sample conditions evenly")
    if not pool == 0:
        logging.warning("old items do not sample pool evenly")
    if not time_section == 0:
        logging.warning("old study items do not sampled time_section")

    for i, remains in enumerate(time_blocked_study_list):
        logging.debug(f"Remaining study session for {i}th grouping: {len(remains)}")

    return old_trials


def create_lure_trials(pools: list[str], lure_types: list, nLures: int) -> list[dict]:
    """
    Create a list of lure trials.
    ------
    INPUTS
        pools: list of different stimulus conditions (NOTE: likely the same as the study session pools)
        lure_types: list of different lure types (NOTE: could be used for adding repeting lures)
    OUTPUT
        lure_trials: list of dictionaries with keys of pool, type, and id
    """
    # create the lure conditions
    # this could be pulled out like it is for study itesm
    lure_conditions = []
    for pool in pools:
        for lure_type in lure_types:
            lure_conditions.append(
                {
                    "pool": pool,
                    "type": lure_type,
                }
            )

    if nLures % len(lure_conditions) != 0:
        logging.warning(
            f"Number of trials will result in imperfect lure condition balancing: nLures({nLures}) mod nLureConditions({len(lure_conditions)}) = {nLures % len(lure_conditions)}"
        )

    j = 0  # tracks the conditions
    lure_trials = []
    for i in range(nLures):
        loop_lure = lure_conditions[j].copy()
        # a negative id is assigned so as to be different from the old items
        loop_lure["id"] = -1 * (i + 1)
        lure_trials.append(loop_lure.copy())
        j += 1
        if j > len(lure_conditions) - 1:
            j = 0

    # shuffle list of lures
    random.shuffle(lure_trials)

    return lure_trials


def create_test_list(
    old_new_order: list[bool],
    old_trials: list[dict],
    new_trials: list[dict],
    verbose=False,
) -> list[dict]:
    """
    Create a list for the test session
    ------
    INPUTS
        old_new_order: list describing if the item should be old
        old_trials: list of old trials for the test, should be length sum(old_new_order) and are in order
        new_trials: list of new items for the test, should be length len(old_new_order) - len(old_trials)
    OUTPUT
        test_list: a list of dictionaries in the order for an experiment
    """
    test_list = []
    for i, old in enumerate(old_new_order):
        if old:
            old_trial = old_trials[0].copy()
            old_trial["old"] = old
            old_trial["test_placement"] = i
            test_list.append(old_trial)
            old_trials.pop(0)
        else:
            new_trial = new_trials[0].copy()
            new_trial["old"] = old
            new_trial["test_placement"] = i
            test_list.append(new_trial)
            new_trials.pop(0)

    if verbose:
        print(f"There are {len(test_list)} test trials")
        if verbose == "VERY":
            for trial in test_list:
                print(trial)

    return test_list


def make_study_block(
    pools: list[str],
    condition_types: list,
    rep_types: list,
    distance_types: list,
    ntrials: int,
    test_length: int,
    old_prop: float,
    time_blocks: int,
    lure_types: list,
) -> dict[str, list]:
    """
    Make a study block with all these params
    ------
    INPUTS
        lots
    OUTPUT
        block_dict: dictionary of list of dictionaries
    """
    conditions = create_conditions(pools, condition_types, rep_types, distance_types)

    trial_set = create_trial_set(ntrials, conditions)

    study_list = complete_list_gen(trial_set, ["spaced-rep", "massed-rep", "1p"])

    old_new_order, nOld, nNew = target_lure_order(test_length, old_prop)

    old_trials = get_old_trials(study_list, time_blocks, nOld, pools, condition_types)

    new_trials = create_lure_trials(pools, lure_types, nNew)

    test_list = create_test_list(old_new_order, old_trials, new_trials)

    block_dict = {
        "study": study_list,
        "test": test_list,
    }

    return block_dict


# block_dict = make_study_block(POOLS, CONDITION_TYPES, REP_TYPES, DISTANCE_TYPES, NTRIALS, TEST_LENGTH, OLD_PROP, TIME_BLOCKS, LURE_TYPES)


def read_all_images(filename_dict: dict) -> dict[str, str]:
    """
    Read in image file names for each pool provided and return a dictionary
    Note: some issues may come up with different csvs
    ------
    INPUTS
        filename_dict: dictionary of filenames; each pool has a name, filename (key, value)
    OUTPUT
        image_dict: dictionary of lists; one key for each condition, the keys match those in filename_dict
    """
    image_dict = {}
    for pool, filename in filename_dict.items():
        # create a dictionary reader
        loop_reader = csv.DictReader(open(filename, "r"))
        # read in all the lines into a list of dicts
        # we only care about the filename
        # NOTE: this could change if the csv structure changes
        loop_list = [l["filename"] for l in loop_reader]

        # shuffle the list of images
        random.shuffle(loop_list)
        # add them to the dictionary
        image_dict[pool] = loop_list

    return image_dict


# image_dict = read_all_images(FILENAME_DICT)


def add_images(block_dict, image_dict):
    """
    Add images to the block dictionary. Removes images from the image dict so as to prevent using images for multiple trials

    ------
    INPUTS
        block_dict: dictionary with test and study (or whatever) keys with lists of trials with block unique ids for pictures
        image_dict: dictionary with key for each pool, lists of all images not used
    OUTPUT
        block_dict: the same input with added image_filename key added to each trial
        image_dict: the same input with all images used removed
    """

    # Get all unique trial ids
    ids_for_images = []
    for trial_set in block_dict.values():
        for trial in trial_set:
            if not trial["id"] in ids_for_images:
                ids_for_images.append((trial["id"], trial["pool"]))

    # get ids sorted into image pools
    pool_ids = {}
    for pool in image_dict:
        # ids_for_images has trials with the trial id at 0, and pool type at 1
        pool_ids[pool] = [trial[0] for trial in ids_for_images if trial[1] == pool]

    # check to make sure we have enough items
    for pool, image_list in pool_ids.items():
        logging.debug(
            f"{pool} ids; required={len(image_list)} > available={len(image_dict[pool])}"
        )
        assert len(image_list) <= len(
            image_dict[pool]
        ), f"Not enough images for {pool} ids; required={len(image_list)} > available={len(image_dict[pool])}"

    # pair image file names and ids
    id_imagefile = {}
    for pool, trial_ids in pool_ids.items():
        # for each pool add images to all trials
        for id in trial_ids:
            # making the image id pair
            id_imagefile[id] = image_dict[pool][0]
            # remove the image so it can't be used again
            image_dict[pool].pop(0)

    for pool in image_dict:
        logging.debug(f"{pool} images left: {len(image_dict[pool])}")

    # add image_filename to block lists
    for trial_list in block_dict.values():
        # for each trial list (ie study, test)
        for trial in trial_list:
            trial["image_filename"] = id_imagefile[trial["id"]]

    return block_dict, image_dict


def create_experiment(block_params, nBlocks, nSubjects, filename_dict):
    """
    Take in params and create a list of subjects,
        for each subject a list of blocks,
            for each block a dictionary of study and test
                both of which are lists of dictionaries of trials
    """
    experiment_dict = {}

    for sub in range(nSubjects):
        logging.debug(f"completing subject {sub}")

        # each subject should use one set of images for all their blocks
        sub_images = read_all_images(filename_dict)
        sub_blocks = {}
        for block in range(nBlocks):
            logging.debug(f"completing block {block}")
            # # Setting a seed for debuging
            # random.seed("Please work like I think you do")
            block_dict = make_study_block(**block_params)
            block_dict, sub_images = add_images(block_dict, sub_images)
            sub_blocks["block_" + str(block)] = block_dict

        experiment_dict["subj_" + str(sub)] = sub_blocks

    return experiment_dict
