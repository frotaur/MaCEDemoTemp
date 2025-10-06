"""
    Just a DEFAULTS dictionary with optional default parameters for each automaton.
    When an automaton is loaded, these parameters will be added in addition to the
    required (h,w) size, and device parameters.
"""


DEFAULTS = {
    "Lenia" : {
        "interest_files": "./demo_data/demo_lenia",
        "save_dir": "./data/lenia_saved_params"
    },
    "FlowLenia" : {
        "interest_files": "./demo_data/demo_macelenia",
        "save_dir": "./data/flowlenia_saved_params"
    },
    "MaCELenia" :{
        "interest_files": "./demo_data/demo_macelenia",
        "save_dir": "./data/macelenia_saved_params"
    },
    "MaCELeniaXChan" :{
        "interest_files": "./demo_data/demo_macelenia_xchan",
        "save_dir": "./data/macelenia_xchan_saved_params"
    },
    "MaCENCA" : {
        "saved_folder": "./demo_data/demo_macenca",
    }
}