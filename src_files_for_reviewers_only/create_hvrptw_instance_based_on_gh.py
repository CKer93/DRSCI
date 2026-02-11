#####
# Template .py script that generates an HVRPTW from a VRPTW instance and custom heterogeneous fleet rules
#####


# Import general Python packages and functions
import os
# Import custom/internal Python packages and functions
from core.vrp import VRP_OBJECT

# specify the INPUT dataset: 
instance_name = "C1_2_1"
updated_file_name = "C1_2_1_HFVRPTW"

# Set the input and output directories to transform a VRPTW instance into an HVRPTW instance (both are in the end INPUT data files)
local_path_to_instances = f"~gehring_homberger/instance_json"
path_to_save_json = f"~kerscher_et_al_hfvrptw/instance_json"


# Custom rules to setting up a heterogeneous fleet rules.
fleet_scenario = "f_v_even" # Options: ["v_dominant", "f_dominant", "f_v_even"]
fix_var_multiplier = 1
no_of_vehicle_types = 3
avail_ratio = None # NOTE: uniform availability of all larger vehicle types is assumed
sum_avail_large_veh = 0.8
capa_interval_str = "double"
capa_cost_ratio = 0.9
xs_fleet_vail = "pend" # Options: ["pend", "bks_homo", "min_buffer"]

instance = VRP_OBJECT(
    name=instance_name, type="VRPTW"
).load_instance(os.path.join(local_path_to_instances, f"{instance_name.lower()}.json"),read_json=True)
instance.setup_heterogeneous_fleet(
    no_of_vehicle_types,
    sum_avail_large_veh,
    avail_ratio,
    xs_fleet_vail,
    0.1, # vehicle avail buffer
    capa_interval_str,
    capa_cost_ratio,
    fleet_scenario,
    fix_var_multiplier,
)
instance.name = updated_file_name
instance.write_to_json(path_to_save_json, init_file_name=updated_file_name)