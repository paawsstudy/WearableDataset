def map_label_to_intensity(label):
    # hard-coded value of all label
    label_to_met = {
            "In_Position_Kneeling" : 1.3,
            "In_Position_Reclining/Slouching" : 1.3,
            "In_Position_Sitting" : 1.3,
            "Lying_On_Back" : 1.3,
            "Lying_On_Left_Side" : 1.3,
            "Lying_On_Right_Side" : 1.3,
            "Lying_On_Stomach" : 1.3,
            "Bathing_Pet" : 3.3,
            "Cleaning" : 3.3,
            "Cooking/Prepping_Food" : 2.0,
            "Cycling_Regular_Bicycle" : 3.5,
            "Dancing" : 3.0,
            "Doing_Alternative_Exercise/Therapy/Stretching" : 2.5,
            "Doing_Common_Housework_Light" : 2.0,
            "Doing_Common_Housework" : 2.3,
            "Doing_Dishes" : 1.8,
            "Doing_Home_Repair_Light" : 2.0,
            "Doing_Home_Repair" : 3.0,
            "Doing_Laundry" : 2.0,
            "Doing_Martial_Arts" : 5.3,
            "Doing_Resistance_Training" : 4.5,
            "Dressing/Undressing" : 2.5,
            "Eating/Dining" : 1.5,
            "Exercising_Gym_Other" : 5.0,
            "Gardening" : 2.3,
            "Grooming/Self_Care" : 2.0,
            "In_Transit_Driving_Car" : 2.5,
            "In_Transit_Passive_Car" : 1.3,
            "In_Transit_Passive_Train/Bus/Plane" : 1.3,
            "Loading/Unloading_Vehicle" : 3.5,
            "Meeting_Social_Gathering_In_Person" : 1.5,
            "Meeting_Formal_Gathering_In_Person" : 1.3,
            "Packing/Unpacking_Something" : 2.0,
            "Playing_Exergame" : 7.2,
            "Playing_Musical_Instrument" : 2.0,
            "Playing_Sports/Games" : 4.0,
            "Playing_Videogame" : 1.0,
            "Playing_With_Children_Light" : 1.5,
            "Playing_With_Pets" : 3.0,
            "Relaxing_Passive" : 1.8,
            "Riding_Elevator_Down" : 1.3,
            "Riding_Elevator_Up" : 1.3,
            "Riding_Escalator_Down" : 1.3,
            "Riding_Escalator_Up" : 1.3,
            "Shopping_Grocery" : 2.5,
            "Shopping_Non-Grocery" : 2.3,
            "Using_Computer/Screen" : 1.3,
            "Walking_Pet" : 3.5,
            "Watching_Lecture" : 1.3,
            "WatchingTV/Movies" : 1.3,
            "Working_Other" : 2.0,
            "Applying_Makeup" : 2.0,
            "Bathing" : 1.5,
            "Blowdrying_Hair" : 2.5,
            "Brushing_Teeth" : 2.0,
            "Brushing/Combing/Tying_Hair" : 2.5,
            "Cycling_Active_Pedaling_Regular_Bicycle" : 3.5,
            "Cycling_Active_Pedaling_Stationary_Bike" : 3.5,
            "Doing_Resistance_Training_Free_Weights" : 4.5,
            "Doing_Resistance_Training_Other" : 4.5,
            "Dry_Mopping" : 3.3,
            "Dusting" : 2.3,
            "Flossing_Teeth" : 2.0,
            "Folding_Clothes" : 2.0,
            "Ironing" : 1.8,
            "Kneeling_Still" : 1.3,
            "Kneeling_With_Movement" : 2.0,
            "Loading/Unloading_Washing_Machine/Dryer" : 2.0,
            "Lying_Still" : 1.3,
            "Lying_With_Movement" : 1.3,
            "Organizing_Shelf/Cabinet" : 3.5,
            "Playing_Frisbee" : 3.0,
            "Puttering_Around" : 2.0,
            "Putting_Clothes_Away" : 2.3,
            "Running_Treadmill" : 9.0,
            "Showering" : 2.0,
            "Shoveling_Mud_Snow" : 6.0,
            "Sitting_Still" : 1.3,
            "Sitting_With_Movement" : 1.3,
            "Standing_Still" : 1.3,
            "Standing_With_Movement" : 1.8,
            "Sweeping" : 3.3,
            "Vacuuming" : 3.3,
            "Walking_Down_Stairs" : 3.5,
            "Walking_Fast" : 5.0,
            "Walking_Slow" : 2.8,
            "Walking_Treadmill" : 4.3,
            "Walking_Up_Stairs" : 4.0,
            "Walking" : 3.5,
            "Washing_Face" : 2.0,
            "Washing_Hands" : 1.8,
            "Watering_Plants" : 2.5,
            "Wet_Mopping" : 3.0,
            'Stationary_Biking_300_Lab':4.0,
            'Treadmill_2mph_Lab':2.5,
            'Treadmill_3mph_Conversation_Lab':3.8,
            'Treadmill_3mph_Drink_Lab':3.8,
            'Treadmill_3mph_Free_Walk_Lab':3.8,
            'Treadmill_3mph_Briefcase_Lab':3.8,
            'Treadmill_3mph_Hands_Pockets_Lab':3.8,
            'Treadmill_3mph_Phone_Lab':3.8,
            'Treadmill_4mph_Lab':5.5,
            'Treadmill_5.5mph_Lab':8.5,
            'Stand_Shelf_Unload_Lab':3.5,
            'Stand_Shelf_Load_Lab':3.5,
            'Chopping_Food_Lab':2.0,
            'Washing_Dishes_Lab':2.0,
            'Lying_On_Left_Side_Lab':1.0,
            'Lying_On_Stomach_Lab':1.0,
            'Stand_Conversation_Lab':1.5,
            'Arm_Curls_Lab':2.8,
            'Push_Up_Modified_Lab':3.5,
            'Push_Up_Lab':3.5,
            'Ab_Crunches_Lab':2.8    
        }
    colname = ['POSTURE', 'PA_TYPE', 'HIGH_LEVEL_BEHAVIOR', 'CONTEXTUAL_PARAMETERS']
    met_value = []
    for _, row in label.iterrows():
        max_val = 0
        for c in colname:
            label_value = str(row[c])
            for act in label_to_met:
                if act in label_value:
                    max_val = max(max_val, label_to_met[act])
        met_value.append(max_val)
    label['GROUND_TRUTH'] = ['N/A' if met == 0 else 'SB' if met <=1.5 else 'LPA' if met <= 3 else 'MPA' if met <= 6 else 'VPA' for met in met_value]
    return label

# example mapping
def map_label_to_posture(label):
    posture_mapper = {
            'In_Position_Reclining/Slouching':'recline',
            'In_Position_Sitting':'sit',
            'In_Position_Upright':'upright',
            'Lying_On_Back':'lie',
            'Lying_On_Left_Side':'lie',
            'Lying_On_Right_Side':'lie',
            'Lying_On_Stomach':'lie'
        }
    label['GROUND_TRUTH'] = [posture_mapper.get(pos, 'N/A') for pos in label['POSTURE']]
    return label
    
# example mapping
def map_label_to_pa_type(label):
    pa_mapper = {
            'Cycling_Active_Pedaling_Regular_Bicycle':'cycling',
            'Cycling_Active_Pedaling_Stationary_Bike':'cycling',
            'Lying_Still': 'sedentary_lie',
            'Lying_With_Movement':'sedentary_lie',
            'Running_Non-Treadmill':'walking',
            'Running_Treadmill':'walking_jogging',
            'Sitting_Still':'sedentary_sit',
            'Sitting_With_Movement':'sedentary_sit',
            'Standing_Still':'standing',
            'Standing_With_Movement':'standing',
            'Walking_Down_Stairs':'walking_stairs_down',
            'Walking_Fast':'walking_fast',
            'Walking_Slow':'walking_slow',
            'Walking_Treadmill':'walking',
            'Walking_Up_Stairs':'walking_stairs_up',
            'Walking':'walking',
            'Lying_On_Back_Lab':'sedentary_lie',
            'Stationary_Biking_300_Lab':'cycling',
            'Treadmill_2mph_Lab':'walking_slow',
            'Treadmill_3mph_Conversation_Lab':'walking',
            'Treadmill_3mph_Drink_Lab':'walking',
            'Treadmill_3mph_Briefcase_Lab':'walking',
            'Treadmill_3mph_Hands_Pockets_Lab':'walking',
            'Treadmill_3mph_Phone_Lab':'walking',
            'Treadmill_4mph_Lab':'walking_fast',
            'Treadmill_5.5mph_Lab':'walking_jogging',
            'Chopping_Food_Lab':'household_cooking_chopping_food',
            'Washing_Dishes_Lab':'household_dishes',
            'Lying_On_Left_Side_Lab':'sedentary_lie',
            'Lying_On_Stomach_Lab':'sedentary_lie',
            'Lying_On_Right_Side_Lab':'sedentary_lie',
            'Sit_Recline_Talk_Lab':'sedentary_sit',
            'Stand_Conversation_Lab':'standing',
            'Sit_Recline_Web_Browse_Lab':'sedentary_sit',
            'Sit_Writing_Lab':'sedentary_sit',
            'Sit_Typing_Lab':'sedentary_sit',
            'Folding_Clothes':'household_laundry_folding_clothes',
            'Loading/Unloading_Washing_Machine/Dryer':'household_laundry_loading_machine',
            'Organizing_Shelf/Cabinet':'household_organizing',
            'Putting_Clothes_Away':'household_laundry_putting_clothes_away',
            'Sweeping':'household_sweeping',
            'Wet_Mopping':'household_mopping',
            'Ironing': 'household_laundry_ironing',
            'Vacuuming':'household_vacuuming',
            'Stand_Shelf_Unload_Lab':'household_organizing',
            'Stand_Shelf_Load_Lab':'household_organizing'
        }
    label['GROUND_TRUTH'] =  [pa_mapper.get(pa, 'N/A') for pa in label['PA_TYPE']]
    return label

# example mapping
def map_label_to_high_level_pa_type(label):
    label = map_label_to_pa_type(label)
    import pandas as pd
    label['GROUND_TRUTH'] = [str(x).split('_')[0] if not pd.isna(x) else 'N/A' for x in label]
    return label