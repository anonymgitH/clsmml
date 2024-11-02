
from estemated_direction import estemated_direction
from estemated_direction import estemated_direction_2
from read_weights import read_last_weights

from extract_tubingen_truth import extract_tubingen_truth
from audrc import calculate_AUDRC
from accuracy import calculate_accuracy

## Run Tuegingen Dataset
file_path_truth= '~/Truth'
file_path = '~/Tuebingen'
ground_truth_direction=extract_tubingen_truth(file_path_truth)
estimated_direction, confidence_estimated,t=estemated_direction(file_path)

sumsum = sum([1 if gt == est else 0 for gt, est in zip(
    [item[1] for item in ground_truth_direction], 
    estimated_direction
)])
estimator_value = [1 if gt[1] == est[1] else 0 for gt, est in zip(([item[1] for item in ground_truth_direction] ), estimated_direction)]
audrc = calculate_AUDRC( ([item[2] for item in ground_truth_direction]),estimator_value,confidence_estimated)
# weights_pairs=read_last_weights(file_path_)
acc= calculate_accuracy(([item[1] for item in ground_truth_direction]),estimated_direction,weights_pairs)

## Run Dataset  SIM/SIM-c/SIM-ln/SIM-G

# file_path_SIM = '~/SIM-ln'
# estimated_direction, confidence_estimated,ground_truth=estemated_direction(file_path_SIM)
# estimator_value = [1 if est == '->' else 0 for est in estimated_direction]
# audrc_sim = calculate_AUDRC(ground_truth,estimator_value,confidence_estimated)
# weights_pairs = [1] * len(estimated_direction)
# acc= calculate_accuracy(ground_truth,estimator_value,weights_pairs)

## Run Dataset  AN/AN-s/LS/LS-s/MN-U
 
# file_path_AN = '~/LS'
# estimated_direction, confidence_estimated,ground_truth=estemated_direction(file_path_AN)
# estimator_value = [1 if est == '->' else 0 for est in estimated_direction]
# audrc_an = calculate_AUDRC(ground_truth,estimator_value,confidence_estimated)
# weights_pairs = [1] * len(estimated_direction)
# acc= calculate_accuracy(ground_truth,estimator_value,weights_pairs)

## Run Dataset  Multi/Net/Cha

# file_path_Multi = '~/Multi'
# estimated_direction, confidence_estimated,ground_truth=estemated_direction_2(file_path_Multi)
# estimator_value = [1 if est == '->' else -1 for est in estimated_direction]
# acc= calculate_accuracy(ground_truth,estimator_value,weights_pairs
# weights_pairs = [1] * len(estimated_direction)
# audrc_multi = calculate_AUDRC(ground_truth,estimator_value,confidence_estimated)





