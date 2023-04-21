#The proposed numerical calculation of the CCPP optimum generation capacity in python. Using a neural network model, this Python code calculates the optimal Combined Cycle Power Plant (CCPP) generation capacity. The input data, including temperature, evaporation, atmospheric pressure, and relative humidity, are initially scaled using predefined scaling factors and offsets. The neural network comprises two perceptron layers with weights and biases, which compute the outputs of the first layer and the final production. The output is subsequently unscaled using an unscaling factor and offset. The optimal CCPP power generation is calculated and displayed based on the provided input values, resulting in a power generation of at least 462 MW.

import numpy as np

# Scaling values
temp_scale = 7.452469826
temp_offset = 19.65119934
ev_scale = 12.70790005
ev_offset = 54.30580139
ap_scale = 5.938789845
ap_offset = 1013.26001
rh_scale = 14.60029984
rh_offset = 73.30899811

# Perceptron weights and biases
p1_w0 = np.array([0.593324, 0.00657032, 0.0933036, 0.0310159])
p1_b0 = 0.688402
p1_w1 = np.array([0.288555, 0.204765, -0.144645, 0.070106])
p1_b1 = -0.110117
p2_w0 = np.array([-1.34001, -1.13091])
p2_b0 = 0.582504

# Unscaling factors
us_scale = 17.06699944
us_offset = 454.3649902

# Function to scale the input data
def scale_input_data(temp, ev, ap, rh):
    scaled_temp = (temp - temp_offset) / temp_scale
    scaled_ev = (ev - ev_offset) / ev_scale
    scaled_ap = (ap - ap_offset) / ap_scale
    scaled_rh = (rh - rh_offset) / rh_scale
    return scaled_temp, scaled_ev, scaled_ap, scaled_rh

# Function to unscale the output data
def unscale_output_data(output):
    unscaled_output = output * us_scale + us_offset
    return unscaled_output

# Function to calculate the output
def neural_network_output(temp, ev, ap, rh):
    # Scale the input data
    scaled_temp, scaled_ev, scaled_ap, scaled_rh = scale_input_data(temp, ev, ap, rh)

    # Outputs of the first perceptron layer
    p1_output0 = np.tanh(p1_b0 + np.dot(np.array([scaled_temp, scaled_ev, scaled_ap, scaled_rh]), p1_w0))
    p1_output1 = np.tanh(p1_b1 + np.dot(np.array([scaled_temp, scaled_ev, scaled_ap, scaled_rh]), p1_w1))

    # Output of the second perceptron layer
    p2_output0 = p2_b0 + np.dot(np.array([p1_output0, p1_output1]), p2_w0)

    # Unscale the output data
    unscaled_output = unscale_output_data(p2_output0)

    return unscaled_output

# Optimal values (Power Generation	≥ 462 [MW])
temp = 19.4
ev = 25.4
ap = 1021.4
rh = 60.8

plant_output = neural_network_output(temp, ev, ap, rh)
print(f"The CCPP power generation at optimal operating condition [MW]: {plant_output:.2f}")
