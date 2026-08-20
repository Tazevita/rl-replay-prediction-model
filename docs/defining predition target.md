# Defining the Prediction Target

The prediction target defines what player intent the model should predict from the current and recent game state. Each player perspective should receive one clear intent label describing their likely behavior over a fixed future period, such as challenging, rotating, supporting, or possessing.

The intent categories, prediction horizon, and labeling rules must be consistent across every replay. Labels should be reviewed on sample replays before processing the full dataset, since the model will learn whatever behavior those rules define.
