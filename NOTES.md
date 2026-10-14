
## Session 1 (9/29/26)
- Setup done, session 1 recorded (me_s1)
- Next: record session 2 w/ different lighting and test the results again
## Session 2 (10/01/26)
- Recorded session 2 (me_s2)
- Baseline accuracy training on session 1 and testing on session 2: ~65%
- X -> O and I -> D have a suspicious number of errors - rerecording may be required for these at some point
- Normalization in the code being used: 
	- Step 1: Change x and y values so that even if data is recorded with multiple cameras with multiple unique aspect rations
	- Step 2: Make all data's origin value at the wrist so superficial changes in position matter less than actual changes in hand shape
	- Step 3: Define the palm as one unit long to account for differences in palm size between samples
	- Step 4: Account for differences in handedness by flipping all left-handed samples
	- Rotation normalization is not performed because some letters are distinguished by rotation
## Session 3 (10/12/26)
- Had Frank record session 3 (frank_s1)
- Created the model training pipeline (train.py), which uses a standard MLP on 63 inputs with 2 hidden layers, which feed into 24 output nodes representing the 24 letters accounted for in the data
![Confusion matrix for the current version of the model (MLP trained on 3 sessions of data)](/home/william/Downloads/asl-tutor-project/confusion_matrix_s3.png) 
- At present, the mean accuracy is at 52.4% against a baseline of 62.1%. This is quite low, and more granular changes (possibly to the model architecture) will need to be made if this number does not improve after adding more diverse data
## Session 4 (10/14/26)
- Scaffolded the web page with Vite. It now runs locally, and shows the user their camera and their hand with the skeleton of landmarks on it. The coordinates of these landmarks are recorded and ready to be used.
- The next step is to integrate the model with the webpage using the data already being collected
- Additionally, a few more sessions of data collection will be necessary
