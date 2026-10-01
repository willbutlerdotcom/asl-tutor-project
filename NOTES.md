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
	- Rotation normalization is not performed because some letters are distinguished from one another through hand rotation

