**TOPAZ: TOpologically-based Parameter inference for Agent-based model
optimization**

**[Overview]{.underline}**

Overall, this code has both the full code used to run the simulations
and pipeline as seen in the paper as well as two simpler Jupyter
notebooks that go through a shorter example of the full code. Please
read this ReadMe file fully to understand what each file does.

Each iteration of this code has three big sections: the simulation or
topological data analysis (TDA) step, the approximate Bayesian
computation step (ABC), and the Bayesian information criterion step
(BIC). These represent the three main steps of the TOPAZ algorithm, and
each step has been broken up into smaller sub steps if needed.

Within the Full\_Project\_Code section, most of the files have been
optimized to run on the HPC. Even on the HPC, the first four steps of
the code can take 1-2 days each to run. All of the other code here takes
at most an hour to run but most only up to a few minutes.

**[Full\_Project\_Code]{.underline}**

Note: Most of this code is designed to be run using High-Performance
Computing (HPC). Each .sh file is designed to run on the HPC. Some are
fully functioning on their own whereas others run a corresponding .py
file.

**Contents**

Full\_CLW\_Simulation -- contains code that generates a full simulation
of all C, L, and W combinations and uses that information to create the
3D and 2D t-SNE plots

ChangeW\_00 -- contains code that runs the simulation scenario where the
ground truth value for W=0 (No Alignment) but ABC simulations include W
(Alignment Model)

ChangeW\_05 -- contains code that runs the simulation scenario where the
ground truth value for W=0.05 (With Alignment) and the ABC simulations
include W (Alignment Model)

NoChangeW\_00 -- contains code that runs the simulation scenario where
the ground truth value for W=0 (No Alignment) and ABC simulations do not
include W (W=0 always, D'Orsogna Model)

NoChangeW\_05 -- contains code that runs the simulation scenario where
the ground truth value for W=0.05 (With Alignment) but ABC simulations
do not include W (W=0 always, D'Orsogna Model)

copy\_crocker\_plots.sh -- extracts the Crocker plots generated for each
chosen simulation combination for all four simulation types
(ChangeW\_00, ChangeW\_05, NoChangeW\_00, NoChangeW\_05) for an easier
comparison

copy\_folders.sh -- extracts the files of the chosen parameter
combinations for all four simulation types from all parameter
combinations so easier to compare

get\_results.py -- collects all of the important information from each
chosen parameter combination simulation runs and puts it all together in
a csv file

submit\_results.sh -- runs get\_results.py on the HPC

**Contents Within Each "Change" Folder -- run in this order**

S01\_simulate\_grid.py -- runs the ground-truth simulations for each C,
L, and W combination for the respective W value (run on HPC using
submitS01.sh file)

S02\_crockerplot\_save.py - calculates the Crocker plots for each
ground-truth simulation above (run on HPC using submitS02.sh file)

ABC\_S01\_samples.py -- runs the 10,000 random ABC simulations (run on
HPC using submit1.sh file)

ABC\_S02\_crockerplot\_save.py -- calculates the Crocker plots for each
of the 10,000 simulations (run on HPC using submit2.sh file)

ABC\_S03-crockerplot\_distance.py -- calculates the sample losses
between the 10,000 simulations and each ground-truth simulation of your
choosing (we chose 10 examples for each W=0 and W=0.05) (run on HPC
using submit3.sh file)

ABC\_S04\_p01\_tolerance.py -- calculates the ABC-estimated median
values and posterior density plots for each of the chosen simulations
(run on HPC using submit4.sh file)

ABC\_S05\_process\_sim.py -- runs the simulation for the ABC-estimated
median values (run on HPC using submit5.sh file)

ABC\_S06\_process\_crocker.py -- calculates the Crocker plot for the
above ABC-estimated median value simulations (run on HPC using
submit6.sh file)

bic.py -- calculates the BIC score for each of the chosen simulations
(run on HPC using submitBIC.sh file)

ic\_vec.npy -- initial conditions needed for ABM simulations

Scripts -- folder that contains behind-the-scenes files to use
throughout the coding process (explained below)

**Contents Within Scripts**

Alignment.py -- adds the alignment features to extend the D'Orsogna
model

arrow\_head\_marker.py -- Used in the Full\_CLW\_Simulation below to
help create the arrow plots

DorsognaNondim\_Align\_w00.py -- Runs the D'Orsogna model where W=0
always (D'Orsogna model)

DorsognaNondim\_Align\_w05.py -- Runs the D'Orsogna model (in name only)
where W=0.05 always (Alignment model)

DorsognaNondim\_Align.py -- Runs the D'Orsogna model where W can be any
inputted value (for ABC Alignment model simulations)

filtering\_df.py -- Helps filter the simulation before running the
Crocker plots

crocker.py -- Calculates the Crocker matrices and creates the Crocker
plots

No longer needed but included files -- Dorsogna\_fluidization.py,
Dorsogna.py, DorsognaNondim.py, Helper\_Practice.py,
parallel\_ABM\_simulate.py, parallel\_crocker.py,
parallel\_parameter\_estimation.py, simplex\_PE.py

**Contents Within Full\_CLW\_Simulation**

last\_frame\_params.csv -- Contains which C, L, W combinations to create
arrow plots for

S01\_simulate\_grid\_allW.py -- runs the ground-truth simulations for
each C, L, and W combination for every possible W value (run on HPC
using submitS01.sh file)

S02\_crockerplot\_save.py - calculates the Crocker plots for each
ground-truth simulation above (run on HPC using submitS02.sh file)

S03\_crockerplot\_tsneplot\_gray\_and\_color.py -- Creates the 2D and 3D
t-SNE plots for the full C, L, W simulation grid (run in terminal)

ProcessResults\_allFrames.ipynb -- creates the arrow plots for all
frames of the simulation for chosen C, L, W combinations

ProcessResults\_lastFrames.ipynb -- creates the arrow plots for the last
frame of the simulation for chosen C, L, W combinations

Scripts -- same files as above

Not Needed -- two different files that also run PCA and t-SNE but not
used

**[Notebook\_Tutorials]{.underline}**

**Ready\_To\_Use\_Example**

TOPAZ\_CLW.ipynb -- Jupyter notebook containing a smaller,
self-contained version of the simulation. It runs one TDA simulation,
takes in 30 ABC samples, and then calculates one BIC score

sample\_30 -- contains the data frames and Crocker matrices for 30
random ABC samples

Modules\_CLW -- contains most of the files listed above to run the full
simulation process just modified for this smaller scale

Scripts -- the same script files as above

Example\_created\_code -- contains the results of running through the
process one time

ic\_vec.npy -- the same initial conditions as used above

mega\_w\_pic\_dark -- figure of the TOPAZ pipeline shown in the notebook

**Insert\_Own\_ABM\_Example**

TOPAZ\_General.ipynb -- Jupyter notebook containing a framework for you
to insert your own ABM simulation results into the TOPAZ pipeline

Modules\_General -- contains the files included in the general pipeline
while hiding some of the files that are to be replaced with the new ABM
simulation

mega\_w\_pic\_dark -- figure of the TOPAZ pipeline shown in the notebook
