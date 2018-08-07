# Neural Network KSOM based Visual Motor Coordination (VMC)
This work is just implementation of a part of paper "A hybrid neural control scheme for visual-motor coordination"
L Behera, N Kirubanandan - IEEE control systems, 1999 which was given to us as a course project by our Professor Dr. L Behera

Here's what is happening..
We have a 3-link robot manipulator and we want that we just tell it the coordinate where we want to reach and the algorithm gives the
link angles required so that robotic arm end effector reach that position. The problem is easy if you have the inverse kinematics
relation but here we do not have that. For 3-link manipulator you may be able to find one but as the degrees of freedom increase it gets
tougher and tougher to find the inverse kinematics hence what we are trying to do is we are making our KSOM to learn the inverse kinematics.
How it is being achieved, I'll upload the concept soon but right now I'm uploading the code.

Training Takes time hence I have also uploaded workspace after training.

You just download the following files to one folder
------------------------------------------------------------------------------------------------------

fwd_kin.m

plotUpdate.m

VMC_data_27_Apr_9_36_AM_10000_samples.mat  <-----This is the workspace after training

VMC_KSOM_running.m   <---------------------------This is the main file

------------------------------------------------------------------------------------------------------

After downloading, open VMC_KSOM_running.m in matlab and go to the heading %% Complicated Trajectory
and start running the code from there. It will automatically load the
workspace form VMC_data_27_Apr_9_36_AM_10000_samples.mat file and start running.

If you get any problem feel free to ask....
If you have any suggestion feel free to comment as i'm just a student :-)




P.S: This is not my original work. It is just implementation of a technique suggested in the paper I mentioned on the top. For coding
the concept is take from the following book: Intelligent Systems and Control: Principles and Applications written by
Dr.Laxmidhar Behera and Dr.Indrani Kar
