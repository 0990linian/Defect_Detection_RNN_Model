################################################################################
# Explanation of the pogo-field animation and pogo-hist data.
################################################################################

"""

There are two kinds of waves, longitudinal wave and shear wave.

The main output we would like to get from the ultrasonic emitter is the 
longitudinal wave.  However, there will always be some inevitable shear wave, 
due to possible errors in the experimental equipment, or something else 
[possible reference].  

The shear wave magnitude is normally significantly small comparing to the 
longitudinal wave magnitude.  Therefore, it should be straight forward to 
neglect.

On the field animation, longitudinal wave is the string yellow wave that comes 
first.  The light blue wave followed by is the shear wave.

We only care about the longitudinal wave, and since shear wave is quite far from 
it, it should be quite easy for us to tell the difference.

The Young's modulus 210e9 is the default Young's modulus of the material.  If we 
change Ecirc value to other values, that's the time we define a circular 
inclusion into the material.

By changing it to different values, we can see the field animation generating 
different wave forms.  The smaller the value, the easier we could see the 
existence of the circle.


The blue line of pogo-hist data marks the data for emitter, the red line marks 
the receiver.  We should only care about the red line data.
"""
