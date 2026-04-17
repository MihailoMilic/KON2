hole 030


face 1 badly merged into few
possibly because of the bulb
[FIX]: hardcode this color of green as boundry


[EASY-FIX] 
hole 37 bs on the right


[FIXED-WITH-BACKGROUND-CYCLES]
hole 054 one vertex not shown
culprit: [MERGE] stage=7c-nearby-merge | face 29: vid 47 @ [ 568. 1101.] -> 42 @ [ 563.75 1100.25]

hole 066 - 37 and 137, 

083 completely broken - fixed
see why grayscale breaks, 

hole 090 same problem as 068

[FIXED-WITH-BACKGROUND-CYCLES]
face 068 and 090 problem is that there is no way for our algorithm to tell something is a triangle. 
[NEED-T0-FIGURE-OUT]: how to distinguish triangles from quads, one method is to look at pair of vertices of a face and their edge, extend it in its orthogonal direction and check if it is going into another face. if enough of these rays are going into the background than the face must be a triangle with a corner facing the background.
fixed with hsv case set to 148 instead of 150