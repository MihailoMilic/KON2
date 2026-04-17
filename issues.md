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

[WARNING] dilation radius for adjacancy is now 7 because of hole 054

1. ok 
2. ok
3. top and bottom bulbs being overlapped by other vertices, add small 2px merge?
4. ok
5. ok
6. ok
7. ok
8. same problem as 3
9. same problem, small 2px merge only on these left over nodes (not cycles or edges)
10. ok
11.ok
12. ok
13. ok
14. ok
15. ok
16. ok
17. ok
18. ok
19. ok
20. ok
21. ok
22. ok
23. ok, strange same shape as 3,8,9
24. again same problem, same shape
25. ok
26. ok
27. ok
28. ok
29. ok
30. edge between 10 and 6 is unsaturated and should have gotten another vertex
31. ok
32. ok
33.ok
34. ok
35. ok
36. ok
37. ok
38. ok
39. ok
40. ok
41. same problem as 3,8,9...
42. ok
43. ok
44. ok
45. ok
46. ok
47. ok
48. ok
49.ok
50. ok 
51. ok
52. ok
53. ok
54. ok
55. ok
56. ok
57. ok
58. ok
59. ok
60. ok
61. ok
62. ok
63. ok
64. ok
65. ok
66. same problem as 3,8,9.. also 139 should have not been assigned
67.