CityStreet: Multi-View Crowd Counting Dataset
Copyright (c) 2019
Qi Zhang, Antoni B. Chan
City University of Hong Kong

The input of the image2world projection is the camera view coordinate (x, y, height):
0<x<2704,
0<y<1520,
height is calculated through searching among (1560, 1980, 10)mm.

The coordinate range (x, y) in the label jsons is 0~2704 and 0~1520;
if the coordinate (x, y) is from input image (0~676, 0~380), resize it to the original image resolution:
x = x*4,
y = y*4.

Then use the Image_to_World projection function to get the world coordinate wc.

After we get the world coordinate wc, it should be further normalized through:

    bbox = [352*0.8, 522*0.8]
    image_size = [380, 676]
    resolution_scaler = 76.25
    view1_wc_offset = view1_wc / resolution_scaler
    view1_wc_offset_x = view1_wc_offset[:, 0:1] + bbox[0]  # * resolution_scaler
    view1_wc_offset_y = view1_wc_offset[:, 1:2] + bbox[1]  #

(view_wc_offset_x, view1_wc_offset_y) is the final world coordinate and should be in the range of (h=768, w=640).

The world2image projection will be the reverse of the above process.
