CityStreet: Multi-View Crowd Counting Dataset
Copyright (c) 2019
Qi Zhang, Antoni B. Chan
City University of Hong Kong

The scene map is used for visualization. The scene map resolution is h*w = 461*438. For visualization, the scene map coordinate should be consistent with ground-plane density map (resolution h*w = 192*160). 
First resize the scene map GP_scene_map to  h*w = 412*392, denoted as GP_scene_map;
Then, increase dmap's resolution by 2 times (h*w = 384*320), denoted as GT_dmap_i.
Finally, pad the dmap using following code:

	GT_dmap_padded = np.zeros((GP_scene_map.shape[0], GP_scene_map.shape[1]))
	w0 = GT_dmap_i.shape[1]
	h0 = GT_dmap_i.shape[0]
	dx = 36
	dy = 5
	GT_dmap_padded[2*dy:2*dy+h0, 2*dx:2*dx+w0] = GT_dmap_i.copy()
