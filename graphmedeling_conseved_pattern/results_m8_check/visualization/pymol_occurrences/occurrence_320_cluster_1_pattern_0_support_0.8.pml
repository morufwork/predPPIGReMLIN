load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7sy0.ent", occ_320_c1_p0_s0.8
hide everything, occ_320_c1_p0_s0.8
show cartoon, occ_320_c1_p0_s0.8 and chain B+E
color palegreen, occ_320_c1_p0_s0.8 and chain B
color lightblue, occ_320_c1_p0_s0.8 and chain E
select hotspot_source, occ_320_c1_p0_s0.8 and ((chain B and resi 500))
select hotspot_target, occ_320_c1_p0_s0.8 and ((chain E and resi 41))
select hotspot_all, occ_320_c1_p0_s0.8 and ((chain B and resi 500) or (chain E and resi 41))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_320_c1_p0_s0.8 and chain B+E
set_name hotspot_all, hotspot_occurrence_320
set_name hotspot_source, hotspot_source_320
set_name hotspot_target, hotspot_target_320
bg_color white
# patternId=0 support=0.8 graphId=207
