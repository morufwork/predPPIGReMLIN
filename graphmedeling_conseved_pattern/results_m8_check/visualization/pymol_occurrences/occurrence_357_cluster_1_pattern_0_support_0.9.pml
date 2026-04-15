load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7v84.ent", occ_357_c1_p0_s0.9
hide everything, occ_357_c1_p0_s0.9
show cartoon, occ_357_c1_p0_s0.9 and chain A+F
color palegreen, occ_357_c1_p0_s0.9 and chain A
color lightblue, occ_357_c1_p0_s0.9 and chain F
select hotspot_source, occ_357_c1_p0_s0.9 and ((chain A and resi 500))
select hotspot_target, occ_357_c1_p0_s0.9 and ((chain F and resi 41))
select hotspot_all, occ_357_c1_p0_s0.9 and ((chain A and resi 500) or (chain F and resi 41))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_357_c1_p0_s0.9 and chain A+F
set_name hotspot_all, hotspot_occurrence_357
set_name hotspot_source, hotspot_source_357
set_name hotspot_target, hotspot_target_357
bg_color white
# patternId=0 support=0.9 graphId=230
