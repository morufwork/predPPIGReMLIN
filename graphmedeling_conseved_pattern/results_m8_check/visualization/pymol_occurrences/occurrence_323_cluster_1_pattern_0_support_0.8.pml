load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7v84.ent", occ_323_c1_p0_s0.8
hide everything, occ_323_c1_p0_s0.8
show cartoon, occ_323_c1_p0_s0.8 and chain A+F
color palegreen, occ_323_c1_p0_s0.8 and chain A
color lightblue, occ_323_c1_p0_s0.8 and chain F
select hotspot_source, occ_323_c1_p0_s0.8 and ((chain A and resi 500))
select hotspot_target, occ_323_c1_p0_s0.8 and ((chain F and resi 41))
select hotspot_all, occ_323_c1_p0_s0.8 and ((chain A and resi 500) or (chain F and resi 41))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_323_c1_p0_s0.8 and chain A+F
set_name hotspot_all, hotspot_occurrence_323
set_name hotspot_source, hotspot_source_323
set_name hotspot_target, hotspot_target_323
bg_color white
# patternId=0 support=0.8 graphId=230
