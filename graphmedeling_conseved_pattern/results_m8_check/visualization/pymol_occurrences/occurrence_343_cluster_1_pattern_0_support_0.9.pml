load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7ekc.ent", occ_343_c1_p0_s0.9
hide everything, occ_343_c1_p0_s0.9
show cartoon, occ_343_c1_p0_s0.9 and chain A+B
color palegreen, occ_343_c1_p0_s0.9 and chain A
color lightblue, occ_343_c1_p0_s0.9 and chain B
select hotspot_source, occ_343_c1_p0_s0.9 and ((chain A and resi 41))
select hotspot_target, occ_343_c1_p0_s0.9 and ((chain B and resi 500))
select hotspot_all, occ_343_c1_p0_s0.9 and ((chain A and resi 41) or (chain B and resi 500))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_343_c1_p0_s0.9 and chain A+B
set_name hotspot_all, hotspot_occurrence_343
set_name hotspot_source, hotspot_source_343
set_name hotspot_target, hotspot_target_343
bg_color white
# patternId=0 support=0.9 graphId=91
