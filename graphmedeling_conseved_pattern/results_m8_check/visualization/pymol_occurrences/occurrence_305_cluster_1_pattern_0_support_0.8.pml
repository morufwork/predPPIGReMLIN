load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7dhx.ent", occ_305_c1_p0_s0.8
hide everything, occ_305_c1_p0_s0.8
show cartoon, occ_305_c1_p0_s0.8 and chain A+B
color palegreen, occ_305_c1_p0_s0.8 and chain A
color lightblue, occ_305_c1_p0_s0.8 and chain B
select hotspot_source, occ_305_c1_p0_s0.8 and ((chain A and resi 41))
select hotspot_target, occ_305_c1_p0_s0.8 and ((chain B and resi 500))
select hotspot_all, occ_305_c1_p0_s0.8 and ((chain A and resi 41) or (chain B and resi 500))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_305_c1_p0_s0.8 and chain A+B
set_name hotspot_all, hotspot_occurrence_305
set_name hotspot_source, hotspot_source_305
set_name hotspot_target, hotspot_target_305
bg_color white
# patternId=0 support=0.8 graphId=35
