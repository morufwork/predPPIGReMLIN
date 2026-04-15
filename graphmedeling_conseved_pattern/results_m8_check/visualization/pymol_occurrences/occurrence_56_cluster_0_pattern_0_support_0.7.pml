load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7wse.ent", occ_56_c0_p0_s0.7
hide everything, occ_56_c0_p0_s0.7
show cartoon, occ_56_c0_p0_s0.7 and chain A+B
color palegreen, occ_56_c0_p0_s0.7 and chain A
color lightblue, occ_56_c0_p0_s0.7 and chain B
select hotspot_source, occ_56_c0_p0_s0.7 and ((chain A and resi 79))
select hotspot_target, occ_56_c0_p0_s0.7 and ((chain B and resi 486))
select hotspot_all, occ_56_c0_p0_s0.7 and ((chain A and resi 79) or (chain B and resi 486))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_56_c0_p0_s0.7 and chain A+B
set_name hotspot_all, hotspot_occurrence_56
set_name hotspot_source, hotspot_source_56
set_name hotspot_target, hotspot_target_56
bg_color white
# patternId=0 support=0.7 graphId=319
