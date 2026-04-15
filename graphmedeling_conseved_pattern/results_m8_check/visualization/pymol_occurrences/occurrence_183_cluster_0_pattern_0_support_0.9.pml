load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7wpa.ent", occ_183_c0_p0_s0.9
hide everything, occ_183_c0_p0_s0.9
show cartoon, occ_183_c0_p0_s0.9 and chain A+D
color palegreen, occ_183_c0_p0_s0.9 and chain A
color lightblue, occ_183_c0_p0_s0.9 and chain D
select hotspot_source, occ_183_c0_p0_s0.9 and ((chain A and resi 452))
select hotspot_target, occ_183_c0_p0_s0.9 and ((chain D and resi 34))
select hotspot_all, occ_183_c0_p0_s0.9 and ((chain A and resi 452) or (chain D and resi 34))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_183_c0_p0_s0.9 and chain A+D
set_name hotspot_all, hotspot_occurrence_183
set_name hotspot_source, hotspot_source_183
set_name hotspot_target, hotspot_target_183
bg_color white
# patternId=0 support=0.9 graphId=286
